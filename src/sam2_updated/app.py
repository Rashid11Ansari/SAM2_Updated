"""Web layer: Gradio UI mounted in FastAPI, started by `uv run sam2-updated`.

Data folder layout (--data-dir, default ./data):
    videos/       input videos -- copy them here (scp) or upload in the browser
    outputs/      every export is written here (one folder + zip per export)
    checkpoints/  SAM2 weights, downloaded on first start (override with --checkpoint-dir)
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI

from . import model

# Gradio re-encodes our mp4v preview for the browser and warns every time -- expected, not an error.
warnings.filterwarnings("ignore", message="Video does not have browser-compatible container")

LOAD_LABEL = "Load video"
PROPAGATE_LABEL = "Propagate across video"
POINT_HIT_RADIUS = 15     # px -- clicking this close to an existing point deletes it
FRAME_CACHE_SIZE = 64     # decoded frames kept in RAM for fast stepping
UI_UPDATE_EVERY = 0.25    # seconds between progress updates on the buttons

# Left/right arrow keys step one frame (ignored while typing in a text/number field).
KEYBOARD_JS = """
() => {
  document.addEventListener('keydown', (e) => {
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    const id = e.key === 'ArrowLeft' ? 'prev-frame-btn' : e.key === 'ArrowRight' ? 'next-frame-btn' : null;
    if (!id) return;
    const btn = document.getElementById(id);
    if (btn && btn.offsetParent !== null) { e.preventDefault(); btn.click(); }
  });
}
"""
CSS = """
#frame-nav-box { padding: 6px 8px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
#frame-nav { align-items: center; gap: 8px; background: transparent; }
#prev-frame-btn, #next-frame-btn { min-width: 44px !important; max-width: 52px; height: 44px; font-size: 18px; align-self: center;
  border: 1px solid var(--border-color-primary); background: var(--button-secondary-background-fill); border-radius: 8px; }
#prev-frame-btn:hover, #next-frame-btn:hover { background: var(--button-secondary-background-fill-hover); }
"""


def _fresh_session() -> dict:
    return {
        "video_path": None,
        "frames_dir": None,
        "frame_names": [],
        "fps": 30.0,
        "inference_state": None,
        "points": {},          # (frame_idx, obj_id) -> {"points": [...], "labels": [...], "box": [...] | None}
        "pending_box": None,   # first corner clicked in Box mode
        "cur_frame_idx": 0,
        "cur_obj_id": 1,
        "obj_names": {1: "Object 1"},
        "video_segments": {},  # frame_idx -> {obj_id: mask} from the last propagation
        "action_history": [],  # what Undo pops: {"frame_idx", "obj_id", "type": "point"|"box"}
        "stop_requested": False,
    }


def build_ui(predictor, device: torch.device, data_dir: Path) -> gr.Blocks:
    """All UI state lives in this closure -- nothing global."""
    video_dir = data_dir / "videos"
    output_dir = data_dir / "outputs"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = model.find_ffmpeg()

    S = _fresh_session()
    frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------ helpers
    def push_history(f, o, kind):
        S["action_history"].append({"frame_idx": f, "obj_id": o, "type": kind})

    def discard_history(f, o, kind):
        for i in range(len(S["action_history"]) - 1, -1, -1):
            h = S["action_history"][i]
            if (h["frame_idx"], h["obj_id"], h["type"]) == (f, o, kind):
                S["action_history"].pop(i)
                return

    def frame_rgb(idx) -> np.ndarray:
        img = frame_cache.get(idx)
        if img is None:
            img = model.read_rgb(S["frames_dir"], S["frame_names"][idx])
            frame_cache[idx] = img
            if len(frame_cache) > FRAME_CACHE_SIZE:
                frame_cache.popitem(last=False)
        else:
            frame_cache.move_to_end(idx)
        return img.astype(np.float32)

    def obj_name(oid):
        return S["obj_names"].get(oid) or f"Object {oid}"

    def any_prompts():
        return any(v["points"] or v.get("box") for v in S["points"].values())

    def legend_html():
        if not any_prompts():
            return f"_No prompts yet -- click the image to add a point for **{obj_name(S['cur_obj_id'])}**, or draw a box._"
        rows = []
        for oid in sorted({o for (_, o) in S["points"]} | {S["cur_obj_id"]}):
            r, g, b = model.color_for(oid).astype(int)
            n_pts = sum(len(v["points"]) for k, v in S["points"].items() if k[1] == oid)
            has_box = any(v.get("box") for k, v in S["points"].items() if k[1] == oid)
            parts = ([f"{n_pts} point(s)"] if n_pts else []) + (["box"] if has_box else [])
            current = " <b>(current)</b>" if oid == S["cur_obj_id"] else ""
            rows.append(f'<span style="color: rgb({r},{g},{b}); font-size:16px;">&#9679;</span> '
                        f'{obj_name(oid)} -- {" + ".join(parts) or "no prompts yet"}{current}')
        return "<br>".join(rows)

    def prompts_on(idx):
        return {o: e for (f, o), e in S["points"].items() if f == idx}

    def frame_with_prompts(idx):
        return model.draw_prompts(frame_rgb(idx), prompts_on(idx))

    def render_current():
        idx = S["cur_frame_idx"]
        img = frame_with_prompts(idx)
        seg = S["video_segments"].get(idx)
        if seg:
            model.overlay_masks(img, seg, alpha=0.4)
        return img.astype(np.uint8)

    def recompute(idx, oid, entry):
        return model.add_prompts(predictor, S["inference_state"], device, idx, oid,
                                 entry["points"], entry["labels"], entry.get("box"))

    def image_with_masks(idx, masks):
        return model.overlay_masks(frame_with_prompts(idx), masks).astype(np.uint8)

    def video_choices():
        return model.list_videos(video_dir)

    # ------------------------------------------------------------ data folder
    def refresh_videos(selected=None):
        choices = video_choices()
        value = selected if selected in choices else (choices[0] if choices else None)
        return gr.update(choices=choices, value=value)

    def on_upload(uploaded):
        """Save a browser upload into data/videos so it is kept on disk, and select it."""
        if not uploaded:
            return gr.update(), "No file uploaded."
        src = Path(uploaded)
        dest = video_dir / src.name
        if dest.exists() and dest.stat().st_size != src.stat().st_size:
            dest = video_dir / f"{src.stem}_{datetime.now():%Y%m%d-%H%M%S}{src.suffix}"
        if not dest.exists():
            shutil.copy(src, dest)
        return refresh_videos(dest.name), f"Saved upload to {dest}. Click '{LOAD_LABEL}'."

    # ------------------------------------------------------------ stop
    def request_stop():
        S["stop_requested"] = True
        return gr.update(value="Stopping...", interactive=False)

    # ------------------------------------------------------------ load
    def label_controls(visible):
        v = gr.update(visible=visible)
        return {nav_group: v, prev_btn: v, next_btn: v, frame_slider: v, label_type: v, prompt_mode: v,
                obj_name_box: v, legend_md: v, new_obj_btn: v, undo_btn: v,
                export_overlay: v, export_masks: v, export_json: v}

    def load_video(video_name):
        def busy(text):
            return {load_btn: gr.update(value=text, interactive=False),
                    stop_load_btn: gr.update(visible=True, value="Stop", interactive=True)}

        def idle():
            return {load_btn: gr.update(value=LOAD_LABEL, interactive=True), stop_load_btn: gr.update(visible=False)}

        if not video_name:
            yield {**idle(), status: f"Pick a video from the list (or upload one) first. Videos folder: {video_dir}"}
            return
        video_path = video_dir / video_name
        if not video_path.exists():
            yield {**idle(), status: f"{video_path} not found -- click Refresh."}
            return

        S["stop_requested"] = False
        frames_dir = Path(tempfile.mkdtemp(prefix="frames_"))
        try:
            fps, expected = model.video_info(video_path)
            yield {**busy("Extracting frames... 0%"), status: f"Extracting frames from {video_name}..."}
            proc = model.start_frame_extraction(video_path, frames_dir, ffmpeg)
            while proc.poll() is None:
                if S["stop_requested"]:
                    proc.kill()
                    proc.wait()
                    shutil.rmtree(frames_dir, ignore_errors=True)
                    yield {**idle(), status: "Loading stopped."}
                    return
                done = len(list(frames_dir.iterdir()))
                yield busy(f"Extracting frames... {min(99, int(100 * done / expected))}%" if expected
                           else f"Extracting frames... {done}")
                time.sleep(UI_UPDATE_EVERY)
            if proc.returncode != 0:
                raise RuntimeError("ffmpeg could not read this video: " + proc.stderr.read().decode(errors="ignore")[-300:])
            frame_names = model.list_frames(frames_dir)
            if not frame_names:
                raise RuntimeError("No frames could be extracted from this video.")

            yield {**busy(f"Preparing SAM2 ({len(frame_names)} frames)..."), status: "Initializing SAM2 tracking session..."}
            old_frames_dir = S.get("frames_dir")
            inference_state = model.init_session(predictor, frames_dir, device)
            if S["stop_requested"]:
                del inference_state
                shutil.rmtree(frames_dir, ignore_errors=True)
                yield {**idle(), status: "Loading stopped."}
                return

            S.clear()
            S.update(_fresh_session())
            S.update({"video_path": video_path, "frames_dir": frames_dir, "frame_names": frame_names,
                      "fps": fps, "inference_state": inference_state})
            frame_cache.clear()
            if old_frames_dir:
                shutil.rmtree(old_frames_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            n = len(frame_names)
            yield {
                **idle(), **label_controls(True),
                image_display: gr.update(value=frame_rgb(0).astype(np.uint8), visible=True),
                frame_slider: gr.update(maximum=max(0, n - 1), value=0, visible=True, label=f"Frame  (0 - {n - 1})"),
                obj_name_box: gr.update(value=obj_name(1), visible=True),
                legend_md: gr.update(value=legend_html(), visible=True),
                propagate_btn: gr.update(visible=True, interactive=False, value=PROPAGATE_LABEL),
                export_btn: gr.update(visible=True, interactive=False),
                video_info_md: gr.update(value=f"**{video_name}** &middot; {n} frames &middot; {fps:.1f} fps &middot; {n / fps:.1f}s",
                                         visible=True),
                status: "Loaded. Name your first object (optional), then click the image to add points -- or switch to Box mode.",
                tabs: gr.Tabs(selected=1),
            }
        except Exception as e:  # keep the UI usable and show the reason
            shutil.rmtree(frames_dir, ignore_errors=True)
            yield {**idle(), status: f"Error while loading: {e}"}

    # ------------------------------------------------------------ navigation
    def show_frame(idx):
        n = len(S["frame_names"])
        if n == 0:
            return {}
        idx = max(0, min(int(idx), n - 1))
        S["cur_frame_idx"] = idx
        S["pending_box"] = None
        return {image_display: render_current(), frame_slider: gr.update(value=idx)}

    def scrub_frame(new_idx):
        # Only the picture goes back while dragging -- sending the slider value too would fight the drag.
        return show_frame(new_idx).get(image_display, gr.update())

    def step_frame(delta):
        return show_frame(S["cur_frame_idx"] + delta)

    def warm():
        if S["inference_state"] is not None:
            model.warm_features(predictor, S["inference_state"], device, S["cur_frame_idx"])

    # ------------------------------------------------------------ objects
    def rename_object(name):
        S["obj_names"][S["cur_obj_id"]] = (name or "").strip() or f"Object {S['cur_obj_id']}"
        return legend_html()

    def new_object():
        S["cur_obj_id"] = max([S["cur_obj_id"], *S["obj_names"]]) + 1
        oid = S["cur_obj_id"]
        S["obj_names"].setdefault(oid, f"Object {oid}")
        S["pending_box"] = None
        return f"Now placing prompts for {obj_name(oid)}. Type a name above if you like.", legend_html(), gr.update(value=obj_name(oid))

    # ------------------------------------------------------------ clicks / undo
    def click_impl(prompt_mode, label_type, evt):
        if S["frames_dir"] is None:
            return None, "Load a video first.", legend_html(), gr.update(interactive=False)
        x, y = evt.index
        idx, oid = S["cur_frame_idx"], S["cur_obj_id"]
        name, key = obj_name(oid), (idx, oid)

        if prompt_mode == "Box":
            if S["pending_box"] is None:
                S["pending_box"] = (x, y)
                img = frame_with_prompts(idx)
                cv2.drawMarker(img, (int(x), int(y)), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
                return img.astype(np.uint8), "Box mode: click the opposite corner to finish the box.", legend_html(), gr.update(interactive=any_prompts())
            x0, y0 = S["pending_box"]
            S["pending_box"] = None
            entry = S["points"].setdefault(key, {"points": [], "labels": [], "box": None})
            entry["box"] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
            push_history(idx, oid, "box")
            return image_with_masks(idx, recompute(idx, oid, entry)), f"{name}: box set on frame {idx}.", legend_html(), gr.update(interactive=True)

        entry = S["points"].get(key)
        if entry and entry["points"]:
            hits = [i for i, (px, py) in enumerate(entry["points"]) if ((px - x) ** 2 + (py - y) ** 2) ** 0.5 <= POINT_HIT_RADIUS]
            if hits:
                entry["points"].pop(hits[0])
                entry["labels"].pop(hits[0])
                discard_history(idx, oid, "point")
                if entry["points"] or entry.get("box"):
                    img = image_with_masks(idx, recompute(idx, oid, entry))
                    msg = f"Removed a point from {name}. {len(entry['points'])} point(s) remain."
                else:
                    del S["points"][key]
                    img = frame_with_prompts(idx).astype(np.uint8)
                    msg = f"Removed the only prompt for {name} on frame {idx}. Click to redefine it."
                return img, msg, legend_html(), gr.update(interactive=any_prompts())

        entry = S["points"].setdefault(key, {"points": [], "labels": [], "box": None})
        entry["points"].append([x, y])
        entry["labels"].append(1 if label_type == "foreground" else 0)
        push_history(idx, oid, "point")
        return (image_with_masks(idx, recompute(idx, oid, entry)),
                f"{name}: {len(entry['points'])} point(s) placed on frame {idx}.", legend_html(), gr.update(interactive=True))

    def on_image_click(prompt_mode, label_type, evt: gr.SelectData):
        t0 = time.time()
        img, msg, legend, prop = click_impl(prompt_mode, label_type, evt)
        if img is not None and "opposite corner" not in msg:
            msg = f"{msg}  ({time.time() - t0:.2f}s)"
        return img, msg, legend, prop

    def undo_point():
        # Pops the most recent action across every object/frame and jumps to where it happened.
        while S["action_history"]:
            a = S["action_history"].pop()
            f, oid, kind = a["frame_idx"], a["obj_id"], a["type"]
            entry = S["points"].get((f, oid))
            if not entry:
                continue
            if kind == "point" and entry["points"]:
                entry["points"].pop()
                entry["labels"].pop()
            elif kind == "box" and entry.get("box"):
                entry["box"] = None
            else:
                continue
            S.update(cur_frame_idx=f, cur_obj_id=oid, pending_box=None)
            name = obj_name(oid)
            if entry["points"] or entry.get("box"):
                img = image_with_masks(f, recompute(f, oid, entry))
                msg = f"Undid last prompt. {len(entry['points'])} point(s) remain for {name} on frame {f}."
            else:
                del S["points"][(f, oid)]
                img = render_current()
                msg = f"Removed the last prompt for {name} on frame {f}. Add a new point or box to redefine it."
            return img, msg, legend_html(), gr.update(interactive=any_prompts()), gr.update(value=f), gr.update(value=name)
        return render_current(), "Nothing left to undo.", legend_html(), gr.update(interactive=any_prompts()), gr.update(), gr.update()

    # ------------------------------------------------------------ propagate
    def propagate():
        def busy(text):
            return {propagate_btn: gr.update(value=text, interactive=False),
                    stop_prop_btn: gr.update(visible=True, value="Stop", interactive=True)}

        def idle(can_run=True):
            return {propagate_btn: gr.update(value=PROPAGATE_LABEL, interactive=can_run), stop_prop_btn: gr.update(visible=False)}

        if S["inference_state"] is None:
            yield {**idle(False), status: "Load a video first."}
            return
        if not any_prompts():
            yield {**idle(False), status: "Add at least one point or box before propagating."}
            return

        S["stop_requested"] = False
        total = len(S["frame_names"])
        segments, stopped = {}, False
        try:
            yield {**busy(f"Propagating 0/{total}"), status: "Propagating masks through the video..."}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = last_ui = time.time()
            for count, (f_idx, masks) in enumerate(model.propagate(predictor, S["inference_state"], device), start=1):
                segments[f_idx] = masks
                if S["stop_requested"]:
                    stopped = True
                    break
                if time.time() - last_ui > UI_UPDATE_EVERY:
                    last_ui = time.time()
                    yield busy(f"Propagating {count}/{total}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            secs = time.time() - t_start
            S["video_segments"] = segments

            if stopped:
                S["stop_requested"] = False
                yield {**idle(), **show_frame(S["cur_frame_idx"]), export_btn: gr.update(interactive=bool(segments)),
                       status: f"Stopped after {len(segments)}/{total} frames -- those masks are kept. "
                               "Check them on the Label tab, add corrections, and propagate again, or export the partial result."}
                return

            preview_path = Path(tempfile.mkdtemp(prefix="preview_")) / "preview.mp4"
            last_ui = time.time()
            for done, n in model.iter_overlay_video(S["frames_dir"], S["frame_names"], segments, S["fps"], preview_path):
                if S["stop_requested"]:
                    S["stop_requested"] = False
                    yield {**idle(), **show_frame(S["cur_frame_idx"]), export_btn: gr.update(interactive=True),
                           status: f"Propagated all {total} frames; preview skipped. Export is ready."}
                    return
                if time.time() - last_ui > UI_UPDATE_EVERY:
                    last_ui = time.time()
                    yield busy(f"Rendering preview {done}/{n}")
            fps_done = len(segments) / secs if secs > 0 else 0.0
            yield {**idle(), **show_frame(S["cur_frame_idx"]), preview_video: str(preview_path),
                   export_btn: gr.update(interactive=True), tabs: gr.Tabs(selected=2),
                   status: f"Propagated across {len(segments)} frames in {secs:.1f}s ({fps_done:.1f} fps). "
                           "Ready to export -- or go back to the Label tab, check for drift on another frame, add a correction, and propagate again."}
        except Exception as e:
            S["stop_requested"] = False
            yield {**idle(), status: f"Error while propagating: {e}"}

    # ------------------------------------------------------------ export
    def export(do_overlay, do_masks, do_json, progress=gr.Progress()):
        segments = S["video_segments"]
        if not segments:
            return None, "Propagate first before exporting."
        stem = Path(S["video_path"]).stem
        out = output_dir / f"{stem}_{datetime.now():%Y%m%d-%H%M%S}"
        out.mkdir(parents=True, exist_ok=True)
        names = {oid: obj_name(oid) for oid in S["obj_names"]}
        if do_overlay:
            model.write_overlay_video(S["frames_dir"], S["frame_names"], segments, S["fps"], out / "labeled_video.mp4",
                                      progress=lambda i, n: progress(0.6 * i / n, desc=f"Writing overlay video {i}/{n}"))
        if do_masks:
            progress(0.7, desc="Writing PNG masks...")
            model.write_png_masks(segments, names, out / "masks")
        if do_json:
            progress(0.85, desc="Building COCO JSON...")
            model.write_coco_json(segments, names, Path(S["video_path"]).name, S["fps"], len(S["frame_names"]),
                                  out / "annotations.json")
        progress(1.0, desc="Zipping export...")
        zip_path = shutil.make_archive(str(out), "zip", out)
        return zip_path, f"Export saved on the server: {out}  (zip: {zip_path}) -- download below."

    # ------------------------------------------------------------ reset
    def reset_session():
        old_frames = S.get("frames_dir")
        S.clear()
        S.update(_fresh_session())
        frame_cache.clear()
        if old_frames:
            shutil.rmtree(old_frames, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {**label_controls(False), image_display: None, frame_slider: gr.update(value=0, visible=False),
                propagate_btn: gr.update(visible=False, interactive=False, value=PROPAGATE_LABEL),
                export_btn: gr.update(visible=False, interactive=False), video_info_md: gr.update(visible=False),
                status: "Session reset -- GPU memory freed. Pick a video to start again.", tabs: gr.Tabs(selected=0)}

    # ------------------------------------------------------------ layout
    with gr.Blocks(title="SAM2 Point Labeling") as demo:
        gr.Markdown("# SAM2 Point-Prompt Video Labeling")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Load")
                with gr.Row():
                    video_select = gr.Dropdown(choices=video_choices(), value=(video_choices() or [None])[0],
                                               label="Video", info=f"from {video_dir}", scale=4)
                    refresh_btn = gr.Button("Refresh", scale=1, min_width=80)
                with gr.Row():
                    load_btn = gr.Button(LOAD_LABEL, variant="primary", scale=3)
                    stop_load_btn = gr.Button("Stop", variant="stop", visible=False, scale=1)
                reset_btn = gr.Button("Reset session")
                status = gr.Textbox(label="Status", interactive=False)
                video_info_md = gr.Markdown(visible=False)

                gr.Markdown("### 2. Points")
                with gr.Group(elem_id="frame-nav-box", visible=False) as nav_group:
                    with gr.Row(elem_id="frame-nav", equal_height=True):
                        prev_btn = gr.Button("<", visible=False, scale=0, min_width=44, elem_id="prev-frame-btn")
                        frame_slider = gr.Slider(0, 1, step=1, value=0, label="Frame", visible=False, scale=8)
                        next_btn = gr.Button(">", visible=False, scale=0, min_width=44, elem_id="next-frame-btn")
                label_type = gr.Radio(["foreground", "background"], value="foreground", label="Point type", visible=False)
                prompt_mode = gr.Radio(["Point", "Box"], value="Point", label="Prompt mode", visible=False)
                obj_name_box = gr.Textbox(label="Current object name", value="Object 1", visible=False,
                                          placeholder="e.g. Fish 1", max_lines=1)
                legend_md = gr.Markdown(visible=False)
                with gr.Row():
                    new_obj_btn = gr.Button("New object", visible=False)
                    undo_btn = gr.Button("Undo last prompt", visible=False)

                gr.Markdown("### 3. Propagate")
                with gr.Row():
                    propagate_btn = gr.Button(PROPAGATE_LABEL, variant="primary", visible=False, interactive=False, scale=3)
                    stop_prop_btn = gr.Button("Stop", variant="stop", visible=False, scale=1)

                gr.Markdown("### 4. Export")
                export_overlay = gr.Checkbox(value=True, label="Overlay video", visible=False)
                export_masks = gr.Checkbox(value=True, label="PNG masks", visible=False)
                export_json = gr.Checkbox(value=True, label="COCO JSON", visible=False)
                export_btn = gr.Button("Export", visible=False, interactive=False)

            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
                    with gr.Tab("1. Upload", id=0):
                        gr.Markdown(f"Pick a video on the left -- or upload one here; it is saved to `{video_dir}`.")
                        video_upload = gr.File(label="Upload video", file_types=["video"])
                    with gr.Tab("2. Label", id=1):
                        # JPEG display: much smaller than PNG; SAM2 and the exports use the original frames.
                        image_display = gr.Image(label="Click to add points, or draw a box  (arrow keys: previous / next frame)",
                                                 interactive=False, format="jpeg")
                    with gr.Tab("3. Preview & download", id=2):
                        preview_video = gr.Video(label="Preview")
                        download = gr.File(label="Download")

        all_outputs = [load_btn, stop_load_btn, reset_btn, status, video_info_md,
                       nav_group, prev_btn, frame_slider, next_btn, label_type, prompt_mode, obj_name_box, legend_md,
                       new_obj_btn, undo_btn, propagate_btn, stop_prop_btn,
                       export_overlay, export_masks, export_json, export_btn,
                       image_display, preview_video, download, tabs]
        nav_outputs = [image_display, frame_slider]
        GPU = dict(concurrency_id="gpu", concurrency_limit=1)   # SAM2 calls never run at the same time

        refresh_btn.click(refresh_videos, inputs=[video_select], outputs=[video_select], show_progress="hidden")
        video_upload.upload(on_upload, inputs=[video_upload], outputs=[video_select, status])
        load_btn.click(load_video, inputs=[video_select], outputs=all_outputs, show_progress="hidden", **GPU)
        stop_load_btn.click(request_stop, outputs=[stop_load_btn], queue=False, show_progress="hidden")

        frame_slider.input(scrub_frame, inputs=[frame_slider], outputs=[image_display], show_progress="hidden",
                           trigger_mode="always_last").then(warm, show_progress="hidden", **GPU)
        prev_btn.click(lambda: step_frame(-1), outputs=nav_outputs, show_progress="hidden", trigger_mode="multiple"
                       ).then(warm, show_progress="hidden", **GPU)
        next_btn.click(lambda: step_frame(+1), outputs=nav_outputs, show_progress="hidden", trigger_mode="multiple"
                       ).then(warm, show_progress="hidden", **GPU)

        image_display.select(on_image_click, inputs=[prompt_mode, label_type],
                             outputs=[image_display, status, legend_md, propagate_btn], show_progress="hidden", **GPU)
        obj_name_box.change(rename_object, inputs=[obj_name_box], outputs=[legend_md], show_progress="hidden")
        new_obj_btn.click(new_object, outputs=[status, legend_md, obj_name_box])
        undo_btn.click(undo_point, outputs=[image_display, status, legend_md, propagate_btn, frame_slider, obj_name_box], **GPU)

        propagate_btn.click(propagate, outputs=all_outputs, show_progress="hidden", **GPU)
        stop_prop_btn.click(request_stop, outputs=[stop_prop_btn], queue=False, show_progress="hidden")

        export_btn.click(export, inputs=[export_overlay, export_masks, export_json], outputs=[download, status])
        reset_btn.click(reset_session, outputs=all_outputs)

    return demo


def create_app(data_dir: Path, model_size: str, checkpoint_dir: Path, compile_model: bool = False) -> FastAPI:
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    device = model.pick_device()
    print("Device:", model.describe_device(device), flush=True)
    predictor = model.load_predictor(model_size, checkpoint_dir, device, compile_model)
    print(f"SAM2 video predictor ({model_size}) loaded on {device}", flush=True)
    print(f"Data folder: {data_dir}", flush=True)

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"device": str(device), "model": model_size, "data_dir": str(data_dir)}

    demo = build_ui(predictor, device, data_dir)
    return gr.mount_gradio_app(app, demo, path="/", allowed_paths=[str(data_dir)], js=KEYBOARD_JS, css=CSS)


def main():
    p = argparse.ArgumentParser(description="SAM2 point-prompt video labeling web app")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = only this machine / via SSH tunnel (default)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="where videos are read from and exports written to (default: ./data)")
    p.add_argument("--model", default="large", choices=list(model.MODEL_SIZES),
                   help="SAM2 size (default: large; use small/tiny without an NVIDIA GPU)")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="where the SAM2 weights are stored (default: <data-dir>/checkpoints)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model -- NVIDIA + Linux only; slow first run, faster tracking after")
    args = p.parse_args()

    data_dir = args.data_dir.expanduser()
    checkpoint_dir = (args.checkpoint_dir or data_dir / "checkpoints").expanduser()
    app = create_app(data_dir, args.model, checkpoint_dir, args.compile)
    print(f"\nOpen http://localhost:{args.port} in your browser", flush=True)
    # access_log=False: don't print every browser request (one line per image) -- keeps the job log readable
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
