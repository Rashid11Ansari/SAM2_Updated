"""SAM2 point-prompt video labeling -- local / server version of SAM2_Point_Labeling_HARDENED.ipynb.

Run:   python app.py                 (then open http://localhost:7860)
       python app.py --model small   (faster, for CPU / Apple Silicon / small GPUs)
       python app.py --share         (also print a public gradio.live link)

Differences from the Colab notebook: no Google Drive / pip installs inside the code,
checkpoint auto-downloaded to ./checkpoints, works on NVIDIA GPU (CUDA), Apple Silicon (MPS)
or CPU, ffmpeg bundled via imageio-ffmpeg, fps read with OpenCV instead of ffprobe.
"""
import argparse, os, sys, json, shutil, tempfile, subprocess, time, contextlib, platform, urllib.request

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported ops fall back to CPU on Apple Silicon

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- arguments
_SIZES = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}
parser = argparse.ArgumentParser(description="SAM2 point-prompt video labeling app")
parser.add_argument("--model", default=os.environ.get("SAM2_MODEL", "large"), choices=_SIZES,
                    help="model size (default: large; use small/tiny without an NVIDIA GPU)")
parser.add_argument("--port", type=int, default=int(os.environ.get("SAM2_PORT", "7860")))
parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = only this machine (default)")
parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
parser.add_argument("--compile", action="store_true",
                    help="torch.compile the model (vos_optimized) -- NVIDIA + Linux only; slow first run, faster after")
parser.add_argument("--checkpoint-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
ARGS = parser.parse_args()

# ---------------------------------------------------------------- device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: CUDA --", torch.cuda.get_device_name(0), flush=True)
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: Apple Silicon (MPS) -- works, but slower than an NVIDIA GPU. Consider --model small.", flush=True)
else:
    device = torch.device("cpu")
    print("Device: CPU only -- this will be SLOW. Use --model tiny, or run on Colab / a GPU machine.", flush=True)

def _autocast():
    # SAM2 mixes bfloat16 internals with float32 activations unless every call
    # into the model runs inside this context -- without it you get
    # "mat1 and mat2 must have the same dtype, but got BFloat16 and Float".
    # Only applies on CUDA; MPS/CPU run in float32.
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

# ---------------------------------------------------------------- ffmpeg
FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ---------------------------------------------------------------- checkpoint
CKPT_NAME = {"tiny": "tiny", "small": "small", "base_plus": "base_plus", "large": "large"}[ARGS.model]
ACTIVE_CKPT = f"sam2.1_hiera_{CKPT_NAME}.pt"
MODEL_CFG = f"configs/sam2.1/sam2.1_hiera_{_SIZES[ARGS.model]}.yaml"
CHECKPOINT = os.path.join(ARGS.checkpoint_dir, ACTIVE_CKPT)

if not os.path.exists(CHECKPOINT):
    os.makedirs(ARGS.checkpoint_dir, exist_ok=True)
    url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{ACTIVE_CKPT}"
    print(f"Downloading {ACTIVE_CKPT} (one time only) ...", flush=True)
    tmp = CHECKPOINT + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, CHECKPOINT)

# ---------------------------------------------------------------- model
from sam2.build_sam import build_sam2_video_predictor

use_compile = ARGS.compile and device.type == "cuda" and platform.system() == "Linux"
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=device, vos_optimized=use_compile)
print(f"SAM2 video predictor ({ARGS.model}) loaded on {device}" + (" -- torch.compile on" if use_compile else ""), flush=True)

# ---------------------------------------------------------------- web app (unchanged from the notebook below)
import gradio as gr
import cv2, os, json, shutil, tempfile, subprocess
import numpy as np
import matplotlib.pyplot as plt
import contextlib

_cmap = plt.get_cmap("tab10")
_POINT_HIT_RADIUS = 15  # pixels -- click this close to an existing point to delete it instead of adding a new one

def _color_for(obj_id):
    return np.array(_cmap(obj_id % 10)[:3]) * 255

app_state = {
    "video_dir": None,
    "frame_names": [],
    "fps": 30.0,
    "inference_state": None,
    "points": {},          # (frame_idx, obj_id) -> {"points": [...], "labels": [...], "box": [x0,y0,x1,y1] or None}
    "pending_box": None,   # first corner clicked while in Box mode, waiting for the second
    "cur_frame_idx": 0,
    "cur_obj_id": 1,
    "video_segments": {},  # frame_idx -> {obj_id: mask} from the last propagate() run
    "orig_video_path": None,
    "action_history": [],  # [{"frame_idx", "obj_id", "type": "point"|"box"}, ...] in the order they were added, across every object/frame -- what Undo actually pops
}

def _push_history(frame_idx, obj_id, kind):
    app_state["action_history"].append({"frame_idx": frame_idx, "obj_id": obj_id, "type": kind})

def _discard_history(frame_idx, obj_id, kind):
    # Used when a point/box is removed some way other than Undo (e.g. click-to-delete)
    # so the history stack doesn't reference prompts that no longer exist.
    for i in range(len(app_state["action_history"]) - 1, -1, -1):
        h = app_state["action_history"][i]
        if h["frame_idx"] == frame_idx and h["obj_id"] == obj_id and h["type"] == kind:
            app_state["action_history"].pop(i)
            return

def _read_rgb(idx):
    img = cv2.imread(os.path.join(app_state["video_dir"], app_state["frame_names"][idx]))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

def _any_prompts():
    return any(len(v["points"]) > 0 or v.get("box") for v in app_state["points"].values())

def _legend_html():
    if not _any_prompts():
        return "_No objects yet -- click the image to add a point, or draw a box._"
    obj_ids = sorted({oid for (_, oid) in app_state["points"].keys()})
    rows = []
    for oid in obj_ids:
        color = _color_for(oid).astype(int)
        total_pts = sum(len(v["points"]) for k, v in app_state["points"].items() if k[1] == oid)
        has_box = any(v.get("box") for k, v in app_state["points"].items() if k[1] == oid)
        parts = []
        if total_pts:
            parts.append(f"{total_pts} point(s)")
        if has_box:
            parts.append("box")
        detail = " + ".join(parts) if parts else "no prompts"
        current = " (current)" if oid == app_state["cur_obj_id"] else ""
        rows.append(
            '<span style="color: rgb(%d,%d,%d); font-size:16px;">&#9679;</span> Object %d -- %s%s'
            % (color[0], color[1], color[2], oid, detail, current)
        )
    return "<br>".join(rows)

def _frame_with_markers(idx):
    img = _read_rgb(idx)
    for (f_idx, obj_id), entry in app_state["points"].items():
        if f_idx != idx:
            continue
        color = _color_for(obj_id)
        for (x, y), lbl in zip(entry["points"], entry["labels"]):
            cv2.drawMarker(
                img, (int(x), int(y)), color.tolist() if lbl == 1 else (255, 0, 0),
                markerType=cv2.MARKER_STAR, markerSize=16, thickness=2,
            )
        box = entry.get("box")
        if box:
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(img, (x0, y0), (x1, y1), color.tolist(), 2)
    return img

def render_current_frame():
    idx = app_state["cur_frame_idx"]
    img = _frame_with_markers(idx)
    seg = app_state["video_segments"].get(idx)
    if seg:
        for obj_id, mask in seg.items():
            color = _color_for(obj_id)
            img[mask] = img[mask] * 0.6 + color * 0.4
    return img

def _recompute_mask(idx, obj_id, entry):
    points = np.array(entry["points"], dtype=np.float32) if entry["points"] else None
    labels = np.array(entry["labels"], dtype=np.int32) if entry["labels"] else None
    box = np.array(entry["box"], dtype=np.float32) if entry.get("box") else None
    with torch.inference_mode(), _autocast():
        return predictor.add_new_points_or_box(
            inference_state=app_state["inference_state"],
            frame_idx=idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box,
        )[1:]  # (out_obj_ids, out_mask_logits)

def load_video(video_file, progress=gr.Progress()):
    hidden = gr.update(visible=False)
    empty_outputs = (None, hidden, hidden, hidden, hidden, hidden, hidden,
                      gr.update(visible=False, interactive=False),
                      hidden, hidden, hidden,
                      gr.update(visible=False, interactive=False),
                      hidden, "Upload a video first.", gr.Tabs())

    if video_file is None:
        return empty_outputs

    progress(0.05, desc="Reading video...")
    video_dir = tempfile.mkdtemp(prefix="frames_")
    progress(0.15, desc="Extracting frames (ffmpeg)...")
    subprocess.run(
        [FFMPEG, "-y", "-i", video_file, "-q:v", "2", "-start_number", "0",
         f"{video_dir}/%06d.jpg", "-hide_banner", "-loglevel", "error"],
        check=True,
    )
    frame_names = sorted(
        [p for p in os.listdir(video_dir) if p.endswith(".jpg")],
        key=lambda p: int(os.path.splitext(p)[0]),
    )

    progress(0.55, desc="Reading video metadata...")
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    progress(0.8, desc="Initializing SAM2 tracking session...")
    # Keep a local reference to the old state and only overwrite app_state's
    # key once the new one is fully built -- never `del app_state[key]`
    # outright, since a failure mid-rebuild would otherwise leave that key
    # permanently missing and break every later function that reads it.
    old_state = app_state.get("inference_state")
    with torch.inference_mode(), _autocast():
        inference_state = predictor.init_state(
            video_path=video_dir,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )
        predictor.reset_state(inference_state)

    if old_state is not None:
        del old_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    app_state.update({
        "video_dir": video_dir,
        "frame_names": frame_names,
        "fps": fps,
        "inference_state": inference_state,
        "points": {},
        "pending_box": None,
        "cur_frame_idx": 0,
        "cur_obj_id": 1,
        "video_segments": {},
        "orig_video_path": video_file,
        "action_history": [],
    })

    progress(1.0, desc="Ready")
    first_frame = _read_rgb(0).astype(np.uint8)
    duration = len(frame_names) / fps if fps else 0
    info_md = f"**{len(frame_names)} frames** &middot; {fps:.1f} fps &middot; {duration:.1f}s"
    status = f"Loaded. Click the image to add points for object {app_state['cur_obj_id']}, or switch to Box mode."

    return (
        gr.update(value=first_frame, visible=True),                              # image_display
        gr.update(maximum=max(0, len(frame_names) - 1), value=0, visible=True),  # frame_slider
        gr.update(visible=True),                                                 # label_type
        gr.update(visible=True),                                                 # prompt_mode
        gr.update(value=_legend_html(), visible=True),                           # legend_md
        gr.update(visible=True),                                                 # new_obj_btn
        gr.update(visible=True),                                                 # undo_btn
        gr.update(visible=True, interactive=False),                              # propagate_btn
        gr.update(visible=True),                                                 # export_overlay
        gr.update(visible=True),                                                 # export_masks
        gr.update(visible=True),                                                 # export_json
        gr.update(visible=True, interactive=False),                              # export_btn
        gr.update(value=info_md, visible=True),                                  # video_info_md
        status,                                                                  # status
        gr.Tabs(selected=1),                                                     # jump straight to the Label tab
    )

def change_frame(new_idx):
    app_state["cur_frame_idx"] = int(new_idx)
    app_state["pending_box"] = None
    return render_current_frame().astype(np.uint8)

def reset_session():
    old_state = app_state.get("inference_state")
    app_state.update({
        "video_dir": None,
        "frame_names": [],
        "fps": 30.0,
        "inference_state": None,
        "points": {},
        "pending_box": None,
        "cur_frame_idx": 0,
        "cur_obj_id": 1,
        "video_segments": {},
        "orig_video_path": None,
        "action_history": [],
    })
    if old_state is not None:
        del old_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hidden = gr.update(visible=False)
    return (
        None, None, hidden, hidden, hidden, hidden, hidden, hidden,
        gr.update(visible=False, interactive=False),
        hidden, hidden, hidden,
        gr.update(visible=False, interactive=False),
        hidden,
        "Session reset -- GPU memory freed. Upload a video to start again.",
        gr.Tabs(selected=0),
    )

def new_object():
    app_state["cur_obj_id"] += 1
    app_state["pending_box"] = None
    status = f"Now placing prompts for object {app_state['cur_obj_id']}."
    return status, _legend_html()

def on_image_click(prompt_mode, label_type, evt: gr.SelectData):
    if app_state["video_dir"] is None:
        return None, "Load a video first.", _legend_html(), gr.update(interactive=False)

    x, y = evt.index
    idx = app_state["cur_frame_idx"]
    obj_id = app_state["cur_obj_id"]
    key = (idx, obj_id)

    if prompt_mode == "Box":
        pending = app_state["pending_box"]
        if pending is None:
            app_state["pending_box"] = (x, y)
            img = _frame_with_markers(idx)
            cv2.drawMarker(img, (int(x), int(y)), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
            return img.astype(np.uint8), "Box mode: click the opposite corner to finish the box.", _legend_html(), gr.update(interactive=_any_prompts())

        x0, y0 = pending
        app_state["pending_box"] = None
        entry = app_state["points"].setdefault(key, {"points": [], "labels": [], "box": None})
        entry["box"] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
        _push_history(idx, obj_id, "box")

        out_obj_ids, out_mask_logits = _recompute_mask(idx, obj_id, entry)
        img = _frame_with_markers(idx)
        for i, oid in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
            img[mask] = img[mask] * 0.5 + _color_for(oid) * 0.5
        status = f"Object {obj_id}: box set on frame {idx}."
        return img.astype(np.uint8), status, _legend_html(), gr.update(interactive=True)

    entry = app_state["points"].get(key)
    if entry and entry["points"]:
        hits = [
            i for i, (px, py) in enumerate(entry["points"])
            if ((px - x) ** 2 + (py - y) ** 2) ** 0.5 <= _POINT_HIT_RADIUS
        ]
        if hits:
            entry["points"].pop(hits[0])
            entry["labels"].pop(hits[0])
            _discard_history(idx, obj_id, "point")
            img = _frame_with_markers(idx)
            if entry["points"] or entry.get("box"):
                out_obj_ids, out_mask_logits = _recompute_mask(idx, obj_id, entry)
                for i, oid in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
                    img[mask] = img[mask] * 0.5 + _color_for(oid) * 0.5
                status = f"Removed a point from object {obj_id}. {len(entry['points'])} point(s) remain."
            else:
                del app_state["points"][key]
                status = f"Removed the only prompt for object {obj_id} on frame {idx}. Click to redefine it."
            return img.astype(np.uint8), status, _legend_html(), gr.update(interactive=_any_prompts())

    entry = app_state["points"].setdefault(key, {"points": [], "labels": [], "box": None})
    entry["points"].append([x, y])
    entry["labels"].append(1 if label_type == "foreground" else 0)
    _push_history(idx, obj_id, "point")

    out_obj_ids, out_mask_logits = _recompute_mask(idx, obj_id, entry)
    img = _frame_with_markers(idx)
    for i, oid in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
        img[mask] = img[mask] * 0.5 + _color_for(oid) * 0.5
    status = f"Object {obj_id}: {len(entry['points'])} point(s) placed on frame {idx}."
    return img.astype(np.uint8), status, _legend_html(), gr.update(interactive=True)

def undo_point():
    # Pops the most recent action across every object/frame -- not just
    # whatever object happens to be currently selected. Previously this only
    # ever looked at (cur_frame_idx, cur_obj_id), so once you undid the
    # newest object there was no way back to an older one (the "current
    # object" counter only ever increases). This walks the real history
    # instead, and jumps the view to wherever that action happened.
    while app_state["action_history"]:
        action = app_state["action_history"].pop()
        f_idx, obj_id, kind = action["frame_idx"], action["obj_id"], action["type"]
        key = (f_idx, obj_id)
        entry = app_state["points"].get(key)

        if not entry:
            continue  # already gone some other way -- check the next one back

        if kind == "point" and entry["points"]:
            entry["points"].pop()
            entry["labels"].pop()
        elif kind == "box" and entry.get("box"):
            entry["box"] = None
        else:
            continue  # this recorded action was already undone/removed -- keep going back

        app_state["cur_frame_idx"] = f_idx
        app_state["cur_obj_id"] = obj_id
        app_state["pending_box"] = None

        if entry["points"] or entry.get("box"):
            out_obj_ids, out_mask_logits = _recompute_mask(f_idx, obj_id, entry)
            img = _frame_with_markers(f_idx)
            for i, oid in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
                img[mask] = img[mask] * 0.5 + _color_for(oid) * 0.5
            status = f"Undid last prompt. {len(entry['points'])} point(s) remain for object {obj_id} on frame {f_idx}."
        else:
            del app_state["points"][key]
            img = render_current_frame()
            status = f"Removed the last prompt for object {obj_id} on frame {f_idx}. Add a new point or box to redefine it."

        return img.astype(np.uint8), status, _legend_html(), gr.update(interactive=_any_prompts()), gr.update(value=f_idx)

    return render_current_frame().astype(np.uint8), "Nothing left to undo.", _legend_html(), gr.update(interactive=_any_prompts()), gr.update()

def propagate(progress=gr.Progress()):
    if app_state["inference_state"] is None:
        return None, "Load a video first.", gr.update(interactive=False), gr.Tabs()
    if not _any_prompts():
        return None, "Add at least one point or box before propagating.", gr.update(interactive=False), gr.Tabs()

    frame_names, video_dir, fps = app_state["frame_names"], app_state["video_dir"], app_state["fps"]
    total = len(frame_names)
    video_segments = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.time()
    with torch.inference_mode(), _autocast():
        for count, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(
            predictor.propagate_in_video(app_state["inference_state"]), start=1
        ):
            video_segments[out_frame_idx] = {
                oid: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
                for i, oid in enumerate(out_obj_ids)
            }
            progress(count / total, desc=f"Propagating frame {count}/{total}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    propagate_seconds = time.time() - t_start

    propagate_fps = len(video_segments) / propagate_seconds if propagate_seconds > 0 else 0.0
    app_state["video_segments"] = video_segments

    h, w = cv2.imread(os.path.join(video_dir, frame_names[0])).shape[:2]
    preview_path = os.path.join(tempfile.mkdtemp(prefix="preview_"), "preview.mp4")
    writer = cv2.VideoWriter(preview_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, name in enumerate(frame_names):
        progress(i / total, desc=f"Rendering preview {i + 1}/{total}")
        frame = cv2.imread(os.path.join(video_dir, name)).astype(np.float32)
        for oid, mask in video_segments.get(i, {}).items():
            frame[mask] = frame[mask] * 0.5 + _color_for(oid) * 0.5
        writer.write(frame.astype(np.uint8))
    writer.release()
    status = (
        f"Propagated across {len(video_segments)} frames in {propagate_seconds:.1f}s ({propagate_fps:.1f} fps). "
        "Ready to export -- or go back to the Label tab, check for drift on another frame, add a correction point/box, and propagate again."
    )
    return preview_path, status, gr.update(interactive=True), gr.Tabs(selected=2)

def export(do_overlay, do_masks, do_json, progress=gr.Progress()):
    video_segments = app_state["video_segments"]
    if not video_segments:
        return None, "Propagate first before exporting."

    out_dir = tempfile.mkdtemp(prefix="export_")
    frame_names, video_dir, fps = app_state["frame_names"], app_state["video_dir"], app_state["fps"]
    total = len(frame_names)

    if do_overlay:
        h, w = cv2.imread(os.path.join(video_dir, frame_names[0])).shape[:2]
        writer = cv2.VideoWriter(os.path.join(out_dir, "labeled_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i, name in enumerate(frame_names):
            progress(0.6 * i / total, desc=f"Writing overlay video {i + 1}/{total}")
            frame = cv2.imread(os.path.join(video_dir, name)).astype(np.float32)
            for oid, mask in video_segments.get(i, {}).items():
                frame[mask] = frame[mask] * 0.5 + _color_for(oid) * 0.5
            writer.write(frame.astype(np.uint8))
        writer.release()

    if do_masks:
        progress(0.7, desc="Writing PNG masks...")
        masks_dir = os.path.join(out_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        for idx, objs in video_segments.items():
            for oid, mask in objs.items():
                cv2.imwrite(os.path.join(masks_dir, f"frame_{idx:06d}_obj_{oid}.png"), mask.astype(np.uint8) * 255)

    if do_json:
        progress(0.85, desc="Building COCO JSON...")
        from pycocotools import mask as mask_util
        all_ids = sorted({oid for objs in video_segments.values() for oid in objs})
        coco = {
            "info": {"description": "SAM2 point-prompt video labels"},
            "videos": [{"id": 1, "file_name": os.path.basename(app_state["orig_video_path"]), "fps": fps, "num_frames": len(frame_names)}],
            "categories": [{"id": oid, "name": f"object_{oid}"} for oid in all_ids],
            "annotations": [],
        }
        ann_id = 1
        for f_idx, objs in video_segments.items():
            for oid, mask in objs.items():
                rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                ys, xs = np.where(mask)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())] if len(xs) else [0, 0, 0, 0]
                coco["annotations"].append({
                    "id": ann_id, "video_id": 1, "frame_index": f_idx, "category_id": oid,
                    "segmentation": rle, "bbox": bbox, "area": int(mask.sum()),
                })
                ann_id += 1
        with open(os.path.join(out_dir, "annotations.json"), "w") as f:
            json.dump(coco, f)

    progress(1.0, desc="Zipping export...")
    zip_path = shutil.make_archive(out_dir, "zip", out_dir)
    return zip_path, "Export ready -- download below."


with gr.Blocks(title="SAM2 Point Labeling") as demo:
    gr.Markdown("# SAM2 Point-Prompt Video Labeling")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Load")
            load_btn = gr.Button("Load video", variant="primary")
            reset_btn = gr.Button("Reset session")
            status = gr.Textbox(label="Status", interactive=False)
            video_info_md = gr.Markdown(visible=False)

            gr.Markdown("### 2. Points")
            frame_slider = gr.Slider(0, 1, step=1, value=0, label="Key frame index", visible=False)
            label_type = gr.Radio(["foreground", "background"], value="foreground", label="Point type", visible=False)
            prompt_mode = gr.Radio(["Point", "Box"], value="Point", label="Prompt mode", visible=False)
            legend_md = gr.Markdown(visible=False)
            with gr.Row():
                new_obj_btn = gr.Button("New object (+1 id)", visible=False)
                undo_btn = gr.Button("Undo last prompt", visible=False)

            gr.Markdown("### 3. Propagate")
            propagate_btn = gr.Button("Propagate across video", variant="primary", visible=False, interactive=False)

            gr.Markdown("### 4. Export")
            export_overlay = gr.Checkbox(value=True, label="Overlay video", visible=False)
            export_masks = gr.Checkbox(value=True, label="PNG masks", visible=False)
            export_json = gr.Checkbox(value=True, label="COCO JSON", visible=False)
            export_btn = gr.Button("Export", visible=False, interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs() as tabs:
                with gr.Tab("1. Upload", id=0):
                    video_input_upload = gr.Video(label="Upload video")
                with gr.Tab("2. Label", id=1):
                    image_display = gr.Image(label="Click to add points, or draw a box", interactive=False)
                with gr.Tab("3. Preview & download", id=2):
                    preview_video = gr.Video(label="Preview")
                    download = gr.File(label="Download")

    load_btn.click(
        load_video,
        inputs=[video_input_upload],
        outputs=[
            image_display, frame_slider, label_type, prompt_mode, legend_md,
            new_obj_btn, undo_btn, propagate_btn,
            export_overlay, export_masks, export_json, export_btn,
            video_info_md, status, tabs,
        ],
    )
    frame_slider.change(change_frame, inputs=[frame_slider], outputs=[image_display])
    image_display.select(on_image_click, inputs=[prompt_mode, label_type], outputs=[image_display, status, legend_md, propagate_btn])
    new_obj_btn.click(new_object, outputs=[status, legend_md])
    undo_btn.click(undo_point, outputs=[image_display, status, legend_md, propagate_btn, frame_slider])
    propagate_btn.click(propagate, outputs=[preview_video, status, export_btn, tabs])
    export_btn.click(export, inputs=[export_overlay, export_masks, export_json], outputs=[download, status])
    reset_btn.click(reset_session,outputs=[video_input_upload, image_display, frame_slider, label_type, prompt_mode, legend_md,new_obj_btn, undo_btn, propagate_btn,
            export_overlay, export_masks, export_json, export_btn,
            video_info_md, status, tabs,
        ],
    )

if __name__ == "__main__":
    print(f"\nOpen http://localhost:{ARGS.port} in your browser" + (" (a public gradio.live link will also be printed)" if ARGS.share else ""), flush=True)
    demo.queue().launch(server_name=ARGS.host, server_port=ARGS.port, share=ARGS.share)

