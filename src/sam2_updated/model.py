"""SAM2 model + video helpers (no UI, no global state).

Everything the web layer needs from SAM2 lives here as plain functions:
device selection, checkpoint download, model loading, frame extraction,
prompting, propagation, drawing and exporting.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import urllib.request
import warnings
from pathlib import Path
from typing import Callable, Iterator

# Let ops that Apple Silicon (MPS) doesn't support fall back to the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# SAM2's optional CUDA extension (sam2._C, only used for small hole-filling) is skipped on purpose;
# SAM2 itself says this warning is safe to ignore -- don't repeat it in the log on every click.
warnings.filterwarnings("ignore", message=r"cannot import name '_C'")

import cv2
import numpy as np
import torch

MODEL_SIZES = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{name}"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

Masks = dict[int, np.ndarray]          # obj_id -> boolean mask (H, W)
Segments = dict[int, Masks]            # frame_idx -> Masks


# ---------------------------------------------------------------- device / model
def pick_device() -> torch.device:
    """NVIDIA GPU if available, else Apple Silicon, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"CUDA -- {torch.cuda.get_device_name(0)}"
    if device.type == "mps":
        return "Apple Silicon (MPS) -- works, but slower than an NVIDIA GPU. Consider --model small."
    return "CPU only -- this will be SLOW. Use --model tiny, or run on a GPU machine."


def autocast(device: torch.device):
    """SAM2 mixes bfloat16 internals with float32 activations; every model call must run
    inside this context on CUDA (otherwise: "mat1 and mat2 must have the same dtype")."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def ensure_checkpoint(model_size: str, checkpoint_dir: Path) -> Path:
    """Return the checkpoint path, downloading it once if it is missing."""
    name = f"sam2.1_hiera_{model_size}.pt"
    path = Path(checkpoint_dir) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} (one time only) ...", flush=True)
        tmp = path.with_suffix(".pt.part")
        urllib.request.urlretrieve(CHECKPOINT_URL.format(name=name), tmp)
        tmp.replace(path)
    return path


def load_predictor(model_size: str, checkpoint_dir: Path, device: torch.device, compile_model: bool = False):
    """Build the SAM2 video predictor. torch.compile only on NVIDIA + Linux."""
    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = ensure_checkpoint(model_size, checkpoint_dir)
    config = f"configs/sam2.1/sam2.1_hiera_{MODEL_SIZES[model_size]}.yaml"
    use_compile = compile_model and device.type == "cuda" and platform.system() == "Linux"
    return build_sam2_video_predictor(config, str(checkpoint), device=device, vos_optimized=use_compile)


# ---------------------------------------------------------------- video / frames
def find_ffmpeg() -> str:
    """System ffmpeg if installed, otherwise the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    return exe


def list_videos(video_dir: Path) -> list[str]:
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        return []
    return sorted(p.name for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


def video_info(video_path: str | Path) -> tuple[float, int]:
    """(fps, expected frame count -- 0 if unknown)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return fps, count


def start_frame_extraction(video_path: str | Path, frames_dir: Path, ffmpeg: str) -> subprocess.Popen:
    """Start ffmpeg writing frames as 000000.jpg, 000001.jpg ... (caller polls / may kill it)."""
    return subprocess.Popen(
        [ffmpeg, "-y", "-i", str(video_path), "-q:v", "2", "-start_number", "0",
         f"{frames_dir}/%06d.jpg", "-hide_banner", "-loglevel", "error"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def list_frames(frames_dir: Path) -> list[str]:
    return sorted((p for p in os.listdir(frames_dir) if p.endswith(".jpg")),
                  key=lambda p: int(os.path.splitext(p)[0]))


def read_rgb(frames_dir: Path, frame_name: str) -> np.ndarray:
    bgr = cv2.imread(os.path.join(frames_dir, frame_name))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- SAM2 calls
def init_session(predictor, frames_dir: Path, device: torch.device):
    """Load all frames into a fresh SAM2 tracking session."""
    with torch.inference_mode(), autocast(device):
        state = predictor.init_state(video_path=str(frames_dir),
                                     offload_video_to_cpu=False, offload_state_to_cpu=False)
        predictor.reset_state(state)
    return state


def add_prompts(predictor, state, device: torch.device, frame_idx: int, obj_id: int,
                points: list | None, labels: list | None, box: list | None) -> Masks:
    """Set the points/box of one object on one frame; returns the masks of all objects on that frame."""
    pts = np.array(points, dtype=np.float32) if points else None
    lbl = np.array(labels, dtype=np.int32) if labels else None
    bx = np.array(box, dtype=np.float32) if box else None
    with torch.inference_mode(), autocast(device):
        _, obj_ids, logits = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=frame_idx, obj_id=obj_id, points=pts, labels=lbl, box=bx)
    return {oid: (logits[i] > 0.0).cpu().numpy().squeeze(0) for i, oid in enumerate(obj_ids)}


def warm_features(predictor, state, device: torch.device, frame_idx: int) -> None:
    """Pre-compute the image features of one frame so the next click only runs the mask decoder."""
    if frame_idx in state["cached_features"]:
        return
    with torch.inference_mode(), autocast(device):
        predictor._get_image_feature(state, frame_idx, 1)


def propagate(predictor, state, device: torch.device, reverse: bool = False,
              start_frame: int | None = None) -> Iterator[tuple[int, Masks]]:
    """Track every prompted object one frame at a time.
    Forward (default): from the earliest prompted frame to the end.
    reverse=True: from start_frame back to frame 0 (covers frames before an object's first click)."""
    with torch.inference_mode(), autocast(device):
        for frame_idx, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=start_frame, reverse=reverse):
            yield frame_idx, {oid: (logits[i] > 0.0).cpu().numpy().squeeze(0) for i, oid in enumerate(obj_ids)}


def iter_warm_up(predictor, device: torch.device, num_frames: int = 100) -> Iterator[str]:
    """Warm the GPU with a synthetic video that looks like a real job, ONE SMALL GPU STEP PER next().

    The caller runs each step only while the user is not using the GPU (see app.GpuGate), so the
    warm-up never delays a load, a click or a propagation by more than one step.
    Measured on the MIG slice: first propagate of 104 frames 16-20 s, the second one 10.7 s.
    So the dummy mimics a real run: ~100 frames, 720p, ONE object with a few clicks; and it does
    NOT call torch.cuda.empty_cache() at the end, so the GPU memory PyTorch grabbed stays ready
    for your own session (asking the driver for memory the first time is slow)."""
    import tempfile as _tf
    tmp = Path(_tf.mkdtemp(prefix="sam2_warmup_"))
    state = None
    try:
        h, w = 720, 1280                                  # CPU only: write the synthetic frames
        rng = np.random.default_rng(0)
        base = (rng.random((h, w, 3)) * 60 + 80).astype(np.uint8)
        for i in range(num_frames):                       # one moving blob on a noisy background
            img = base.copy()
            cv2.ellipse(img, (240 + 6 * i, 360), (120, 50), 0, 0, 360, (200, 200, 200), -1)
            cv2.imwrite(str(tmp / f"{i:06d}.jpg"), img)
            if i % 25 == 24:
                yield "frames"
        if device.type == "cuda":                         # tiny first GPU step: CUDA context + kernels
            with autocast(device):
                x = torch.randn(256, 256, device=device)
                (x @ x).sum().item()
        yield "cuda"
        state = init_session(predictor, tmp, device)      # frames to GPU + image encoder on frame 0
        yield "session"
        for k, pts in enumerate([[[240, 360]], [[240, 360], [200, 350]], [[240, 360], [200, 350], [290, 370]]]):
            add_prompts(predictor, state, device, 0, 1, pts, [1] * len(pts), None)   # 3 clicks, one object
            yield "click"
        for _ in propagate(predictor, state, device):     # one frame per step
            yield "propagate"
    finally:
        if state is not None:
            predictor.reset_state(state)
            del state
        shutil.rmtree(tmp, ignore_errors=True)
        # deliberately NO torch.cuda.empty_cache(): keep the memory pool warm for the real session


# ---------------------------------------------------------------- drawing
_PALETTE = [  # tab10 colours, RGB 0-255
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]


def color_for(obj_id: int) -> np.ndarray:
    return np.array(_PALETTE[obj_id % 10], dtype=np.float32)


def overlay_masks(img: np.ndarray, masks: Masks, alpha: float = 0.5) -> np.ndarray:
    """Blend coloured masks onto a float32 RGB image (in place) and return it."""
    for oid, mask in masks.items():
        img[mask] = img[mask] * (1 - alpha) + color_for(oid) * alpha
    return img


def draw_prompts(img: np.ndarray, prompts: dict) -> np.ndarray:
    """prompts: obj_id -> {"points": [[x, y]...], "labels": [1/0...], "box": [x0,y0,x1,y1] | None}."""
    for obj_id, entry in prompts.items():
        color = color_for(obj_id).tolist()
        for (x, y), lbl in zip(entry["points"], entry["labels"]):
            cv2.drawMarker(img, (int(x), int(y)), color if lbl == 1 else (255, 0, 0),
                           markerType=cv2.MARKER_STAR, markerSize=16, thickness=2)
        if entry.get("box"):
            x0, y0, x1, y1 = map(int, entry["box"])
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    return img


# ---------------------------------------------------------------- export
def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "object"


def iter_overlay_video(frames_dir: Path, frame_names: list[str], segments: Segments, fps: float,
                       out_path: Path) -> Iterator[tuple[int, int]]:
    """Write an H.264 mp4 with coloured masks, yielding (frames_done, total) after each frame.

    H.264 (not OpenCV's mp4v) so it plays directly in every browser and in QuickTime -- the
    bundled imageio-ffmpeg binary has libx264, so this works without a system ffmpeg (cluster).
    Stop early by simply not consuming the rest (the file is closed either way)."""
    h, w = cv2.imread(os.path.join(frames_dir, frame_names[0])).shape[:2]
    cmd = [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",          # libx264/yuv420p needs even sizes
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    finished = False
    try:
        for i, name in enumerate(frame_names):
            frame = cv2.imread(os.path.join(frames_dir, name)).astype(np.float32)
            for oid, mask in segments.get(i, {}).items():
                frame[mask] = frame[mask] * 0.5 + color_for(oid)[::-1] * 0.5   # frames are BGR here
            proc.stdin.write(frame.astype(np.uint8).tobytes())
            yield i + 1, len(frame_names)
        finished = True
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        if finished:
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg could not write the video: " + proc.stderr.read().decode(errors="ignore")[-300:])
        else:
            proc.kill()
            proc.wait()


def write_overlay_video(frames_dir: Path, frame_names: list[str], segments: Segments, fps: float,
                        out_path: Path, progress: Callable[[int, int], None] | None = None) -> None:
    for done, total in iter_overlay_video(frames_dir, frame_names, segments, fps, out_path):
        if progress:
            progress(done, total)


def write_png_masks(segments: Segments, names: dict[int, str], out_dir: Path) -> None:
    """One binary PNG per object per frame; the id stays in the name so equal names can't collide."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, objs in segments.items():
        for oid, mask in objs.items():
            fname = f"frame_{idx:06d}_obj{oid}_{slug(names.get(oid, f'Object {oid}'))}.png"
            cv2.imwrite(str(out_dir / fname), mask.astype(np.uint8) * 255)


def write_coco_json(segments: Segments, names: dict[int, str], video_name: str, fps: float,
                    num_frames: int, out_path: Path) -> None:
    from pycocotools import mask as mask_util

    all_ids = sorted({oid for objs in segments.values() for oid in objs})
    coco = {
        "info": {"description": "SAM2 point-prompt video labels"},
        "videos": [{"id": 1, "file_name": video_name, "fps": fps, "num_frames": num_frames}],
        "categories": [{"id": oid, "name": names.get(oid, f"Object {oid}")} for oid in all_ids],
        "annotations": [],
    }
    ann_id = 1
    for f_idx, objs in sorted(segments.items()):
        for oid, mask in objs.items():
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            ys, xs = np.where(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())] if len(xs) else [0, 0, 0, 0]
            coco["annotations"].append({"id": ann_id, "video_id": 1, "frame_index": f_idx, "category_id": oid,
                                        "segmentation": rle, "bbox": bbox, "area": int(mask.sum())})
            ann_id += 1
    out_path.write_text(json.dumps(coco))
