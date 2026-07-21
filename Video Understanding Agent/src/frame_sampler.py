import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List
from PIL import Image


def _safe_video_key(video_path: str, num_frames: int) -> str:
    raw = f"{os.path.abspath(video_path)}::{num_frames}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# Uniformly sample frames from a video using midpoint sampling.
def sample_uniform_frames(video_path: str,num_frames: int = 32,cache_root: str = "outputs/frame_cache",image_size: int = 448,reuse_cache: bool = True,) -> List[Dict]:
    video_path = os.path.abspath(video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    video_key = _safe_video_key(video_path, num_frames)
    cache_dir = Path(cache_root) / f"uniform_{num_frames}" / video_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "frames_meta.json"

    if reuse_cache and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            frames = meta.get("frames", [])
            if frames and all(os.path.exists(x["path"]) for x in frames):
                return frames
        except Exception:
            pass

    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError(
            "decord is required for video frame sampling. "
            "Install with: pip install decord"
        ) from exc

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)

    if total_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = 30.0

    if fps <= 0:
        fps = 30.0

    sampled = []

    for i in range(num_frames):
        frame_index = int((i + 0.5) * total_frames / num_frames)
        frame_index = min(max(frame_index, 0), total_frames - 1)

        arr = vr[frame_index].asnumpy()
        img = Image.fromarray(arr).convert("RGB")

        if image_size and image_size > 0:
            img.thumbnail((image_size, image_size))

        frame_path = cache_dir / f"frame_{i:03d}.jpg"
        img.save(frame_path, quality=90)

        timestamp_sec = frame_index / fps

        sampled.append(
            {
                "frame_id": i,
                "path": str(frame_path.resolve()),
                "timestamp_sec": timestamp_sec,
                "timestamp": _format_timestamp(timestamp_sec),
                "frame_index": int(frame_index),
            }
        )

    meta = {
        "video_path": video_path,
        "num_frames": num_frames,
        "total_frames": total_frames,
        "fps": fps,
        "frames": sampled,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return sampled