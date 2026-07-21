from pathlib import Path
from typing import Dict, List

from PIL import Image
from decord import VideoReader, cpu

from src.agent_utils import sec_to_time


def get_video_info(video_path: str) -> Dict:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    duration = total_frames / fps

    return {"total_frames": total_frames,"fps": fps,"duration_sec": duration,"duration_text": sec_to_time(duration),}


def sample_interval_frames(video_path: str,start_sec: float,end_sec: float,num_frames: int,cache_root: str,image_size: int = 448,) -> List[Dict]:
    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    duration = total_frames / fps

    start_sec = max(0.0, float(start_sec))
    end_sec = min(duration, float(end_sec))

    if end_sec <= start_sec:
        end_sec = min(duration, start_sec + 1.0)

    frames = []

    for i in range(num_frames):
        ts = start_sec + (i + 0.5) * (end_sec - start_sec) / num_frames
        frame_index = int(ts * fps)
        frame_index = min(max(frame_index, 0), total_frames - 1)

        arr = vr[frame_index].asnumpy()
        img = Image.fromarray(arr).convert("RGB")

        if image_size and image_size > 0:
            img.thumbnail((image_size, image_size))

        frame_path = cache_dir / f"frame_{i:03d}.jpg"
        img.save(frame_path, quality=90)

        frames.append(
            {
                "frame_id": i,
                "path": str(frame_path.resolve()),
                "timestamp_sec": ts,
                "timestamp": sec_to_time(ts),
                "frame_index": frame_index,
            }
        )

    return frames
