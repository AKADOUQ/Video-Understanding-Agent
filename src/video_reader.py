from __future__ import annotations

from pathlib import Path
from typing import Dict

from decord import VideoReader, cpu


def probe_video(path: str | Path) -> Dict[str, float]:
    vr = VideoReader(str(path), ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    frames = len(vr)
    return {"frames": frames,"fps": fps,"duration_sec": frames / fps if fps else 0.0}
