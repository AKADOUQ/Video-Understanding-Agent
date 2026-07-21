import json
import re
from typing import Any, Dict, Optional, Tuple

def parse_time_to_sec(t: str) -> float:
    t = t.strip()
    parts = t.split(":")
    parts = [int(x) for x in parts]

    if len(parts) == 2:
        return parts[0] * 60 + parts[1]

    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    raise ValueError(f"Bad timestamp: {t}")


def sec_to_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def parse_window_from_text(text: str) -> Optional[Tuple[float, float]]:
    if not text:
        return None

    m = re.search(
        r"(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:-|to|~|???|???)\s*(\d{1,2}:\d{2}(?::\d{2})?)",
        text,
        flags=re.IGNORECASE,
    )

    if not m:
        return None

    start = parse_time_to_sec(m.group(1))
    end = parse_time_to_sec(m.group(2))

    if end <= start:
        return None

    return start, end


def clamp_window(start: float, end: float, duration: float, min_len: float = 20.0):
    start = max(0.0, float(start))
    end = min(float(duration), float(end))

    if end <= start:
        end = min(float(duration), start + min_len)

    if end - start < min_len:
        mid = (start + end) / 2
        start = max(0.0, mid - min_len / 2)
        end = min(float(duration), mid + min_len / 2)

    return start, end
