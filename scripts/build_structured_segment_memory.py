import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from src.agent_utils import extract_json_object, sec_to_time
from src.interval_sampler import get_video_info, sample_interval_frames
from src.model_client import Qwen3VLClient
from src.structured_memory_prompt_builder import (
    build_structured_segment_memory_messages,
)


DEFAULT_FIELDS = [
    "scene",
    "characters",
    "actions",
    "objects",
    "visible_text",
    "counting_cues",
    "temporal_cues",
    "search_keywords",
]


def normalize_memory_obj(raw_text: str) -> Dict:
    obj = extract_json_object(raw_text)

    if not isinstance(obj, dict):
        obj = {}

    normalized = {}

    for key in DEFAULT_FIELDS:
        value = obj.get(key, "")

        if key == "search_keywords":
            if isinstance(value, list):
                normalized[key] = [str(x).strip() for x in value if str(x).strip()]
            elif isinstance(value, str) and value.strip():
                normalized[key] = [
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                ]
            else:
                normalized[key] = []
        else:
            if isinstance(value, list):
                normalized[key] = ", ".join(str(x) for x in value)
            elif value is None:
                normalized[key] = ""
            else:
                normalized[key] = str(value).strip()

    # If JSON parsing failed, keep the raw text as a weak scene summary.
    if not any(
        normalized.get(k)
        for k in [
            "scene",
            "characters",
            "actions",
            "objects",
            "visible_text",
            "counting_cues",
            "temporal_cues",
        ]
    ):
        normalized["scene"] = raw_text.strip()

    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--segment-sec", type=float, default=60.0)
    parser.add_argument("--frames-per-segment", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        print(f"[Skip] structured memory already exists: {out_path}")
        print("Use --overwrite to rebuild.")
        return

    print("=" * 80)
    print("[Build] Structured fine-grained segment memory")
    print(f"[Video] {args.video_path}")
    print(f"[Video key] {args.video_key}")
    print(f"[Segment sec] {args.segment_sec}")
    print(f"[Frames per segment] {args.frames_per_segment}")
    print(f"[Output] {out_path}")
    print("=" * 80)

    client = Qwen3VLClient(
        model_path=args.model_path,
        dtype="bf16",
        max_new_tokens=args.max_new_tokens,
    )

    info = get_video_info(args.video_path)
    duration = info["duration_sec"]

    segments: List[Dict] = []
    start = 0.0
    segment_id = 0

    while start < duration:
        end = min(duration, start + args.segment_sec)

        start_text = sec_to_time(start)
        end_text = sec_to_time(end)

        print("\n" + "=" * 80)
        print(f"[Segment {segment_id}] {start_text}-{end_text}")

        frames = sample_interval_frames(
            video_path=args.video_path,
            start_sec=start,
            end_sec=end,
            num_frames=args.frames_per_segment,
            cache_root=(
                f"outputs/frame_cache/structured_memory/"
                f"{args.video_key}/seg_{segment_id:03d}"
            ),
            image_size=args.image_size,
        )

        messages = build_structured_segment_memory_messages(
            frames=frames,
            start_text=start_text,
            end_text=end_text,
        )

        raw_output = client.generate_from_messages(
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

        memory_obj = normalize_memory_obj(raw_output)

        print("[Raw]")
        print(raw_output)
        print("[Parsed]")
        print(json.dumps(memory_obj, ensure_ascii=False, indent=2))

        segments.append(
            {
                "segment_id": segment_id,
                "start_sec": start,
                "end_sec": end,
                "start_text": start_text,
                "end_text": end_text,
                "memory_obj": memory_obj,
                "raw_output": raw_output,
                "frames": frames,
            }
        )

        segment_id += 1
        start = end

    memory = {
        "video_key": args.video_key,
        "video_path": os.path.abspath(args.video_path),
        "duration_sec": duration,
        "duration_text": info["duration_text"],
        "segment_sec": args.segment_sec,
        "frames_per_segment": args.frames_per_segment,
        "memory_type": "structured_fine_grained_segment_memory",
        "segments": segments,
    }

    out_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"[Output] {out_path}")
    print(f"[Segments] {len(segments)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
