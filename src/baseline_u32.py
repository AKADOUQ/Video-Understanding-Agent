import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.frame_sampler import sample_uniform_frames
from src.model_client import Qwen3VLClient
from src.prompt_builder import (
    build_lvbench_u32_messages,
    clean_question_text,
    normalize_gt_answer,
    normalize_options,
    parse_choice_answer,
)


VIDEO_KEYS = [
    "video_path",
    "video",
    "video_name",
    "video_file",
    "filename",
    "path",
]

QUESTION_KEYS = [
    "question",
    "query",
    "Q",
    "question_text",
]

OPTION_KEYS = [
    "options",
    "choices",
    "candidates",
    "answer_options",
]

ANSWER_KEYS = [
    "answer",
    "gt_answer",
    "correct_answer",
    "label",
    "gt",
]


def load_manifest(path: str) -> List[Dict]:
    """
    Load JSON or JSONL manifest.

    Supported top-level JSON:
    - list[dict]
    - {"data": list[dict]}
    - {"annotations": list[dict]}
    - {"questions": list[dict]}
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["data", "annotations", "questions", "items"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(f"Unsupported manifest format: {path}")


def get_first_existing(item: Dict, keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def resolve_video_path(video_value: Any, video_root: str) -> str:
    if video_value is None:
        raise ValueError("Missing video path field.")

    video_str = str(video_value)

    candidates = []

    if os.path.isabs(video_str):
        candidates.append(video_str)
    else:
        candidates.append(os.path.join(video_root, video_str))

    base_candidates = list(candidates)
    for base in base_candidates:
        stem, ext = os.path.splitext(base)
        if ext == "":
            for e in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
                candidates.append(base + e)

    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)

    raise FileNotFoundError(
        "Cannot resolve video path. Tried:\n"
        + "\n".join(candidates[:20])
    )


def get_question(item: Dict) -> str:
    q = get_first_existing(item, QUESTION_KEYS)
    if q is None:
        raise ValueError(f"Missing question field. Keys={list(item.keys())}")
    return str(q)


def get_options(item: Dict):
    raw_options = get_first_existing(item, OPTION_KEYS)
    return normalize_options(raw_options, item=item)


def get_answer(item: Dict):
    return get_first_existing(item, ANSWER_KEYS)


def get_item_id(item: Dict, idx: int) -> str:
    for key in ["id", "question_id", "qid", "uid", "sample_id"]:
        if key in item and item[key] is not None:
            return str(item[key])
    return f"item_{idx:06d}"


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str, record: Dict):
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True, help="Path to LVBench-style JSON/JSONL manifest.")
    parser.add_argument("--video-root", required=True, help="Root directory containing videos.")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""), help="Local Qwen3-VL model path.")
    parser.add_argument("--output", required=True, help="Output JSONL prediction file.")

    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bf16")

    parser.add_argument("--limit", type=int, default=-1, help="Run only first N items if > 0.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--reuse-cache", action="store_true")

    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("Please provide --model-path or set MODEL_PATH.")

    items = load_manifest(args.manifest)

    if args.limit and args.limit > 0:
        selected = items[args.start_index: args.start_index + args.limit]
    else:
        selected = items[args.start_index:]

    print("=" * 80)
    print("[Baseline] Uniform-Sampling Direct QA")
    print(f"[Manifest] {args.manifest}")
    print(f"[Video root] {args.video_root}")
    print(f"[Model path] {args.model_path}")
    print(f"[Num frames] {args.num_frames}")
    print(f"[Image size] {args.image_size}")
    print(f"[Total manifest items] {len(items)}")
    print(f"[Selected items] {len(selected)}")
    print(f"[Output] {args.output}")
    print("=" * 80)

    client = Qwen3VLClient(
        model_path=args.model_path,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    correct = 0
    judged = 0
    total = 0

    for local_idx, item in enumerate(selected):
        global_idx = args.start_index + local_idx
        item_id = get_item_id(item, global_idx)

        print("\n" + "-" * 80)
        print(f"[Item] {local_idx + 1}/{len(selected)} | global_idx={global_idx} | id={item_id}")

        t0 = time.time()

        try:
            video_value = get_first_existing(item, VIDEO_KEYS)
            video_path = resolve_video_path(video_value, args.video_root)
            question = get_question(item)
            options = get_options(item)
            raw_answer = get_answer(item)
            gt_answer = normalize_gt_answer(raw_answer, options)

            print(f"[Video] {video_path}")
            print(f"[GT] {gt_answer}")

            frames = sample_uniform_frames(
                video_path=video_path,
                num_frames=args.num_frames,
                cache_root="outputs/frame_cache",
                image_size=args.image_size,
                reuse_cache=args.reuse_cache,
            )

            messages = build_lvbench_u32_messages(
                question=question,
                options=options,
                frames=frames,
            )

            raw_output = client.generate_from_messages(
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

            pred = parse_choice_answer(raw_output)
            is_correct = bool(gt_answer and pred and pred == gt_answer)

            elapsed = time.time() - t0

            record = {
                "item_id": item_id,
                "global_idx": global_idx,
                "method": f"U{args.num_frames}",
                "video_path": video_path,
                "question": clean_question_text(question),
                "options": [{"label": x[0], "text": x[1]} for x in options],
                "gt_answer": gt_answer,
                "raw_answer": raw_answer,
                "raw_output": raw_output,
                "pred": pred,
                "correct": is_correct,
                "num_frames": args.num_frames,
                "image_size": args.image_size,
                "frames": frames,
                "elapsed_sec": elapsed,
                "error": None,
            }

            total += 1
            if gt_answer:
                judged += 1
                correct += int(is_correct)

            print(f"[Pred] {pred}")
            print(f"[Raw] {raw_output}")
            print(f"[Correct] {is_correct}")
            print(f"[Time] {elapsed:.2f}s")

        except Exception as exc:
            elapsed = time.time() - t0
            record = {
                "item_id": item_id,
                "global_idx": global_idx,
                "method": f"U{args.num_frames}",
                "raw_item": item,
                "pred": "",
                "correct": False,
                "elapsed_sec": elapsed,
                "error": repr(exc),
            }
            print(f"[Error] {repr(exc)}", file=sys.stderr)

        append_jsonl(args.output, record)

        if judged > 0:
            acc = correct / judged
            print(f"[Running Acc] {correct}/{judged} = {acc:.4f}")

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"Total processed: {total}")
    print(f"Judged: {judged}")
    if judged > 0:
        print(f"Accuracy: {correct}/{judged} = {correct / judged:.4f}")
    print(f"Output saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()