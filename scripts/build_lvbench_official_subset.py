import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VIDEO_KEYS = [
    "video",
    "video_id",
    "video_name",
    "video_key",
    "video_uid",
    "video_path",
    "video_file",
    "video_filename",
    "filename",
    "youtube_id",
]

QUESTION_KEYS = [
    "question",
    "query",
    "q",
    "question_text",
    "instruction",
]

ANSWER_KEYS = [
    "answer",
    "gt_answer",
    "correct_answer",
    "label",
    "target",
]

OPTIONS_KEYS = [
    "options",
    "choices",
    "candidates",
    "candidate_answers",
]

TIME_REF_KEYS = [
    "time_reference",
    "timestamp",
    "time",
    "relevant_time",
    "evidence_time",
]

QUESTION_TYPE_KEYS = [
    "question_type",
    "type",
    "category",
    "task_type",
]


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["train", "data", "items", "annotations", "questions", "examples"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

        # Some dataset dumps are dict[video_id] -> list[qa]
        rows = []
        for k, v in obj.items():
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        y = dict(x)
                        y.setdefault("video_key", k)
                        rows.append(y)
        if rows:
            return rows

    raise ValueError(f"Unsupported json structure: {path}")


def load_records_from_dir(path: Path) -> List[Dict[str, Any]]:
    files = sorted(
        list(path.rglob("*.json")) +
        list(path.rglob("*.jsonl"))
    )

    rows = []
    for fp in files:
        try:
            part = read_json_or_jsonl(fp)
            print(f"[Load] {fp}: {len(part)} records")
            rows.extend(part)
        except Exception as exc:
            print(f"[Skip] {fp}: {repr(exc)}")

    return rows


def load_hf_dataset(hf_name: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "datasets is not installed. Use --input with a local json/jsonl instead."
        ) from exc

    ds = load_dataset(hf_name)

    if hasattr(ds, "keys"):
        split_names = list(ds.keys())
        print(f"[HF splits] {split_names}")

        if split is None or str(split).lower() in {"", "auto", "none", "null"}:
            split_name = split_names[0]
        else:
            split_name = split

        if split_name not in ds:
            raise ValueError(
                f"Split {split_name!r} not found. Available splits: {split_names}"
            )

        print(f"[HF split selected] {split_name}")
        part = ds[split_name]
    else:
        part = ds

    return [dict(x) for x in part]


def first_existing(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in [None, ""]:
            return d[k]
    return default


def basename_no_ext(x: Any) -> str:
    if x is None:
        return ""

    x = str(x)
    x = x.strip()

    # Sometimes video field may be a path.
    x = os.path.basename(x)

    for ext in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
        if x.lower().endswith(ext):
            x = x[: -len(ext)]
            break

    return x


def normalize_options(raw_options: Any) -> List[List[str]]:
    labels = ["A", "B", "C", "D", "E", "F"]

    if raw_options is None:
        return []

    if isinstance(raw_options, dict):
        out = []
        for label in labels:
            if label in raw_options:
                out.append([label, str(raw_options[label])])
            elif label.lower() in raw_options:
                out.append([label, str(raw_options[label.lower()])])
        if out:
            return out

        # Arbitrary dict: preserve insertion order.
        out = []
        for i, (_, v) in enumerate(raw_options.items()):
            if i >= len(labels):
                break
            out.append([labels[i], str(v)])
        return out

    if isinstance(raw_options, list):
        out = []
        for i, v in enumerate(raw_options):
            if i >= len(labels):
                break

            if isinstance(v, dict):
                label = str(v.get("label", labels[i])).strip().upper()
                text = str(
                    v.get("text")
                    or v.get("answer")
                    or v.get("option")
                    or v.get("value")
                    or v
                )
                if label not in labels:
                    label = labels[i]
                out.append([label, text])
            else:
                text = str(v)
                # If string starts with "A. xxx", strip prefix.
                m = re.match(r"^\s*([A-Fa-f])[\.\):???]\s*(.*)$", text)
                if m:
                    out.append([m.group(1).upper(), m.group(2).strip()])
                else:
                    out.append([labels[i], text])

        return out

    return []


def normalize_answer(raw_answer: Any, options: List[List[str]]) -> str:
    if raw_answer is None:
        return ""

    ans = str(raw_answer).strip()

    if ans.upper() in {"A", "B", "C", "D", "E", "F"}:
        return ans.upper()

    m = re.match(r"^\s*([A-Fa-f])[\.\):???]", ans)
    if m:
        return m.group(1).upper()

    # Some datasets store integer labels.
    if ans.isdigit():
        idx = int(ans)
        # support both 0-indexed and 1-indexed
        if 0 <= idx < len(options):
            return options[idx][0]
        if 1 <= idx <= len(options):
            return options[idx - 1][0]

    # Some datasets store the answer text.
    ans_l = ans.lower()
    for label, text in options:
        if ans_l == str(text).strip().lower():
            return label

    return ans


def classify_question(question: str, raw_type: Any = None) -> str:
    if raw_type:
        t = str(raw_type).strip().lower()
        if t:
            if any(x in t for x in ["count", "number"]):
                return "counting"
            if any(x in t for x in ["ocr", "text", "subtitle", "caption"]):
                return "visible_text"
            if any(x in t for x in ["why", "cause", "reason"]):
                return "cause_reason"
            if any(x in t for x in ["action", "temporal", "event"]):
                return "action_chain"
            if any(x in t for x in ["scene", "object", "attribute"]):
                return "scene_object"
            if any(x in t for x in ["location", "spatial", "relation"]):
                return "relation_location"

    q = question.lower()

    if any(w in q for w in ["how many", "number of", "count", "times does", "times do"]):
        return "counting"

    if any(w in q for w in ["caption", "subtitle", "text", "word", "letter", "sign", "year appears", "written"]):
        return "visible_text"

    if any(w in q for w in ["why", "reason", "because", "unable", "cannot", "can't"]):
        return "cause_reason"

    if any(w in q for w in ["after", "before", "then", "when"]) and any(
        w in q for w in ["do", "does", "happen", "happens", "did", "next"]
    ):
        return "action_chain"

    if any(w in q for w in ["through the window", "front of", "behind", "beside", "near", "where", "location"]):
        return "relation_location"

    if any(w in q for w in ["weather", "color", "object", "wearing", "holding", "see", "look like"]):
        return "scene_object"

    if q.startswith("what does") or q.startswith("what happens") or q.startswith("what did"):
        return "action_chain"

    return "other"


def find_video_file(video_root: Optional[str], video_key: str) -> Optional[str]:
    if not video_root:
        return None

    root = Path(video_root)
    if not root.exists():
        return None

    for ext in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
        p = root / f"{video_key}{ext}"
        if p.exists():
            return str(p)

    # fallback recursive search, but keep it light
    matches = list(root.glob(f"**/{video_key}.*"))
    for m in matches:
        if m.suffix.lower() in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
            return str(m)

    return None


def normalize_record(raw: Dict[str, Any], idx: int, video_root: Optional[str]) -> Optional[Dict[str, Any]]:
    video_value = first_existing(raw, VIDEO_KEYS)
    video_key = basename_no_ext(video_value)

    question = first_existing(raw, QUESTION_KEYS)
    raw_options = first_existing(raw, OPTIONS_KEYS)
    raw_answer = first_existing(raw, ANSWER_KEYS)
    raw_type = first_existing(raw, QUESTION_TYPE_KEYS)
    time_ref = first_existing(raw, TIME_REF_KEYS, "")

    if not video_key or not question:
        return None

    options = normalize_options(raw_options)
    answer = normalize_answer(raw_answer, options)
    qtype = classify_question(str(question), raw_type)

    video_file = find_video_file(video_root, video_key)

    item = {
        "id": f"official_{idx:05d}",
        "video": f"{video_key}.mp4",
        "video_key": video_key,
        "video_path": video_file or f"{video_key}.mp4",
        "question": str(question),
        "options": options,
        "answer": answer,
        "question_type": qtype,
        "time_reference": str(time_ref) if time_ref is not None else "",
        "raw_item": raw,
    }

    return item


def sample_balanced_from_video(items: List[Dict[str, Any]],per_video: int,type_priority: List[str],rng: random.Random) -> List[Dict[str, Any]]:
    by_type = defaultdict(list)
    for x in items:
        by_type[x["question_type"]].append(x)

    for t in by_type:
        rng.shuffle(by_type[t])

    selected = []

    # Round-robin over types first.
    while len(selected) < per_video:
        made_progress = False

        for t in type_priority:
            if len(selected) >= per_video:
                break

            if by_type[t]:
                selected.append(by_type[t].pop())
                made_progress = True

        if not made_progress:
            break

    # Fill remaining from all leftovers.
    leftovers = []
    for xs in by_type.values():
        leftovers.extend(xs)

    rng.shuffle(leftovers)

    for x in leftovers:
        if len(selected) >= per_video:
            break
        selected.append(x)

    return selected[:per_video]


def choose_videos(by_video: Dict[str, List[Dict[str, Any]]],num_videos: int,per_video: int,require_existing_video: bool,rng: random.Random) -> List[str]:
    candidates = []

    for video_key, items in by_video.items():
        if len(items) < per_video:
            continue

        if require_existing_video:
            if not any(x.get("video_path") and Path(str(x["video_path"])).exists() for x in items):
                continue

        type_count = Counter(x["question_type"] for x in items)
        diversity = len(type_count)

        # More diverse and more questions first.
        candidates.append((diversity, len(items), video_key))

    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

    if len(candidates) < num_videos:
        print(
            f"[Warning] Only {len(candidates)} videos satisfy constraints. "
            f"Requested {num_videos}."
        )

    selected = [x[2] for x in candidates[:num_videos]]
    return selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="",
        help="Local LVBench annotation json/jsonl file or directory. Recommended.",
    )
    parser.add_argument(
        "--hf-name",
        default="",
        help="Optional HuggingFace dataset name if local input is not provided.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split when using --hf-name.",
    )

    parser.add_argument("--video-root", default="", help="Optional local video directory.")
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--per-video", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-existing-video", action="store_true")

    parser.add_argument(
        "--output",
        default="data/manifests/lvbench_official_50.json",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    if args.input:
        p = Path(args.input)
        if p.is_dir():
            raw_records = load_records_from_dir(p)
        else:
            raw_records = read_json_or_jsonl(p)
    elif args.hf_name:
        raw_records = load_hf_dataset(args.hf_name, args.split)
    else:
        raise ValueError(
            "Please provide --input local annotation path, or --hf-name if using HuggingFace cache."
        )

    print(f"[Raw records] {len(raw_records)}")

    normalized = []
    skipped = 0

    for i, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            skipped += 1
            continue

        item = normalize_record(raw, i, args.video_root or None)
        if item is None:
            skipped += 1
            continue

        normalized.append(item)

    print(f"[Normalized] {len(normalized)}")
    print(f"[Skipped] {skipped}")

    by_video = defaultdict(list)
    for x in normalized:
        by_video[x["video_key"]].append(x)

    print(f"[Videos] {len(by_video)}")

    type_priority = [
        "visible_text",
        "scene_object",
        "action_chain",
        "counting",
        "cause_reason",
        "relation_location",
        "other",
    ]

    selected_videos = choose_videos(
        by_video=by_video,
        num_videos=args.num_videos,
        per_video=args.per_video,
        require_existing_video=args.require_existing_video,
        rng=rng,
    )

    print("[Selected videos]")
    for v in selected_videos:
        tc = Counter(x["question_type"] for x in by_video[v])
        print(f"  {v}: {len(by_video[v])} QA | {dict(tc)}")

    selected_items = []

    for v in selected_videos:
        part = sample_balanced_from_video(
            items=by_video[v],
            per_video=args.per_video,
            type_priority=type_priority,
            rng=rng,
        )
        selected_items.extend(part)

    # Reassign clean ids.
    final_items = []
    for i, x in enumerate(selected_items):
        y = dict(x)
        y["id"] = f"official_{i:04d}"
        final_items.append(y)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_items, f, ensure_ascii=False, indent=2)

    video_keys_path = out_path.with_name(out_path.stem + "_video_keys.txt")
    with open(video_keys_path, "w", encoding="utf-8") as f:
        for v in selected_videos:
            f.write(v + "\n")

    stats = {
        "input": args.input,
        "hf_name": args.hf_name,
        "split": args.split,
        "num_raw_records": len(raw_records),
        "num_normalized_records": len(normalized),
        "num_videos_total": len(by_video),
        "selected_num_videos": len(selected_videos),
        "per_video": args.per_video,
        "num_items": len(final_items),
        "selected_videos": selected_videos,
        "question_type_distribution": dict(Counter(x["question_type"] for x in final_items)),
        "video_distribution": dict(Counter(x["video_key"] for x in final_items)),
        "missing_video_files": sorted(
            {
                x["video_key"]
                for x in final_items
                if not x.get("video_path") or not Path(str(x["video_path"])).exists()
            }
        ),
    }

    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[Done]")
    print(f"Output manifest: {out_path}")
    print(f"Video keys: {video_keys_path}")
    print(f"Stats: {stats_path}")
    print(f"Items: {len(final_items)}")
    print(f"Question types: {stats['question_type_distribution']}")
    print(f"Missing video files: {len(stats['missing_video_files'])}")
    if stats["missing_video_files"]:
        print("[Need videos]")
        for v in stats["missing_video_files"]:
            print(v)
    print("=" * 80)


if __name__ == "__main__":
    main()
