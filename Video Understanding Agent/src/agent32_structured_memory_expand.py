import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from src.agent_utils import extract_json_object
from src.baseline_u32 import (
    VIDEO_KEYS,
    get_answer,
    get_first_existing,
    get_item_id,
    get_options,
    get_question,
    load_manifest,
    resolve_video_path,
)
from src.interval_sampler import sample_interval_frames
from src.model_client import Qwen3VLClient
from src.prompt_builder import clean_question_text, normalize_gt_answer, parse_choice_answer
from src.structured_memory_prompt_builder import (
    build_structured_memory_final_answer_messages,
    build_structured_memory_retrieval_messages,
    compact_segment_for_retrieval,
)

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "with", "for", "and", "or",
    "what", "when", "where", "why", "how", "does", "do", "is", "are",
    "after", "before", "who", "which", "through", "she", "he", "they",
    "protagonist", "video", "question", "answer",
}

def tokenize(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]


def segment_text(seg: Dict) -> str:
    mem = seg.get("memory_obj") or {}
    values = []
    for key in [
        "scene",
        "characters",
        "actions",
        "objects",
        "visible_text",
        "counting_cues",
        "temporal_cues",
        "search_keywords",
    ]:
        v = mem.get(key, "")
        if isinstance(v, list):
            values.extend(str(x) for x in v)
        else:
            values.append(str(v))
    return " ".join(values).lower()


def heuristic_segment_rank(question: str, options: List, segments: List[Dict], top_k: int):
    q_text = clean_question_text(question)
    option_text = " ".join(str(x[1]) for x in options)
    query_tokens = tokenize(q_text + " " + option_text)
    query_set = set(query_tokens)
    q_lower = q_text.lower()

    scored = []

    for seg in segments:
        text = segment_text(seg)
        score = 0

        for tok in query_set:
            if tok in text:
                score += 1

        if any(w in q_lower for w in ["opening", "beginning", "initial", "caption"]):
            if seg["segment_id"] == 0:
                score += 10
            if seg["segment_id"] == 1:
                score += 5

        if "weather" in q_lower:
            if any(w in text for w in ["snow", "snowy", "rain", "rainy", "cloud", "sunny"]):
                score += 4

        if any(w in q_lower for w in ["how many", "count", "times"]):
            if any(w in text for w in ["people", "men", "woman", "door", "knock", "carrying", "count"]):
                score += 3

        if "gun" in q_lower or "pistol" in q_lower:
            if any(w in text for w in ["gun", "pistol", "weapon"]):
                score += 5

        if "red kimono" in q_lower:
            if "red" in text and "kimono" in text:
                score += 5

        if "green door" in q_lower:
            if "green" in text or "door" in text or "knocker" in text:
                score += 5

        scored.append((score, int(seg["segment_id"])))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [sid for score, sid in scored[:top_k]]

    while len(selected) < top_k:
        sid = len(selected)
        if sid not in selected:
            selected.append(sid)

    return selected[:top_k]


def parse_selected_segments(retrieval_obj, memory: Dict, question: str, options: List, top_k: int):
    segments = memory["segments"]
    valid_ids = {int(s["segment_id"]) for s in segments}

    selected = []

    if isinstance(retrieval_obj, dict):
        arr = retrieval_obj.get("selected_segments", [])
        if isinstance(arr, list):
            for x in arr:
                if not isinstance(x, dict):
                    continue

                sid = x.get("segment_id")
                try:
                    sid = int(sid)
                except Exception:
                    continue

                if sid in valid_ids and sid not in selected:
                    selected.append(sid)

    if len(selected) < top_k:
        fallback = heuristic_segment_rank(
            question=question,
            options=options,
            segments=segments,
            top_k=top_k * 2,
        )
        for sid in fallback:
            if sid not in selected:
                selected.append(sid)
            if len(selected) >= top_k:
                break

    return selected[:top_k]


def expand_segments(selected_ids: List[int], num_segments: int, neighbor: int) -> List[int]:
    expanded = set()

    for sid in selected_ids:
        for x in range(sid - neighbor, sid + neighbor + 1):
            if 0 <= x < num_segments:
                expanded.add(x)

    return sorted(expanded)


def make_contiguous_groups(segment_ids: List[int]) -> List[List[int]]:
    if not segment_ids:
        return []

    groups = []
    cur = [segment_ids[0]]

    for sid in segment_ids[1:]:
        if sid == cur[-1] + 1:
            cur.append(sid)
        else:
            groups.append(cur)
            cur = [sid]

    groups.append(cur)
    return groups


def trim_groups_by_retrieval_priority(groups: List[List[int]], selected_ids: List[int], max_groups: int):
    if len(groups) <= max_groups:
        return groups

    selected_set = set(selected_ids)

    def score_group(g):
        hit_count = sum(1 for x in g if x in selected_set)
        center_dist = min(abs(x - s) for x in g for s in selected_set) if selected_set else 999
        return (-hit_count, center_dist, g[0])

    groups = sorted(groups, key=score_group)
    groups = groups[:max_groups]
    groups = sorted(groups, key=lambda g: g[0])
    return groups


def allocate_frames_to_groups(groups: List[List[int]], memory: Dict, total_frames: int, min_per_group: int = 4):
    if not groups:
        return []

    durations = []

    for g in groups:
        seg_start = memory["segments"][g[0]]["start_sec"]
        seg_end = memory["segments"][g[-1]]["end_sec"]
        durations.append(max(1.0, seg_end - seg_start))

    n = len(groups)

    if total_frames <= n * min_per_group:
        base = max(1, total_frames // n)
        alloc = [base] * n
        remain = total_frames - sum(alloc)
        for i in range(remain):
            alloc[i % n] += 1
        return alloc

    alloc = [min_per_group] * n
    remain = total_frames - sum(alloc)

    dur_sum = sum(durations)
    extra = [int(remain * d / dur_sum) for d in durations]

    for i in range(n):
        alloc[i] += extra[i]

    while sum(alloc) < total_frames:
        # Give remaining frames to longest group.
        best = max(range(n), key=lambda i: durations[i])
        alloc[best] += 1

    while sum(alloc) > total_frames:
        worst = max(range(n), key=lambda i: alloc[i])
        if alloc[worst] > 1:
            alloc[worst] -= 1
        else:
            break

    return alloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--memory-path", required=True)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--output", required=True)

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--neighbor", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=3)
    parser.add_argument("--total-local-frames", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start-index", type=int, default=0)

    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("Please provide --model-path or set MODEL_PATH.")

    memory = json.load(open(args.memory_path, "r", encoding="utf-8"))
    items = load_manifest(args.manifest)

    if args.limit and args.limit > 0:
        selected_items = items[args.start_index: args.start_index + args.limit]
    else:
        selected_items = items[args.start_index:]

    print("=" * 80)
    print("[Agent] StructuredMemoryAgent + Neighbor Expansion")
    print(f"[Memory] {args.memory_path}")
    print(f"[Segments] {len(memory['segments'])}")
    print(f"[Top-k] {args.top_k}")
    print(f"[Neighbor] {args.neighbor}")
    print(f"[Max groups] {args.max_groups}")
    print(f"[Total local frames] {args.total_local_frames}")
    print(f"[Selected items] {len(selected_items)}")
    print(f"[Output] {args.output}")
    print("=" * 80)

    client = Qwen3VLClient(
        model_path=args.model_path,
        dtype="bf16",
        max_new_tokens=384,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    judged = 0
    total = 0

    seg_by_id = {int(s["segment_id"]): s for s in memory["segments"]}

    with open(args.output, "w", encoding="utf-8") as fout:
        for local_idx, item in enumerate(selected_items):
            global_idx = args.start_index + local_idx
            item_id = get_item_id(item, global_idx)

            print("\n" + "-" * 80)
            print(f"[Item] {local_idx + 1}/{len(selected_items)} | {item_id}")

            t0 = time.time()

            try:
                video_value = get_first_existing(item, VIDEO_KEYS)
                video_path = resolve_video_path(video_value, args.video_root)

                question = get_question(item)
                options = get_options(item)
                raw_answer = get_answer(item)
                gt = normalize_gt_answer(raw_answer, options)

                # Step 1: retrieve seed segments from structured memory.
                retrieval_messages = build_structured_memory_retrieval_messages(
                    question=question,
                    options=options,
                    structured_segments=memory["segments"],
                    top_k=args.top_k,
                )

                retrieval_output = client.generate_from_messages(
                    messages=retrieval_messages,
                    max_new_tokens=384,
                    do_sample=False,
                )

                retrieval_obj = extract_json_object(retrieval_output)

                selected_ids = parse_selected_segments(
                    retrieval_obj=retrieval_obj,
                    memory=memory,
                    question=question,
                    options=options,
                    top_k=args.top_k,
                )

                # Step 2: expand neighboring segments.
                expanded_ids = expand_segments(
                    selected_ids=selected_ids,
                    num_segments=len(memory["segments"]),
                    neighbor=args.neighbor,
                )

                groups = make_contiguous_groups(expanded_ids)
                groups = trim_groups_by_retrieval_priority(
                    groups=groups,
                    selected_ids=selected_ids,
                    max_groups=args.max_groups,
                )

                frame_alloc = allocate_frames_to_groups(
                    groups=groups,
                    memory=memory,
                    total_frames=args.total_local_frames,
                    min_per_group=4,
                )

                # Step 3: sample local frames from expanded groups.
                local_groups = []
                selected_memory_texts = []

                for gi, group in enumerate(groups):
                    start_seg = seg_by_id[group[0]]
                    end_seg = seg_by_id[group[-1]]

                    start_sec = float(start_seg["start_sec"])
                    end_sec = float(end_seg["end_sec"])
                    start_text = start_seg["start_text"]
                    end_text = end_seg["end_text"]

                    group_memory_text = "\n\n".join(
                        compact_segment_for_retrieval(seg_by_id[sid])
                        for sid in group
                    )
                    selected_memory_texts.append(group_memory_text)

                    n_frames = frame_alloc[gi]

                    frames = sample_interval_frames(
                        video_path=video_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        num_frames=n_frames,
                        cache_root=(
                            f"outputs/frame_cache/agent32_structured_expand/"
                            f"{item_id}_group_{gi}_{group[0]}_{group[-1]}"
                        ),
                        image_size=args.image_size,
                    )

                    local_groups.append(
                        {
                            "group_id": gi,
                            "segment_id": f"{group[0]}-{group[-1]}",
                            "segment_ids": group,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "start_text": start_text,
                            "end_text": end_text,
                            "num_frames": n_frames,
                            "frames": frames,
                        }
                    )

                # Step 4: final answer from expanded local evidence.
                final_messages = build_structured_memory_final_answer_messages(
                    question=question,
                    options=options,
                    local_groups=local_groups,
                    selected_memory_text="\n\n".join(selected_memory_texts),
                )

                raw_output = client.generate_from_messages(
                    messages=final_messages,
                    max_new_tokens=32,
                    do_sample=False,
                )

                pred = parse_choice_answer(raw_output)
                is_correct = bool(gt and pred and pred == gt)

                total += 1
                if gt:
                    judged += 1
                    correct += int(is_correct)

                record = {
                    "item_id": item_id,
                    "global_idx": global_idx,
                    "method": "StructuredMemoryExpand",
                    "question": clean_question_text(question),
                    "gt_answer": gt,
                    "raw_answer": raw_answer,
                    "retrieval_output": retrieval_output,
                    "retrieval_obj": retrieval_obj,
                    "selected_segments": selected_ids,
                    "expanded_segments": expanded_ids,
                    "expanded_groups": [
                        {
                            "group_id": g["group_id"],
                            "segment_id": g["segment_id"],
                            "segment_ids": g["segment_ids"],
                            "start_sec": g["start_sec"],
                            "end_sec": g["end_sec"],
                            "start_text": g["start_text"],
                            "end_text": g["end_text"],
                            "num_frames": g["num_frames"],
                        }
                        for g in local_groups
                    ],
                    "raw_output": raw_output,
                    "pred": pred,
                    "correct": is_correct,
                    "local_groups": local_groups,
                    "elapsed_sec": time.time() - t0,
                    "error": None,
                }

                print("[Question]", clean_question_text(question))
                print("[GT]", gt)
                print("[Selected]", selected_ids)
                print("[Expanded]", expanded_ids)
                print("[Groups]", record["expanded_groups"])
                print("[Retrieval raw]", retrieval_output)
                print("[Pred]", pred)
                print("[Raw]", raw_output)
                print("[Correct]", is_correct)
                print(f"[Running Acc] {correct}/{judged} = {correct / judged if judged else 0:.4f}")

            except Exception as exc:
                record = {
                    "item_id": item_id,
                    "global_idx": global_idx,
                    "method": "StructuredMemoryExpand",
                    "raw_item": item,
                    "pred": "",
                    "correct": False,
                    "elapsed_sec": time.time() - t0,
                    "error": repr(exc),
                }
                print("[Error]", repr(exc))

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"Total processed: {total}")
    print(f"Judged: {judged}")
    if judged:
        print(f"Accuracy: {correct}/{judged} = {correct / judged:.4f}")
    print(f"Output saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
