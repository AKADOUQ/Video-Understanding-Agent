import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

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

    scored = []

    q_lower = q_text.lower()

    for seg in segments:
        text = segment_text(seg)
        score = 0

        for tok in query_set:
            if tok in text:
                score += 1

        if any(w in q_lower for w in ["opening", "beginning", "initial", "caption"]):
            if seg["segment_id"] == 0:
                score += 10

        if "weather" in q_lower:
            if any(w in text for w in ["snow", "snowy", "rain", "rainy", "cloud", "sunny"]):
                score += 4

        if any(w in q_lower for w in ["how many", "count", "times"]):
            if "counting_cues" in text or any(w in text for w in ["people", "men", "woman", "door", "knock", "carrying"]):
                score += 2

        scored.append((score, seg["segment_id"]))

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

    # Fill with heuristic fallback if LLM gives invalid / duplicate / too few.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--memory-path", required=True)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--output", required=True)

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--frames-per-segment", type=int, default=8)
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
    print("[Agent] StructuredMemoryAgent")
    print(f"[Memory] {args.memory_path}")
    print(f"[Segments] {len(memory['segments'])}")
    print(f"[Top-k] {args.top_k}")
    print(f"[Frames per segment] {args.frames_per_segment}")
    print(f"[Query-time visual budget] {args.top_k * args.frames_per_segment}")
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

                local_groups = []
                selected_memory_texts = []

                seg_by_id = {
                    int(s["segment_id"]): s
                    for s in memory["segments"]
                }

                for sid in selected_ids:
                    seg = seg_by_id[int(sid)]

                    selected_memory_texts.append(compact_segment_for_retrieval(seg))

                    frames = sample_interval_frames(
                        video_path=video_path,
                        start_sec=seg["start_sec"],
                        end_sec=seg["end_sec"],
                        num_frames=args.frames_per_segment,
                        cache_root=(
                            f"outputs/frame_cache/agent32_structured_memory/"
                            f"{item_id}_seg_{sid}"
                        ),
                        image_size=args.image_size,
                    )

                    local_groups.append(
                        {
                            "segment_id": sid,
                            "start_sec": seg["start_sec"],
                            "end_sec": seg["end_sec"],
                            "start_text": seg["start_text"],
                            "end_text": seg["end_text"],
                            "memory_obj": seg.get("memory_obj", {}),
                            "frames": frames,
                        }
                    )

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
                    "method": "StructuredMemoryAgent_3x8",
                    "question": clean_question_text(question),
                    "gt_answer": gt,
                    "raw_answer": raw_answer,
                    "retrieval_output": retrieval_output,
                    "retrieval_obj": retrieval_obj,
                    "selected_segments": selected_ids,
                    "selected_segment_text": [
                        {
                            "segment_id": g["segment_id"],
                            "start_text": g["start_text"],
                            "end_text": g["end_text"],
                            "memory_obj": g["memory_obj"],
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
                print("[Retrieved]", selected_ids)
                print("[Retrieval raw]", retrieval_output)
                print("[Pred]", pred)
                print("[Raw]", raw_output)
                print("[Correct]", is_correct)
                print(f"[Running Acc] {correct}/{judged} = {correct / judged if judged else 0:.4f}")

            except Exception as exc:
                record = {
                    "item_id": item_id,
                    "global_idx": global_idx,
                    "method": "StructuredMemoryAgent_3x8",
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
