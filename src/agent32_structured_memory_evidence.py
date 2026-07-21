import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

try:
    import torch
except Exception:
    torch = None

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
from src.evidence_prompt_builder import (
    build_evidence_extraction_messages,
    build_evidence_final_answer_messages,
)
from src.interval_sampler import sample_interval_frames
from src.model_client import Qwen3VLClient
from src.prompt_builder import clean_question_text, normalize_gt_answer, parse_choice_answer
from src.structured_memory_prompt_builder import (
    build_structured_memory_retrieval_messages,
    compact_segment_for_retrieval,
)

from src.agent32_structured_memory_expand import (
    allocate_frames_to_groups,
    expand_segments,
    make_contiguous_groups,
    parse_selected_segments,
    trim_groups_by_retrieval_priority,
)


def normalize_evidence_obj(raw_text: str) -> Dict:
    obj = extract_json_object(raw_text)

    if not isinstance(obj, dict):
        return {
            "relevant_evidence": raw_text.strip(),
            "event_sequence": [],
            "counting_observations": "",
            "option_support": {},
            "best_supported_option": "unknown",
            "uncertainty": "high: evidence JSON parsing failed",
        }

    normalized = {}

    normalized["relevant_evidence"] = str(obj.get("relevant_evidence", "")).strip()

    event_sequence = obj.get("event_sequence", [])
    if isinstance(event_sequence, list):
        normalized["event_sequence"] = [str(x).strip() for x in event_sequence if str(x).strip()]
    elif isinstance(event_sequence, str):
        normalized["event_sequence"] = [event_sequence.strip()] if event_sequence.strip() else []
    else:
        normalized["event_sequence"] = []

    normalized["counting_observations"] = str(obj.get("counting_observations", "")).strip()

    option_support = obj.get("option_support", {})
    if isinstance(option_support, dict):
        normalized["option_support"] = {
            str(k).strip().upper(): str(v).strip()
            for k, v in option_support.items()
        }
    else:
        normalized["option_support"] = {}

    best = str(obj.get("best_supported_option", "unknown")).strip().upper()
    if best not in {"A", "B", "C", "D", "UNKNOWN"}:
        best = "unknown"
    normalized["best_supported_option"] = best

    normalized["uncertainty"] = str(obj.get("uncertainty", "")).strip()

    return normalized


def maybe_empty_cuda_cache():
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


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

    parser.add_argument("--retrieval-max-new-tokens", type=int, default=384)
    parser.add_argument("--evidence-max-new-tokens", type=int, default=512)
    parser.add_argument("--final-max-new-tokens", type=int, default=32)

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
    print("[Agent] StructuredMemory + Neighbor Expansion + Evidence Extraction")
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
        max_new_tokens=max(
            args.retrieval_max_new_tokens,
            args.evidence_max_new_tokens,
            args.final_max_new_tokens,
        ),
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
                    max_new_tokens=args.retrieval_max_new_tokens,
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

                # Step 2: neighbor expansion.
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

                local_groups = []
                evidence_records = []

                # Step 3: evidence extraction per expanded group.
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

                    n_frames = frame_alloc[gi]

                    frames = sample_interval_frames(
                        video_path=video_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        num_frames=n_frames,
                        cache_root=(
                            f"outputs/frame_cache/agent32_structured_evidence/"
                            f"{item_id}_group_{gi}_{group[0]}_{group[-1]}"
                        ),
                        image_size=args.image_size,
                    )

                    local_group = {
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

                    local_groups.append(local_group)

                    evidence_messages = build_evidence_extraction_messages(
                        question=question,
                        options=options,
                        group=local_group,
                        group_memory_text=group_memory_text,
                    )

                    evidence_raw = client.generate_from_messages(
                        messages=evidence_messages,
                        max_new_tokens=args.evidence_max_new_tokens,
                        do_sample=False,
                    )

                    evidence_obj = normalize_evidence_obj(evidence_raw)

                    evidence_record = {
                        "group": {
                            "group_id": gi,
                            "segment_id": local_group["segment_id"],
                            "segment_ids": group,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "start_text": start_text,
                            "end_text": end_text,
                            "num_frames": n_frames,
                        },
                        "group_memory_text": group_memory_text,
                        "evidence_raw": evidence_raw,
                        "evidence_obj": evidence_obj,
                    }

                    evidence_records.append(evidence_record)

                    print(f"[Evidence group {gi}] {start_text}-{end_text}")
                    print(json.dumps(evidence_obj, ensure_ascii=False, indent=2))

                    maybe_empty_cuda_cache()

                # Step 4: final answer from extracted evidence.
                final_messages = build_evidence_final_answer_messages(
                    question=question,
                    options=options,
                    evidence_records=evidence_records,
                )

                raw_output = client.generate_from_messages(
                    messages=final_messages,
                    max_new_tokens=args.final_max_new_tokens,
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
                    "method": "StructuredMemoryEvidence",
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
                    "evidence_records": evidence_records,
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
                print("[Pred]", pred)
                print("[Raw]", raw_output)
                print("[Correct]", is_correct)
                print(f"[Running Acc] {correct}/{judged} = {correct / judged if judged else 0:.4f}")

            except Exception as exc:
                record = {
                    "item_id": item_id,
                    "global_idx": global_idx,
                    "method": "StructuredMemoryEvidence",
                    "raw_item": item,
                    "pred": "",
                    "correct": False,
                    "elapsed_sec": time.time() - t0,
                    "error": repr(exc),
                }
                print("[Error]", repr(exc))

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            maybe_empty_cuda_cache()

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
