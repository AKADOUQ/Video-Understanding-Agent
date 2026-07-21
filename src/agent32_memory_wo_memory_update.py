import argparse
import json
import os
import time
from pathlib import Path

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
from src.memory_prompt_builder import (
    build_memory_final_answer_messages,
    build_memory_retrieval_messages,
)
from src.model_client import Qwen3VLClient
from src.prompt_builder import clean_question_text, normalize_gt_answer, parse_choice_answer


def parse_selected_segments(obj, memory, top_k=3):
    segments = memory["segments"]
    valid_ids = {s["segment_id"] for s in segments}

    selected = []

    if isinstance(obj, dict):
        arr = obj.get("selected_segments", [])
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
        for sid in range(min(top_k, len(segments))):
            if sid not in selected:
                selected.append(sid)
            if len(selected) == top_k:
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
    args = parser.parse_args()

    memory = json.load(open(args.memory_path, "r", encoding="utf-8"))
    items = load_manifest(args.manifest)
    selected_items = items[: args.limit] if args.limit > 0 else items

    client = Qwen3VLClient(
        model_path=args.model_path,
        dtype="bf16",
        max_new_tokens=256,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    judged = 0
    total = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(selected_items):
            item_id = get_item_id(item, idx)

            print("\n" + "-" * 80)
            print(f"[Item] {idx+1}/{len(selected_items)} | {item_id}")

            t0 = time.time()

            try:
                video_value = get_first_existing(item, VIDEO_KEYS)
                video_path = resolve_video_path(video_value, args.video_root)

                question = get_question(item)
                options = get_options(item)
                raw_answer = get_answer(item)
                gt = normalize_gt_answer(raw_answer, options)

                retrieval_messages = build_memory_retrieval_messages(
                    question=question,
                    options=options,
                    memory_segments=memory["segments"],
                    top_k=args.top_k,
                )

                retrieval_output = client.generate_from_messages(
                    messages=retrieval_messages,
                    max_new_tokens=256,
                    do_sample=False,
                )

                retrieval_obj = extract_json_object(retrieval_output)
                selected_ids = parse_selected_segments(
                    retrieval_obj,
                    memory=memory,
                    top_k=args.top_k,
                )

                local_groups = []
                selected_memory_texts = []

                for sid in selected_ids:
                    seg = memory["segments"][sid]
                    selected_memory_texts.append(
                        f"[Segment {sid}] {seg['start_text']}-{seg['end_text']}: {seg['summary']}"
                    )

                    frames = sample_interval_frames(
                        video_path=video_path,
                        start_sec=seg["start_sec"],
                        end_sec=seg["end_sec"],
                        num_frames=args.frames_per_segment,
                        cache_root=f"outputs/frame_cache/agent32_memory/{item_id}_seg_{sid}",
                        image_size=args.image_size,
                    )

                    local_groups.append(
                        {
                            "segment_id": sid,
                            "start_sec": seg["start_sec"],
                            "end_sec": seg["end_sec"],
                            "start_text": seg["start_text"],
                            "end_text": seg["end_text"],
                            "summary": seg["summary"],
                            "frames": frames,
                        }
                    )

                final_messages = build_memory_final_answer_messages(
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
                    "method": "MemoryAgent_3x8",
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
                            "summary": g["summary"],
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
                    "method": "MemoryAgent_3x8",
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
    print("Output:", args.output)


if __name__ == "__main__":
    main()
