import argparse
import json
import os
import time
from pathlib import Path

from src.agent_prompt_builder import (
    build_coarse_planner_messages,
    build_final_answer_messages,
)
from src.agent_utils import (
    clamp_window,
    extract_json_object,
    parse_time_to_sec,
    parse_window_from_text,
    sec_to_time,
)
from src.baseline_u32 import (
    get_answer,
    get_item_id,
    get_options,
    get_question,
    get_first_existing,
    load_manifest,
    resolve_video_path,
    VIDEO_KEYS,
)
from src.frame_sampler import sample_uniform_frames
from src.interval_sampler import get_video_info, sample_interval_frames
from src.model_client import Qwen3VLClient
from src.prompt_builder import (
    clean_question_text,
    normalize_gt_answer,
    parse_choice_answer,
)


def choose_reobserve_window(planner_output: str, planner_obj, question: str, duration_sec: float):
    # 1. JSON start_time/end_time
    if isinstance(planner_obj, dict):
        start_t = planner_obj.get("start_time", "")
        end_t = planner_obj.get("end_time", "")

        try:
            if start_t and end_t:
                start = parse_time_to_sec(str(start_t))
                end = parse_time_to_sec(str(end_t))
                return clamp_window(start, end, duration_sec, min_len=30.0)
        except Exception:
            pass

    # 2. Parse from raw text
    parsed = parse_window_from_text(planner_output)
    if parsed:
        return clamp_window(parsed[0], parsed[1], duration_sec, min_len=30.0)

    # 3. Heuristic fallback from question
    q = question.lower()

    if any(w in q for w in ["opening", "beginning", "initial", "intro", "caption"]):
        return clamp_window(0, 60, duration_sec, min_len=30.0)

    if any(w in q for w in ["ending", "final", "last"]):
        return clamp_window(max(0, duration_sec - 60), duration_sec, duration_sec, min_len=30.0)

    # 4. Safe fallback: first 3 minutes
    return clamp_window(0, min(duration_sec, 180), duration_sec, min_len=30.0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--output", required=True)

    parser.add_argument("--coarse-frames", type=int, default=16)
    parser.add_argument("--local-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--limit", type=int, default=-1)
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
    print("[Agent] Agent-32 w/o Memory")
    print(f"[Coarse frames] {args.coarse_frames}")
    print(f"[Local frames] {args.local_frames}")
    print(f"[Total visual budget] {args.coarse_frames + args.local_frames}")
    print(f"[Selected items] {len(selected)}")
    print(f"[Output] {args.output}")
    print("=" * 80)

    client = Qwen3VLClient(
        model_path=args.model_path,
        dtype="bf16",
        max_new_tokens=128,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    judged = 0
    total = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for local_idx, item in enumerate(selected):
            global_idx = args.start_index + local_idx
            item_id = get_item_id(item, global_idx)

            print("\n" + "-" * 80)
            print(f"[Item] {local_idx + 1}/{len(selected)} | id={item_id}")

            t0 = time.time()

            try:
                video_value = get_first_existing(item, VIDEO_KEYS)
                video_path = resolve_video_path(video_value, args.video_root)
                video_info = get_video_info(video_path)

                question = get_question(item)
                options = get_options(item)
                raw_answer = get_answer(item)
                gt = normalize_gt_answer(raw_answer, options)

                # Step 1: coarse scan
                coarse_frames = sample_uniform_frames(
                    video_path=video_path,
                    num_frames=args.coarse_frames,
                    cache_root="outputs/frame_cache/agent32/coarse",
                    image_size=args.image_size,
                    reuse_cache=args.reuse_cache,
                )

                planner_messages = build_coarse_planner_messages(
                    question=question,
                    options=options,
                    frames=coarse_frames,
                    video_duration_text=video_info["duration_text"],
                )

                planner_output = client.generate_from_messages(
                    messages=planner_messages,
                    max_new_tokens=128,
                    do_sample=False,
                )

                planner_obj = extract_json_object(planner_output)

                # If planner directly answered with confidence, still do one local observation
                start_sec, end_sec = choose_reobserve_window(
                    planner_output=planner_output,
                    planner_obj=planner_obj,
                    question=question,
                    duration_sec=video_info["duration_sec"],
                )

                local_frames = sample_interval_frames(
                    video_path=video_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    num_frames=args.local_frames,
                    cache_root=f"outputs/frame_cache/agent32/local/{item_id}_{int(start_sec)}_{int(end_sec)}",
                    image_size=args.image_size,
                )

                # Step 3: final answer
                final_messages = build_final_answer_messages(
                    question=question,
                    options=options,
                    coarse_frames=coarse_frames,
                    local_frames=local_frames,
                    planner_text=planner_output,
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
                    "method": f"Agent{args.coarse_frames}+{args.local_frames}_wo_memory",
                    "video_path": video_path,
                    "question": clean_question_text(question),
                    "options": [{"label": x[0], "text": x[1]} for x in options],
                    "gt_answer": gt,
                    "raw_answer": raw_answer,
                    "planner_output": planner_output,
                    "planner_obj": planner_obj,
                    "chosen_window": [start_sec, end_sec],
                    "chosen_window_text": [sec_to_time(start_sec), sec_to_time(end_sec)],
                    "raw_output": raw_output,
                    "pred": pred,
                    "correct": is_correct,
                    "coarse_frames": coarse_frames,
                    "local_frames": local_frames,
                    "elapsed_sec": time.time() - t0,
                    "error": None,
                }

                print("[Question]", clean_question_text(question))
                print("[GT]", gt)
                print("[Planner]", planner_output)
                print("[Window]", record["chosen_window_text"])
                print("[Pred]", pred)
                print("[Raw]", raw_output)
                print("[Correct]", is_correct)
                print(f"[Running Acc] {correct}/{judged} = {correct / judged if judged else 0:.4f}")

            except Exception as exc:
                record = {
                    "item_id": item_id,
                    "global_idx": global_idx,
                    "method": f"Agent{args.coarse_frames}+{args.local_frames}_wo_memory",
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
