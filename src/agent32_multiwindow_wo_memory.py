import argparse
import json
import os
import time
from pathlib import Path

from src.agent_multiwindow_prompt_builder import (
    build_multiwindow_final_messages,
    build_multiwindow_planner_messages,
)
from src.agent_utils import (
    clamp_window,
    extract_json_object,
    parse_time_to_sec,
    sec_to_time,
)
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
from src.frame_sampler import sample_uniform_frames
from src.interval_sampler import get_video_info, sample_interval_frames
from src.model_client import Qwen3VLClient
from src.prompt_builder import clean_question_text, normalize_gt_answer, parse_choice_answer


def default_windows(question: str, duration_sec: float):
    q = question.lower()

    if any(w in q for w in ["opening", "beginning", "initial", "intro", "caption"]):
        return [
            (0, 90),
            (90, 300),
            (300, 600),
        ]

    return [
        (0, min(duration_sec, 600)),
        (max(0, duration_sec * 0.30), min(duration_sec, duration_sec * 0.45)),
        (max(0, duration_sec * 0.55), min(duration_sec, duration_sec * 0.70)),
    ]


def parse_candidate_windows(planner_output: str, planner_obj, question: str, duration_sec: float):
    windows = []

    if isinstance(planner_obj, dict):
        cands = planner_obj.get("candidate_windows", [])
        if isinstance(cands, list):
            for cand in cands:
                if not isinstance(cand, dict):
                    continue

                st = cand.get("start_time")
                et = cand.get("end_time")

                try:
                    if st and et:
                        s = parse_time_to_sec(str(st))
                        e = parse_time_to_sec(str(et))
                        s, e = clamp_window(s, e, duration_sec, min_len=60.0)
                        windows.append((s, e, cand.get("reason", "")))
                except Exception:
                    pass

    # Fallback.
    if len(windows) < 3:
        for s, e in default_windows(question, duration_sec):
            s, e = clamp_window(s, e, duration_sec, min_len=60.0)
            windows.append((s, e, "fallback"))

    # Deduplicate roughly.
    deduped = []
    seen = set()
    for s, e, r in windows:
        key = (int(s // 30), int(e // 30))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((s, e, r))
        if len(deduped) == 3:
            break

    # Still not enough: add early/middle/later windows.
    if len(deduped) < 3:
        for s, e in default_windows(question, duration_sec):
            s, e = clamp_window(s, e, duration_sec, min_len=60.0)
            deduped.append((s, e, "fallback_fill"))
            if len(deduped) == 3:
                break

    return deduped[:3]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--output", required=True)

    parser.add_argument("--coarse-frames", type=int, default=8)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--reuse-cache", action="store_true")

    args = parser.parse_args()

    if args.windows != 3:
        raise ValueError("This first version expects exactly 3 windows.")

    if not args.model_path:
        raise ValueError("Please provide --model-path or set MODEL_PATH.")

    items = load_manifest(args.manifest)

    if args.limit and args.limit > 0:
        selected = items[args.start_index: args.start_index + args.limit]
    else:
        selected = items[args.start_index:]

    visual_budget = args.coarse_frames + args.windows * args.frames_per_window

    print("=" * 80)
    print("[Agent] Agent-32 Multi-window w/o Memory")
    print(f"[Coarse frames] {args.coarse_frames}")
    print(f"[Windows] {args.windows}")
    print(f"[Frames per window] {args.frames_per_window}")
    print(f"[Total visual budget] {visual_budget}")
    print(f"[Selected items] {len(selected)}")
    print(f"[Output] {args.output}")
    print("=" * 80)

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

                # Step 1: coarse scan.
                coarse_frames = sample_uniform_frames(
                    video_path=video_path,
                    num_frames=args.coarse_frames,
                    cache_root="outputs/frame_cache/agent32_multi/coarse",
                    image_size=args.image_size,
                    reuse_cache=args.reuse_cache,
                )

                planner_messages = build_multiwindow_planner_messages(
                    question=question,
                    options=options,
                    frames=coarse_frames,
                    video_duration_text=video_info["duration_text"],
                )

                planner_output = client.generate_from_messages(
                    messages=planner_messages,
                    max_new_tokens=256,
                    do_sample=False,
                )

                planner_obj = extract_json_object(planner_output)

                windows = parse_candidate_windows(
                    planner_output=planner_output,
                    planner_obj=planner_obj,
                    question=question,
                    duration_sec=video_info["duration_sec"],
                )

                local_groups = []
                for wi, (s, e, reason) in enumerate(windows):
                    frames = sample_interval_frames(
                        video_path=video_path,
                        start_sec=s,
                        end_sec=e,
                        num_frames=args.frames_per_window,
                        cache_root=f"outputs/frame_cache/agent32_multi/local/{item_id}_w{wi}_{int(s)}_{int(e)}",
                        image_size=args.image_size,
                    )

                    local_groups.append(
                        {
                            "window_id": wi + 1,
                            "window": [s, e],
                            "window_text": [sec_to_time(s), sec_to_time(e)],
                            "reason": reason,
                            "frames": frames,
                        }
                    )

                final_messages = build_multiwindow_final_messages(
                    question=question,
                    options=options,
                    coarse_frames=coarse_frames,
                    local_groups=local_groups,
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
                    "method": "Agent8+3x8_wo_memory",
                    "video_path": video_path,
                    "question": clean_question_text(question),
                    "options": [{"label": x[0], "text": x[1]} for x in options],
                    "gt_answer": gt,
                    "raw_answer": raw_answer,
                    "planner_output": planner_output,
                    "planner_obj": planner_obj,
                    "chosen_windows": [
                        {
                            "window_id": g["window_id"],
                            "window": g["window"],
                            "window_text": g["window_text"],
                            "reason": g["reason"],
                        }
                        for g in local_groups
                    ],
                    "raw_output": raw_output,
                    "pred": pred,
                    "correct": is_correct,
                    "coarse_frames": coarse_frames,
                    "local_groups": local_groups,
                    "elapsed_sec": time.time() - t0,
                    "error": None,
                }

                print("[Question]", clean_question_text(question))
                print("[GT]", gt)
                print("[Planner]", planner_output)
                print("[Windows]", record["chosen_windows"])
                print("[Pred]", pred)
                print("[Raw]", raw_output)
                print("[Correct]", is_correct)
                print(f"[Running Acc] {correct}/{judged} = {correct / judged if judged else 0:.4f}")

            except Exception as exc:
                record = {
                    "item_id": item_id,
                    "global_idx": global_idx,
                    "method": "Agent8+3x8_wo_memory",
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
