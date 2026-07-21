from typing import Dict, List, Tuple

from src.prompt_builder import clean_question_text


def build_multiwindow_planner_messages(
    question: str,
    options: List[Tuple[str, str]],
    frames: List[Dict],
    video_duration_text: str,
):
    content = []

    for frame in frames:
        content.append({"type": "image", "image": frame["path"]})
        content.append(
            {
                "type": "text",
                "text": f"Coarse frame {frame['frame_id'] + 1}, timestamp {frame['timestamp']}.",
            }
        )

    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    prompt = f"""You are a temporal planning module for a long-video QA agent.

You are given sparse coarse frames from a long video.
The full video duration is approximately {video_duration_text}.

Your job is to select 3 candidate time windows for re-observation.
Do NOT answer the question directly unless it is trivial.
The goal is to maximize the chance that the missing visual evidence appears in one of the selected windows.

Question:
{question}

Options:
{option_text}

Return JSON only, with this exact format:
{{
  "candidate_windows": [
    {{
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS",
      "reason": "brief reason"
    }},
    {{
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS",
      "reason": "brief reason"
    }},
    {{
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS",
      "reason": "brief reason"
    }}
  ]
}}

Planning rules:
1. Select exactly 3 candidate windows.
2. Each window should be 2 to 5 minutes long when possible.
3. For questions about the opening, beginning, initial caption, or opening weather, always include 00:00:00-00:01:30 as the first candidate window.
4. For counting questions, repeated action questions, or "how many" questions, choose windows around visually relevant scenes and make them long enough to cover the whole action.
5. If the relevant event is not visible in the coarse frames, choose diverse candidate windows rather than repeating the same early segment.
6. Do not output explanation outside JSON.
"""

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_multiwindow_final_messages(
    question: str,
    options: List[Tuple[str, str]],
    coarse_frames: List[Dict],
    local_groups: List[Dict],
    planner_text: str,
):
    content = []

    for frame in coarse_frames:
        content.append({"type": "image", "image": frame["path"]})
        content.append(
            {
                "type": "text",
                "text": f"Coarse frame {frame['frame_id'] + 1}, timestamp {frame['timestamp']}.",
            }
        )

    for group in local_groups:
        window_text = group["window_text"]
        for frame in group["frames"]:
            content.append({"type": "image", "image": frame["path"]})
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Local window {group['window_id']} "
                        f"{window_text[0]}-{window_text[1]}, "
                        f"frame {frame['frame_id'] + 1}, timestamp {frame['timestamp']}."
                    ),
                }
            )

    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    prompt = f"""You are given:
1. sparse coarse frames from the full video;
2. local frames from 3 candidate re-observation windows.

Planner output:
{planner_text}

Question:
{question}

Options:
{option_text}

Answer the question based only on the provided visual evidence.

Rules:
1. Choose the best answer among A, B, C, and D.
2. Pay attention to local window timestamps.
3. For counting questions, count only the relevant event asked in the question.
4. Output exactly one letter: A, B, C, or D.
5. Do not explain.
"""

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]
