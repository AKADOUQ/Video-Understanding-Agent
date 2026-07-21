from typing import Dict, List, Tuple

from src.prompt_builder import clean_question_text


def build_coarse_planner_messages(
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

    prompt = f"""You are a video question-answering agent.

You are given sparse uniformly sampled frames from a long video.
The full video duration is approximately {video_duration_text}.

Your task is NOT to answer directly unless the evidence is clearly visible.
Instead, decide whether more visual evidence is needed.

Question:
{question}

Options:
{option_text}

Return a JSON object only, with this format:
{{
  "can_answer": true or false,
  "answer": "A/B/C/D or empty string",
  "need_reobserve": true or false,
  "start_time": "HH:MM:SS",
  "end_time": "HH:MM:SS",
  "missing_evidence": "brief description"
}}

Rules:
1. If the answer is directly visible in the coarse frames, set can_answer=true and give answer.
2. If not enough evidence is visible, set can_answer=false and need_reobserve=true.
3. Choose a time window likely to contain the missing evidence.
4. Use the question wording to infer the time window when possible:
   - opening / beginning / intro / initial caption: use early part of video, e.g. 00:00:00-00:01:00.
   - final / ending: use late part of video.
   - if a relevant event appears near a coarse frame timestamp, choose a local window around that timestamp.
5. Output JSON only. No explanation outside JSON.
"""

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_final_answer_messages(
    question: str,
    options: List[Tuple[str, str]],
    coarse_frames: List[Dict],
    local_frames: List[Dict],
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

    for frame in local_frames:
        content.append({"type": "image", "image": frame["path"]})
        content.append(
            {
                "type": "text",
                "text": f"Re-observed local frame {frame['frame_id'] + 1}, timestamp {frame['timestamp']}.",
            }
        )

    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    prompt = f"""You are given:
1. sparse coarse frames from the full video;
2. re-observed local frames selected by a temporal observation tool.

Planner output:
{planner_text}

Question:
{question}

Options:
{option_text}

Answer the question based only on the provided visual evidence.

Rules:
1. Choose the best answer among A, B, C, and D.
2. Output exactly one letter: A, B, C, or D.
3. Do not explain.
"""

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]
