from typing import Dict, List, Tuple

from src.prompt_builder import clean_question_text


def build_segment_summary_messages(frames: List[Dict], start_text: str, end_text: str):
    content = []

    for frame in frames:
        content.append({"type": "image", "image": frame["path"]})
        content.append(
            {
                "type": "text",
                "text": f"Segment frame {frame['frame_id'] + 1}, timestamp {frame['timestamp']}.",
            }
        )

    prompt = f"""You are building a persistent memory index for a long video.

The frames are sampled from video segment {start_text}-{end_text}.

Write a concise factual summary of what is visually happening in this segment.

Focus on:
- main characters and objects
- scene/location
- important actions
- visible text/captions
- counts if clear
- distinctive visual cues useful for later question answering

Rules:
1. Do not guess beyond the frames.
2. Use short factual sentences.
3. Mention timestamps when useful.
4. Output only the segment summary.
"""

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_memory_retrieval_messages(
    question: str,
    options: List[Tuple[str, str]],
    memory_segments: List[Dict],
    top_k: int = 3,
):
    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    mem_texts = []
    for seg in memory_segments:
        mem_texts.append(
            f"[Segment {seg['segment_id']}] "
            f"{seg['start_text']}-{seg['end_text']}\n"
            f"{seg['summary']}"
        )

    memory_text = "\n\n".join(mem_texts)

    prompt = f"""You are a retrieval module for a long-video QA agent.

Given a question and a persistent segment memory index, select the {top_k} most relevant video segments for re-observation.

Question:
{question}

Options:
{option_text}

Segment memory:
{memory_text}

Return JSON only, with this exact format:
{{
  "selected_segments": [
    {{"segment_id": 0, "reason": "brief reason"}},
    {{"segment_id": 1, "reason": "brief reason"}},
    {{"segment_id": 2, "reason": "brief reason"}}
  ]
}}

Rules:
1. Select exactly {top_k} different segment_id values.
2. Prefer segments whose summary directly mentions the object, action, person, place, or event in the question.
3. For "opening", "beginning", or "caption" questions, select the earliest segment.
4. For counting questions, select segments that mention the target object/action and nearby context.
5. Output JSON only.
"""

    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def build_memory_final_answer_messages(question: str,options: List[Tuple[str, str]],local_groups: List[Dict],selected_memory_text: str,):
    content = []

    for group in local_groups:
        for frame in group["frames"]:
            content.append({"type": "image", "image": frame["path"]})
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Selected segment {group['segment_id']} "
                        f"{group['start_text']}-{group['end_text']}, "
                        f"frame {frame['frame_id'] + 1}, timestamp {frame['timestamp']}."
                    ),
                }
            )

    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    prompt = f"""You are answering a multiple-choice question using selected local video evidence.

Selected memory context:
{selected_memory_text}

Question:
{question}

Options:
{option_text}

Rules:
1. Choose the best answer among A, B, C, and D.
2. Use the selected video frames as the primary evidence.
3. For counting questions, count only the relevant object/action asked in the question.
4. Output exactly one letter: A, B, C, or D.
5. Do not explain.
"""

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]
