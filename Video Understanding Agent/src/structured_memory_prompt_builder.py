from typing import Dict, List, Tuple

from src.prompt_builder import clean_question_text


def build_structured_segment_memory_messages(frames: List[Dict],start_text: str,end_text: str):
    content = []

    for frame in frames:
        content.append({"type": "image", "image": frame["path"]})
        content.append(
            {
                "type": "text",
                "text": (
                    f"Segment frame {frame['frame_id'] + 1}, "
                    f"timestamp {frame['timestamp']}."
                ),
            }
        )

    prompt = f"""You are building a structured persistent memory index for a long video.

The provided frames are sampled from segment {start_text}-{end_text}.

Create a structured JSON memory entry for this segment.

Return JSON only, with this exact schema:
{{
  "scene": "brief description of location / environment / atmosphere",
  "characters": "main visible people or character types",
  "actions": "important visible actions or event progression",
  "objects": "important visible objects, doors, weapons, vehicles, tools, furniture, text boards, etc.",
  "visible_text": "any visible text, captions, numbers, signs, credits, logos, or empty string if none",
  "counting_cues": "people counts, repeated actions, number of objects, or empty string if unclear",
  "temporal_cues": "whether this looks like opening, transition, confrontation, ritual, travel, ending, etc.",
  "search_keywords": ["keyword1", "keyword2", "keyword3"]
}}

Rules:
1. Do not guess beyond the visible frames.
2. Be factual and concise.
3. If a field is unclear, write an empty string rather than hallucinating.
4. Include distinctive objects/actions that can help retrieve this segment later.
5. For search_keywords, include concrete nouns/actions/colors/places only.
6. Output valid JSON only. No markdown. No explanation.
"""

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def compact_segment_for_retrieval(seg: Dict) -> str:
    mem = seg.get("memory_obj") or {}

    def get(key: str) -> str:
        value = mem.get(key, "")
        if isinstance(value, list):
            return ", ".join(str(x) for x in value)
        return str(value)

    parts = [
        f"[Segment {seg['segment_id']}] {seg['start_text']}-{seg['end_text']}",
        f"scene: {get('scene')}",
        f"characters: {get('characters')}",
        f"actions: {get('actions')}",
        f"objects: {get('objects')}",
        f"visible_text: {get('visible_text')}",
        f"counting_cues: {get('counting_cues')}",
        f"temporal_cues: {get('temporal_cues')}",
        f"keywords: {get('search_keywords')}",
    ]

    return "\n".join(parts)


def build_structured_memory_retrieval_messages(question: str,options: List[Tuple[str, str]],structured_segments: List[Dict],top_k: int = 3):
    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    memory_text = "\n\n".join(
        compact_segment_for_retrieval(seg)
        for seg in structured_segments
    )

    prompt = f"""You are a retrieval module for a long-video QA agent.

You are given a structured segment memory index.
Select the {top_k} most relevant segment_id values for local visual re-observation.

Question:
{question}

Options:
{option_text}

Structured segment memory:
{memory_text}

Return JSON only, with this exact schema:
{{
  "selected_segments": [
    {{"segment_id": 0, "reason": "brief evidence-based reason"}},
    {{"segment_id": 1, "reason": "brief evidence-based reason"}},
    {{"segment_id": 2, "reason": "brief evidence-based reason"}}
  ]
}}

Retrieval rules:
1. Select exactly {top_k} different segment_id values.
2. Use the structured memory fields, not guesswork.
3. Prefer segments that directly mention the object, action, person, color, count, place, or event in the question.
4. For opening / beginning / initial caption / opening weather questions, select segment 0.
5. For counting questions, select segments with matching objects/actions and counting_cues.
6. Avoid selecting segments only because an option sounds plausible.
7. Output valid JSON only. No markdown. No explanation.
"""

    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def build_structured_memory_final_answer_messages(question: str,options: List[Tuple[str, str]],local_groups: List[Dict],selected_memory_text: str):
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
                        f"frame {frame['frame_id'] + 1}, "
                        f"timestamp {frame['timestamp']}."
                    ),
                }
            )

    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    prompt = f"""You are answering a multiple-choice question using selected local video evidence.

Selected structured memory:
{selected_memory_text}

Question:
{question}

Options:
{option_text}

Rules:
1. Choose the best answer among A, B, C, and D.
2. Use the selected video frames as primary evidence.
3. Use structured memory only as guidance; do not answer from memory if frames contradict it.
4. For counting questions, count only the relevant object/action asked in the question.
5. Output exactly one letter: A, B, C, or D.
6. Do not explain.
"""

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]
