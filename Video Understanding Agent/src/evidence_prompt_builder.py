import json
from typing import Dict, List, Tuple

from src.prompt_builder import clean_question_text

# Build a question-aware evidence extraction prompt for one expanded local group.
def build_evidence_extraction_messages(question: str,options: List[Tuple[str, str]],group: Dict,group_memory_text: str,):
    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    content = []

    intro = f"""You are an evidence extraction module for long-video multiple-choice QA.

Your task is NOT to answer directly first.
Your task is to inspect the selected video frames and extract evidence relevant to the question.

Question:
{question}

Options:
{option_text}

Selected structured memory for this local video range:
{group_memory_text}

Local video range:
{group.get("start_text", "")}-{group.get("end_text", "")}

Important:
- Focus only on evidence that helps answer the question.
- Preserve temporal order.
- For counting questions, carefully count visible repeated actions or objects.
- For "after X happens" questions, identify X and then describe what happens next.
- If evidence is insufficient, say so in uncertainty.
"""

    content.append({"type": "text", "text": intro})

    for frame in group["frames"]:
        content.append({"type": "image", "image": frame["path"]})
        content.append(
            {
                "type": "text",
                "text": (
                    f"Frame timestamp: {frame['timestamp']} "
                    f"(frame_id={frame['frame_id']}, video_frame_index={frame.get('frame_index', '')})."
                ),
            }
        )

    prompt = """Return JSON only, with this exact schema:
{
  "relevant_evidence": "concise visual evidence relevant to the question",
  "event_sequence": [
    "timestamp/order-aware event 1",
    "timestamp/order-aware event 2"
  ],
  "counting_observations": "counts or repeated actions observed; empty string if not relevant",
  "option_support": {
    "A": "supporting or contradicting evidence for A",
    "B": "supporting or contradicting evidence for B",
    "C": "supporting or contradicting evidence for C",
    "D": "supporting or contradicting evidence for D"
  },
  "best_supported_option": "A/B/C/D/unknown",
  "uncertainty": "low/medium/high plus brief reason"
}

Rules:
1. Output valid JSON only. No markdown.
2. Do not guess facts not visible in the frames.
3. If the local frames do not contain enough evidence, set best_supported_option to "unknown".
4. For counting/repeated-action questions, describe exactly what you observed frame by frame.
5. For action-chain questions, emphasize before/after order.
"""

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]

# Convert one evidence record to compact text for final answer prompt.
def evidence_obj_to_text(evidence_record: Dict) -> str:
    group = evidence_record.get("group", {})
    obj = evidence_record.get("evidence_obj") or {}
    raw = evidence_record.get("evidence_raw", "")

    def val(key, default=""):
        x = obj.get(key, default)
        if isinstance(x, list):
            return "; ".join(str(v) for v in x)
        if isinstance(x, dict):
            return json.dumps(x, ensure_ascii=False)
        return str(x)

    lines = [
        f"[Evidence group {group.get('group_id')}] {group.get('start_text')}-{group.get('end_text')}",
        f"segments: {group.get('segment_id')}",
        f"relevant_evidence: {val('relevant_evidence')}",
        f"event_sequence: {val('event_sequence')}",
        f"counting_observations: {val('counting_observations')}",
        f"option_support: {val('option_support')}",
        f"best_supported_option: {val('best_supported_option')}",
        f"uncertainty: {val('uncertainty')}",
    ]

    if not any(str(obj.get(k, "")).strip() for k in obj):
        lines.append(f"raw_evidence: {raw}")

    return "\n".join(lines)

# Text-only final answer prompt based on extracted evidence.
def build_evidence_final_answer_messages(question: str,options: List[Tuple[str, str]],evidence_records: List[Dict],):
    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    evidence_text = "\n\n".join(
        evidence_obj_to_text(record)
        for record in evidence_records
    )

    prompt = f"""You are the final reasoning module for long-video multiple-choice QA.

Use the extracted evidence below to choose the best answer.

Question:
{question}

Options:
{option_text}

Extracted evidence:
{evidence_text}

Decision rules:
1. Choose exactly one option: A, B, C, or D.
2. Prefer evidence that directly observes the event, count, action, text, or object asked in the question.
3. For "after X" questions, use the event_sequence and preserve temporal order.
4. For counting questions, use counting_observations.
5. If evidence is incomplete, choose the option best supported by the available evidence, not by prior knowledge.
6. Output exactly one letter: A, B, C, or D.
"""

    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
