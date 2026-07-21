import re
from typing import Any, Dict, List, Tuple
OPTION_LABELS = ["A", "B", "C", "D"]

# Parse LVBench-style options embedded inside question text:
# Question text... (A) option A (B) option B (C) option C (D) option D
def parse_inline_options_from_question(question: str) -> List[Tuple[str, str]]:
    if not question:
        return []

    pattern = re.compile(
        r"\(([ABCD])\)\s*(.*?)(?=\n\([ABCD]\)\s*|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    matches = pattern.findall(question)
    if len(matches) < 4:
        return []

    option_map = {}
    for label, text in matches:
        label = label.upper()
        text = re.sub(r"\s+", " ", text).strip()
        option_map[label] = text

    if all(label in option_map for label in OPTION_LABELS):
        return [(label, option_map[label]) for label in OPTION_LABELS]

    return []


def clean_question_text(question: str) -> str:
    if not question:
        return ""

    # Cut before the first option marker.
    cleaned = re.split(r"\n\s*\([ABCDabcd]\)\s*", question, maxsplit=1)[0]
    return cleaned.strip()


def normalize_options(raw_options: Any, item: Dict = None) -> List[Tuple[str, str]]:
    item = item or {}

    if raw_options is None:
        if all(k in item for k in OPTION_LABELS):
            raw_options = {k: item[k] for k in OPTION_LABELS}
        else:
            question = str(item.get("question", ""))
            parsed = parse_inline_options_from_question(question)
            if parsed:
                return parsed

            # Sometimes the original LVBench row is nested under raw_item.
            raw_item = item.get("raw_item")
            if isinstance(raw_item, dict):
                question = str(raw_item.get("question", ""))
                parsed = parse_inline_options_from_question(question)
                if parsed:
                    return parsed

            raise ValueError(
                "Cannot find options. Expected an 'options' field, "
                "fields A/B/C/D, or inline options in the question."
            )

    if isinstance(raw_options, dict):
        options = []
        for label in OPTION_LABELS:
            if label in raw_options:
                options.append((label, str(raw_options[label])))
            elif label.lower() in raw_options:
                options.append((label, str(raw_options[label.lower()])))
        if len(options) != 4:
            raise ValueError(f"Option dict does not contain A/B/C/D: {raw_options}")
        return options

    if isinstance(raw_options, list):
        if len(raw_options) < 4:
            raise ValueError(f"Option list has fewer than 4 choices: {raw_options}")
        return [(label, str(raw_options[i])) for i, label in enumerate(OPTION_LABELS)]

    raise ValueError(f"Unsupported options format: {type(raw_options)}")


def build_lvbench_u32_messages(question: str,options: List[Tuple[str, str]],frames: List[Dict]) -> List[Dict]:
    content = []

    for frame in frames:
        content.append({"type": "image", "image": frame["path"]})
        content.append(
            {
                "type": "text",
                "text": (
                    f"Frame {frame['frame_id'] + 1}, "
                    f"timestamp {frame['timestamp']}."
                ),
            }
        )

    question = clean_question_text(question)
    option_text = "\n".join([f"{label}. {text}" for label, text in options])

    prompt = f"""You are given uniformly sampled frames from a long video.

Answer the multiple-choice question based only on the visual evidence in these frames.

Question:
{question}

Options:
{option_text}

Important rules:
1. Choose the best answer among A, B, C, and D.
2. Do not explain your reasoning.
3. Output exactly one letter: A, B, C, or D.
"""

    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


def parse_choice_answer(text: str) -> str:
    if text is None:
        return ""

    t = text.strip()

    if t.upper() in OPTION_LABELS:
        return t.upper()

    match = re.search(r"\b([ABCD])\b", t, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\(([ABCD])\)", t, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


def normalize_gt_answer(raw_answer: Any, options: List[Tuple[str, str]]) -> str:
    if raw_answer is None:
        return ""

    if isinstance(raw_answer, int):
        if raw_answer in [0, 1, 2, 3]:
            return OPTION_LABELS[raw_answer]
        if raw_answer in [1, 2, 3, 4]:
            return OPTION_LABELS[raw_answer - 1]

    ans = str(raw_answer).strip()

    if ans.upper() in OPTION_LABELS:
        return ans.upper()

    m = re.match(r"^([ABCDabcd])[\.\)]?\s*", ans)
    if m:
        return m.group(1).upper()

    ans_norm = re.sub(r"\s+", " ", ans.lower()).strip()
    for label, opt in options:
        opt_norm = re.sub(r"\s+", " ", str(opt).lower()).strip()
        if ans_norm == opt_norm:
            return label

    return ""
