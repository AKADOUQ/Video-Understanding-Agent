import argparse
import json
import re
from pathlib import Path


def parse_inline_options(question):
    q = str(question)

    pattern = r"\(([A-Fa-f])\)\s*"
    matches = list(re.finditer(pattern, q))

    if not matches:
        return q.strip(), []

    clean_q = q[:matches[0].start()].strip()

    options = []
    for i, m in enumerate(matches):
        label = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(q)
        text = q[start:end].strip()
        text = re.sub(r"\s+", " ", text)
        options.append([label, text])

    return clean_q, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.load(open(args.input, "r", encoding="utf-8"))

    fixed = []
    parsed_count = 0

    for item in data:
        x = dict(item)

        q = x.get("question", "")
        old_options = x.get("options") or []

        clean_q, inline_options = parse_inline_options(q)

        if inline_options and not old_options:
            x["question_with_inline_options"] = q
            x["question"] = clean_q
            x["options"] = inline_options
            parsed_count += 1

        fixed.append(x)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[Done]")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Items: {len(fixed)}")
    print(f"Parsed inline options: {parsed_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
