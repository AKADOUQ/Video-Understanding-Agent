import argparse
import json
import re


def parse_time_to_sec(t):
    parts = [int(x) for x in t.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(t)


def parse_ref(ref):
    m = re.search(
        r"(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)",
        ref,
    )
    if not m:
        return None
    return parse_time_to_sec(m.group(1)), parse_time_to_sec(m.group(2))


def overlap(a, b):
    s1, e1 = a
    s2, e2 = b
    return max(0, min(e1, e2) - max(s1, s2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest, "r", encoding="utf-8"))
    preds = [
        json.loads(line)
        for line in open(args.pred, "r", encoding="utf-8")
        if line.strip()
    ]

    by_id = {x["id"]: x for x in manifest}

    total = 0
    seed_hit = 0
    expanded_hit = 0
    correct_total = 0
    correct_and_seed_hit = 0
    correct_and_expanded_hit = 0

    print("| id | correct | time_ref | seed_hit | expanded_hit | selected | expanded_groups |")
    print("|---|---|---|---|---|---|---|")

    for r in preds:
        item_id = r.get("item_id")
        item = by_id.get(item_id)
        if not item:
            continue

        raw = item.get("raw_item", {})
        ref = raw.get("time_reference", "")
        ref_sec = parse_ref(ref)
        if not ref_sec:
            continue

        total += 1
        is_correct = bool(r.get("correct"))
        correct_total += int(is_correct)

        # Seed selected segments are one-minute segments.
        selected = r.get("selected_segments", [])
        seed_windows = []

        for sid in selected:
            try:
                sid = int(sid)
            except Exception:
                continue
            seed_windows.append((sid * 60.0, (sid + 1) * 60.0))

        seed_is_hit = any(overlap(ref_sec, w) > 0 for w in seed_windows)
        seed_hit += int(seed_is_hit)
        correct_and_seed_hit += int(is_correct and seed_is_hit)

        expanded_windows = []
        group_texts = []

        for g in r.get("expanded_groups", []):
            s = float(g["start_sec"])
            e = float(g["end_sec"])
            expanded_windows.append((s, e))
            group_texts.append(
                f"{g['segment_id']}:{g['start_text']}-{g['end_text']}"
            )

        expanded_is_hit = any(overlap(ref_sec, w) > 0 for w in expanded_windows)
        expanded_hit += int(expanded_is_hit)
        correct_and_expanded_hit += int(is_correct and expanded_is_hit)

        print(
            f"| {item_id} | {is_correct} | {ref} | {seed_is_hit} | {expanded_is_hit} | "
            f"{selected} | {'; '.join(group_texts)} |"
        )

    print()
    print(f"Seed retrieval hit rate: {seed_hit}/{total} = {seed_hit/total if total else 0:.4f}")
    print(f"Expanded hit rate: {expanded_hit}/{total} = {expanded_hit/total if total else 0:.4f}")
    print(f"Accuracy from pred file: {correct_total}/{total} = {correct_total/total if total else 0:.4f}")
    print(f"Correct and seed hit: {correct_and_seed_hit}/{total} = {correct_and_seed_hit/total if total else 0:.4f}")
    print(f"Correct and expanded hit: {correct_and_expanded_hit}/{total} = {correct_and_expanded_hit/total if total else 0:.4f}")


if __name__ == "__main__":
    main()
