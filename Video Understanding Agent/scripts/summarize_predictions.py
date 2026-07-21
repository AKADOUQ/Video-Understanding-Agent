import argparse
import json
from collections import Counter, defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    judged = [r for r in rows if r.get("gt_answer")]
    correct = [r for r in judged if r.get("correct")]

    print("=" * 80)
    print(f"File: {args.pred}")
    print(f"Total: {total}")
    print(f"Judged: {len(judged)}")
    if judged:
        print(f"Accuracy: {len(correct)}/{len(judged)} = {len(correct)/len(judged):.4f}")

    print("\n[Prediction distribution]")
    print(Counter(r.get("pred", "") for r in rows))

    print("\n[Wrong cases]")
    for r in rows:
        if r.get("gt_answer") and not r.get("correct"):
            print("-" * 80)
            print("id:", r.get("item_id"))
            print("question:", r.get("question"))
            print("gt:", r.get("gt_answer"), "pred:", r.get("pred"))
            print("raw_output:", r.get("raw_output"))
            if r.get("frames"):
                print("first_frame:", r["frames"][0]["timestamp"])
                print("last_frame:", r["frames"][-1]["timestamp"])

if __name__ == "__main__":
    main()
