import argparse
import json
from pathlib import Path

VALID_CHOICES = {"A", "B", "C", "D"}

def load_jsonl(path):
    return [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]


def normalize_pred(x):
    if x is None:
        return ""
    x = str(x).strip().upper()
    return x if x in VALID_CHOICES else ""


def get_evidence_best_options(evidence_record):
    opts = []
    for rec in evidence_record.get("evidence_records", []):
        obj = rec.get("evidence_obj") or {}
        best = str(obj.get("best_supported_option", "")).strip().upper()
        if best in VALID_CHOICES or best == "UNKNOWN":
            opts.append(best)
    return opts


def safe_hybrid_decision(direct_pred, evidence_pred, evidence_best_options):
    direct_pred = normalize_pred(direct_pred)
    evidence_pred = normalize_pred(evidence_pred)

    if direct_pred:
        if not evidence_pred:
            return direct_pred, "KEEP_DIRECT__EVIDENCE_EMPTY_OR_UNKNOWN"
        if evidence_pred == direct_pred:
            return direct_pred, "KEEP_DIRECT__EVIDENCE_AGREES"
        return direct_pred, "KEEP_DIRECT__EVIDENCE_CONFLICTS"

    if evidence_pred:
        return evidence_pred, "FALLBACK_TO_EVIDENCE__DIRECT_EMPTY"

    for opt in evidence_best_options:
        opt = normalize_pred(opt)
        if opt:
            return opt, "FALLBACK_TO_EVIDENCE_BEST_OPTION__DIRECT_EMPTY"

    return "", "NO_VALID_PRED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    direct_records = {x["item_id"]: x for x in load_jsonl(args.direct)}
    evidence_records = {x["item_id"]: x for x in load_jsonl(args.evidence)}
    ids = sorted(set(direct_records) & set(evidence_records))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    judged = 0
    policy_counter = {}

    with open(out, "w", encoding="utf-8") as fout:
        for item_id in ids:
            d = direct_records[item_id]
            e = evidence_records[item_id]
            gt = d.get("gt_answer") or e.get("gt_answer")
            direct_pred = normalize_pred(d.get("pred"))
            evidence_pred = normalize_pred(e.get("pred"))
            evidence_best_options = get_evidence_best_options(e)

            final_pred, policy = safe_hybrid_decision(direct_pred, evidence_pred, evidence_best_options)
            is_correct = bool(gt and final_pred and final_pred == gt)
            judged += int(bool(gt))
            correct += int(is_correct)
            policy_counter[policy] = policy_counter.get(policy, 0) + 1

            record = {
                "item_id": item_id,
                "method": "SafeHybridVerifier",
                "question": d.get("question") or e.get("question"),
                "gt_answer": gt,
                "pred": final_pred,
                "raw_output": final_pred,
                "correct": is_correct,
                "direct_pred": direct_pred,
                "direct_correct": bool(d.get("correct")),
                "evidence_pred": evidence_pred,
                "evidence_correct": bool(e.get("correct")),
                "evidence_best_options": evidence_best_options,
                "hybrid_policy": policy,
                "direct_record": d,
                "evidence_record": e,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("[Safe Hybrid Verifier]")
    print(f"Direct file: {args.direct}")
    print(f"Evidence file: {args.evidence}")
    print(f"Output: {args.output}")
    print(f"Total: {len(ids)}")
    print(f"Judged: {judged}")
    if judged:
        print(f"Accuracy: {correct}/{judged} = {correct / judged:.4f}")
    print("\n[Policy counts]")
    for k, v in sorted(policy_counter.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
