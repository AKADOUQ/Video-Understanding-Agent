from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: str | Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def accuracy(records: Iterable[Dict]) -> tuple[int, int, float]:
    correct = 0
    judged = 0
    for r in records:
        if r.get("gt_answer") or r.get("raw_answer"):
            judged += 1
            correct += int(bool(r.get("correct")))
    return correct, judged, correct / judged if judged else 0.0
