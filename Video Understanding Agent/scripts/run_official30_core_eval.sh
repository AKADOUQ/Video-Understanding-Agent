#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${MANIFEST:-data/manifests/lvbench_official_30_fixed.json}
VIDEO_ROOT=${VIDEO_ROOT:-/path/to/LVBench/videos}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}
GPU=${GPU:-0}
OUT_DIR=${OUT_DIR:-outputs/preds}
mkdir -p "$OUT_DIR"

run_baseline () {
  local FRAMES=$1
  local OUT=$2
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python src/baseline_u32.py \
    --manifest "$MANIFEST" \
    --video-root "$VIDEO_ROOT" \
    --model-path "$MODEL_PATH" \
    --output "$OUT" \
    --num-frames "$FRAMES" \
    --image-size 448 \
    --max-new-tokens 32 \
    --dtype bf16 \
    --limit 30
}

echo "[1/4] U32 baseline"
run_baseline 32 "$OUT_DIR/lvbench_u32_official_30.jsonl"

echo "[2/4] U64 baseline"
run_baseline 64 "$OUT_DIR/lvbench_u64_official_30.jsonl"

echo "[3/4] Multi-window Agent w/o Memory"
MANIFEST="$MANIFEST" VIDEO_ROOT="$VIDEO_ROOT" MODEL_PATH="$MODEL_PATH" \
OUT="$OUT_DIR/lvbench_agent32_multiwindow_wo_memory_official_30.jsonl" \
GPU="$GPU" LIMIT=30 bash scripts/run_agent32_multiwindow_wo_memory.sh

echo "[4/4] Summaries"
python scripts/summarize_predictions.py --pred "$OUT_DIR/lvbench_u32_official_30.jsonl"
python scripts/summarize_predictions.py --pred "$OUT_DIR/lvbench_u64_official_30.jsonl"
python scripts/summarize_predictions.py --pred "$OUT_DIR/lvbench_agent32_multiwindow_wo_memory_official_30.jsonl"

if [ -f "$OUT_DIR/lvbench_structured_memory_expand_official_30.jsonl" ]; then
  python scripts/summarize_predictions.py --pred "$OUT_DIR/lvbench_structured_memory_expand_official_30.jsonl"
  python scripts/analyze_expanded_segment_hit.py \
    --manifest "$MANIFEST" \
    --pred "$OUT_DIR/lvbench_structured_memory_expand_official_30.jsonl"
fi
