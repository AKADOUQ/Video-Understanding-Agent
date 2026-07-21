#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${MANIFEST:-data/manifests/lvbench_official_30_fixed.json}
VIDEO_ROOT=${VIDEO_ROOT:-/path/to/LVBench/videos}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}
OUT=${OUT:-outputs/preds/lvbench_agent32_multiwindow_wo_memory_official_30.jsonl}
GPU=${GPU:-0}
LIMIT=${LIMIT:-30}

CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python src/agent32_multiwindow_wo_memory.py \
  --manifest "$MANIFEST" \
  --video-root "$VIDEO_ROOT" \
  --model-path "$MODEL_PATH" \
  --output "$OUT" \
  --coarse-frames 8 \
  --windows 3 \
  --frames-per-window 8 \
  --image-size 448 \
  --limit "$LIMIT"
