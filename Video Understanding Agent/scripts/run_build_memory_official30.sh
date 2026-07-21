#!/usr/bin/env bash
set -euo pipefail

VIDEO_ROOT=${VIDEO_ROOT:-/path/to/LVBench/videos}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}
GPU=${GPU:-0}
OUT_DIR=${OUT_DIR:-memory/lvbench_structured_segments_official_30}

mkdir -p "$OUT_DIR"

for VIDEO_KEY in vZV2WCKMsKs TJR1oYDDTwg Va_9Q6ekm60; do
  echo "[Build memory] ${VIDEO_KEY}"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python scripts/build_structured_segment_memory.py \
    --video-path "$VIDEO_ROOT/${VIDEO_KEY}.mp4" \
    --video-key "$VIDEO_KEY" \
    --model-path "$MODEL_PATH" \
    --output "$OUT_DIR/${VIDEO_KEY}_60s.json" \
    --segment-sec 60 \
    --frames-per-segment 6 \
    --image-size 448
 done
