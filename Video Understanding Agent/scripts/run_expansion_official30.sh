#!/usr/bin/env bash
set -euo pipefail

VIDEO_ROOT=${VIDEO_ROOT:-/path/to/LVBench/videos}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}
GPU=${GPU:-0}
MANIFEST_DIR=${MANIFEST_DIR:-data/manifests/official_30_by_video}
MEMORY_DIR=${MEMORY_DIR:-memory/lvbench_structured_segments_official_30}
OUT_DIR=${OUT_DIR:-outputs/preds}

mkdir -p "$OUT_DIR"

# Ensure split manifests exist.
if [ ! -f "$MANIFEST_DIR/vZV2WCKMsKs.json" ]; then
  python scripts/split_manifest_by_video.py \
    --manifest data/manifests/lvbench_official_30_fixed.json \
    --out-dir "$MANIFEST_DIR"
fi

for VIDEO_KEY in vZV2WCKMsKs TJR1oYDDTwg Va_9Q6ekm60; do
  echo "[Run expansion] ${VIDEO_KEY}"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python src/agent32_structured_memory_expand.py \
    --manifest "$MANIFEST_DIR/${VIDEO_KEY}.json" \
    --video-root "$VIDEO_ROOT" \
    --memory-path "$MEMORY_DIR/${VIDEO_KEY}_60s.json" \
    --model-path "$MODEL_PATH" \
    --output "$OUT_DIR/lvbench_structured_memory_expand_official_30_${VIDEO_KEY}.jsonl" \
    --top-k 3 \
    --neighbor 2 \
    --max-groups 3 \
    --total-local-frames 24 \
    --image-size 448
 done

python scripts/merge_jsonl.py \
  --inputs \
    "$OUT_DIR/lvbench_structured_memory_expand_official_30_vZV2WCKMsKs.jsonl" \
    "$OUT_DIR/lvbench_structured_memory_expand_official_30_TJR1oYDDTwg.jsonl" \
    "$OUT_DIR/lvbench_structured_memory_expand_official_30_Va_9Q6ekm60.jsonl" \
  --output "$OUT_DIR/lvbench_structured_memory_expand_official_30.jsonl"

python scripts/summarize_predictions.py --pred "$OUT_DIR/lvbench_structured_memory_expand_official_30.jsonl"
python scripts/analyze_expanded_segment_hit.py \
  --manifest data/manifests/lvbench_official_30_fixed.json \
  --pred "$OUT_DIR/lvbench_structured_memory_expand_official_30.jsonl"
