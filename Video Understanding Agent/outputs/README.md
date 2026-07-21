# Outputs Directory

Prediction files and frame caches are generated locally.

Important official-30 prediction files:

```text
outputs/preds/lvbench_u32_official_30.jsonl
outputs/preds/lvbench_u64_official_30.jsonl
outputs/preds/lvbench_agent32_multiwindow_wo_memory_official_30.jsonl
outputs/preds/lvbench_structured_memory_expand_official_30.jsonl
```

I keep frame caches under `outputs/frame_cache/` out of version control because they are generated locally and can become large.
