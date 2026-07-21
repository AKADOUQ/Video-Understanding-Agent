# Memory Directory

Structured memory files are generated locally and are not committed by default.

Expected official-30 memory files after running `scripts/run_build_memory_official30.sh`:

```text
memory/lvbench_structured_segments_official_30/
├── TJR1oYDDTwg_60s.json
├── Va_9Q6ekm60_60s.json
└── vZV2WCKMsKs_60s.json
```

Each memory entry corresponds to a 60-second video segment and contains structured fields:

- `scene`
- `characters`
- `actions`
- `objects`
- `visible_text`
- `counting_cues`
- `temporal_cues`
- `search_keywords`

The memory is isolated by `video_key` to avoid cross-video memory pollution.
