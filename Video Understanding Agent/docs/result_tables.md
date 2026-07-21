# Result Tables

## Official-30 Main Results

Evaluation subset: `data/manifests/lvbench_official_30_fixed.json`.

| Method | Memory | Tool / Observation | Visual Budget | Accuracy |
|---|---:|---|---:|---:|
| Uniform U32 | No | Uniform sampling + direct QA | 32 frames | 10/30 = 33.33% |
| Uniform U64 | No | Uniform sampling + direct QA | 64 frames | 10/30 = 33.33% |
| Multi-window Agent | No | 8 coarse frames + 3 local windows × 8 frames | 32 frames | 13/30 = 43.33% |
| StructuredMemory + Expansion | Yes | Memory retrieval + neighbor expansion + local re-observation | 24 local frames + memory | 16/30 = 53.33% |

## Retrieval Coverage

| Method | Seed Hit | Expanded Hit | Accuracy |
|---|---:|---:|---:|
| StructuredMemory + Expansion | 20/30 = 66.67% | 25/30 = 83.33% | 16/30 = 53.33% |

## Interpretation

- U64 does not improve over U32, indicating that simply increasing uniformly sampled frames is not a reliable solution.
- Multi-window re-observation improves accuracy under the same 32-frame visual budget.
- Structured memory further improves performance by retrieving question-relevant video segments.
- Neighbor expansion improves temporal evidence coverage, showing that adjacent temporal context is important for long-video QA.
