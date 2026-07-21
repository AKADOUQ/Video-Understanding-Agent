# Report Outline

Suggested title:

> A Training-Free Long-Video Understanding Agent with Structured Memory and Temporal Re-observation

## 1. Introduction / Task Understanding

Long-video QA exceeds a model's single-context visual capacity. Uniform sampling causes redundancy and misses short-lived evidence. This project focuses on harness design rather than model training: tool use, persistent memory, temporal retrieval, local re-observation, and perception/reasoning organization.

## 2. System Overview

The final system:

```text
Long Video
  -> Structured Segment Memory Builder
  -> Persistent Video Memory
Question
  -> Memory Retrieval
  -> Neighbor Expansion
  -> Local Temporal Re-observation Tool
  -> Final Visual QA
```

## 3. Multimodal Tool Design

Tool: temporal re-observation.

Instead of increasing global uniform sampling density, the system first identifies likely relevant temporal windows, then samples more frames only from those windows. This concentrates the visual budget on possible evidence segments.

## 4. Memory Mechanism

Memory is stored at 60-second segment granularity. Each segment stores structured fields:

- scene
- characters
- actions
- objects
- visible_text
- counting_cues
- temporal_cues
- search_keywords

Memory is isolated by video key. It is built offline and reused for multiple questions over the same video. This avoids cross-video memory pollution.

## 5. Perception and Reasoning Organization

The system uses weak decoupling:

- Memory building compresses video perception into structured segment memory.
- Query-time retrieval selects candidate evidence segments.
- Final answer is still produced from local visual re-observation, not from memory text alone.

This avoids the compression loss observed when text evidence fully replaces visual answering.

## 6. Dataset and Evaluation

Dataset: LVBench.

Formal subset: 3 long videos, 10 questions per video, 30 QA total.

Question types include visible text, scene/object recognition, action-chain reasoning, counting, cause/reason, and other types.

Base model: Qwen3-VL-8B-Instruct.

## 7. Main Results

Use `docs/result_tables.md`.

Core finding: U32 and U64 both achieve 33.33%; Multi-window Agent reaches 43.33%; StructuredMemory + Expansion reaches 53.33%.

## 8. Error Analysis

Remaining failures are mainly:

1. OCR / visible text: local window may be correct but text is still read incorrectly.
2. Counting: number of dogs, birds, food items, etc. requires dense frame-level evidence.
3. Action chains: after/before/then questions may require longer continuous temporal tracking.
4. Retrieval miss: memory retrieval may select semantically similar but wrong segments.
5. Final reasoning error: even when expanded hit is true, final QA may still select a wrong option.

Conclusion: after temporal localization improves, the bottleneck shifts from finding evidence to fine-grained local evidence understanding.

## 9. Limitations and Future Work

- The main formal evaluation is a 30-question subset rather than the full LVBench benchmark.
- Memory summaries can contain wrong or incomplete information.
- Counting and short-action questions may require a dedicated verifier or denser local frame sampling.
- Evidence extraction improves interpretability but should not blindly override visual answers.

## 10. AI Tool Usage

AI tools were used for code implementation assistance, debugging, command organization, and report draft writing. The system design, experiment selection, execution, result verification, and final analysis were manually reviewed and controlled by the author. AI tools were not used to generate benchmark labels or modify ground-truth answers.
