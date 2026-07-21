<img
  src="figs/agent-logo.png"
  width="82"
  align="left"
  alt="Agent Logo"
>

<h1 align="center">
  A Training-Free Long-Video Understanding Agent<br>
  with Structured Memory and Temporal Re-observation
</h1>

<br clear="left">

> **Important Note:** This repository provides the implementation of a training-free long-video QA agent built on a frozen Qwen3-VL model.  
> The framework is designed to locate sparse question-relevant evidence in long videos through structured segment memory, neighbor expansion, and local temporal re-observation, without model fine-tuning or a ready-made video-agent framework.

---

## Introduction

🌟 **Training-free long-video question answering**

Long videos often contain far more visual information than a multimodal model can process in a single pass. Uniform sampling provides broad coverage, but it may spend most of the visual budget on redundant or question-irrelevant frames while representing short-lived evidence with only an isolated observation.

This repository studies a different strategy: rather than uniformly increasing the number of sampled frames, the system organizes perception around the question. It first constructs reusable segment-level memory for each video, retrieves temporally relevant segments using the question and candidate answers, expands neighboring context to reduce boundary misses, and then returns to the original video for local visual observation.

The system is training-free: all methods use the same frozen Qwen3-VL-8B-Instruct checkpoint, and the differences come from how video observations, memory, retrieval, and tool calls are organized.

## Framework

<p align="center">
  <img src="figs/overall-framework.png" width="1000" alt="Overall framework">
</p>

The framework contains an offline memory-construction stage and an online query-time reasoning stage:

- **Uniform Sampling Baselines:** sample 32 or 64 frames across the full video and perform one-shot question answering.
- **Multi-window Temporal Re-observation:** use coarse observations to select question-dependent temporal windows and reallocate the visual budget to those windows.
- **Structured Segment Memory:** divide each video into zero-based 60-second segments and store reusable structured records containing scenes, entities, actions, objects, visible text, counting cues, temporal cues, and retrieval keywords.
- **Question-aware Memory Retrieval:** retrieve a ranked set of seed segments using the question, answer options, and the memory bank of the current video.
- **Neighbor Expansion and Temporal Merging:** expand each seed to adjacent segments, clip the range to valid video boundaries, and merge overlapping segment groups.
- **Local Visual Re-observation:** sample original video frames from the merged temporal groups for fine-grained evidence acquisition.
- **Visual-grounded Final QA:** use the re-observed frames as the primary evidence, while retaining the selected memory records as auxiliary guidance for final multiple-choice prediction.

The core design principle: **structured memory determines where to look, neighbor expansion reduces temporal boundary misses, and the original local video frames provide the primary evidence for answering.**

---

## Main Results

Evaluation subset: `data/manifests/lvbench_official_30_fixed.json`, containing 30 QA items sampled from three LVBench long videos.

|            Method            | Memory |                             Tool                             |      Visual budget       |    Accuracy    |
| :--------------------------: | :----: | :----------------------------------------------------------: | :----------------------: | :------------: |
|         Uniform U32          |   No   |                 Uniform sampling + direct QA                 |        32 frames         | 10/30 = 33.33% |
|         Uniform U64          |   No   |                 Uniform sampling + direct QA                 |        64 frames         | 10/30 = 33.33% |
|      Multi-window Agent      |   No   |         8 coarse frames + 3 local windows × 8 frames         |        32 frames         | 13/30 = 43.33% |
| StructuredMemory + Expansion |  Yes   | Memory retrieval + neighbor expansion + local re-observation | 24 local frames + memory | 16/30 = 53.33% |

Temporal evidence coverage for `Structured Memory + Expansion`:

|         Metric          |     Value      |
| :---------------------: | :------------: |
| Seed retrieval hit rate | 20/30 = 66.67% |
|    Expanded hit rate    | 25/30 = 83.33% |
|     Final accuracy      | 16/30 = 53.33% |

These results show that simply increasing uniform frames from 32 to 64 does not improve the baseline, while question-aware temporal re-observation and structured memory improve both accuracy and evidence coverage.

---

## Repository Structure

```text
real-video-agent/
├── data/manifests/                         # Evaluation manifests, no videos
│   ├── lvbench_official_30_fixed.json
│   ├── lvbench_official_30_stats.json
│   └── official_30_by_video/
├── docs/                                   # Report materials and result notes
├── memory/README.md                        # Memory files are generated locally
├── outputs/README.md                       # Prediction outputs are generated locally
├── scripts/                                # Dataset, download, memory, evaluation utilities
└── src/                                    # Agent and baseline implementation
```

---

## Environment

Recommended environment used in the experiments:

- Python 3.10
- PyTorch with CUDA
- Qwen3-VL compatible `transformers`
- `decord`, `Pillow`, `datasets`, `huggingface_hub`

Install minimal dependencies:

```bash
pip install -r requirements.txt
```

The experiments were run with a local Qwen3-VL-8B-Instruct checkpoint. Set:

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export VIDEO_ROOT=/path/to/LVBench/videos
export GPU=0
```

---

## Data Preparation

### 1. Build or reuse the official-30 manifest

The submitted repository already includes:

```text
data/manifests/lvbench_official_30_fixed.json
```

To rebuild from HuggingFace LVBench:

```bash
python scripts/build_lvbench_official_subset.py --hf-name lmms-lab/LVBench --split auto --video-root "$VIDEO_ROOT" --num-videos 3 --per-video 10 --seed 2026 --output data/manifests/lvbench_official_30.json

python scripts/fix_lvbench_manifest_options.py --input data/manifests/lvbench_official_30.json --output data/manifests/lvbench_official_30_fixed.json
```

### 2. Download required videos

The LVBench videos are stored in chunked zip files. The low-disk downloader only extracts videos needed by the manifest and removes the zip after scanning.

```bash
python scripts/download_lvbench_needed_videos_resumable.py --manifest data/manifests/lvbench_official_30_fixed.json --video-dir "$VIDEO_ROOT" --start 1 --end 14
```

If a mirror fails, set one endpoint explicitly:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Run Baselines

### U32

```bash
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python src/baseline_u32.py --manifest data/manifests/lvbench_official_30_fixed.json --video-root "$VIDEO_ROOT" --model-path "$MODEL_PATH" --output outputs/preds/lvbench_u32_official_30.jsonl --num-frames 32 --image-size 448 --max-new-tokens 32 --dtype bf16 --limit 30
```

### U64

```bash
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python src/baseline_u32.py --manifest data/manifests/lvbench_official_30_fixed.json --video-root "$VIDEO_ROOT" --model-path "$MODEL_PATH" --output outputs/preds/lvbench_u64_official_30.jsonl --num-frames 64 --image-size 448 --max-new-tokens 32 --dtype bf16 --limit 30
```

---

## Run Multi-window Agent without Memory

```bash
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python src/agent32_multiwindow_wo_memory.py --manifest data/manifests/lvbench_official_30_fixed.json --video-root "$VIDEO_ROOT" --model-path "$MODEL_PATH" --output outputs/preds/lvbench_agent32_multiwindow_wo_memory_official_30.jsonl --coarse-frames 8 --windows 3 --frames-per-window 8 --image-size 448 --limit 30
```

---

## Build Structured Memory

Split the manifest by video:

```bash
python scripts/split_manifest_by_video.py --manifest data/manifests/lvbench_official_30_fixed.json --out-dir data/manifests/official_30_by_video
```

Build memory for each video:

```bash
bash scripts/run_build_memory_official30.sh
```

This creates:

```text
memory/lvbench_structured_segments_official_30/<video_key>_60s.json
```

Each segment stores structured fields: scene, characters, actions, objects, visible text, counting cues, temporal cues, and search keywords.

---

## Run StructuredMemory + Neighbor Expansion

```bash
bash scripts/run_expansion_official30.sh
```

This runs each video-specific manifest with its corresponding memory file, then merges the three JSONL files into:

```text
outputs/preds/lvbench_structured_memory_expand_official_30.jsonl
```

Analyze results:

```bash
python scripts/summarize_predictions.py --pred outputs/preds/lvbench_structured_memory_expand_official_30.jsonl

python scripts/analyze_expanded_segment_hit.py --manifest data/manifests/lvbench_official_30_fixed.json --pred outputs/preds/lvbench_structured_memory_expand_official_30.jsonl
```

---

## One-command Evaluation Helper

For convenience after videos and memory are ready:

```bash
bash scripts/run_official30_core_eval.sh
```

This runs U32, U64, Multi-window Agent, and summarizes all available predictions.

---

## AI Tool Usage

AI tools were used only as auxiliary support for code suggestions, debugging references, command organization, and language polishing. The overall system design, implementation decisions, experiment setup, execution, result checking, and analysis were completed and verified by the author.
