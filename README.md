# GOLF: Guidance-Optimized Learning with Feedback

<div align="center">

**Bootstrapping Exploration with Group-Level Natural Language Feedback in Reinforcement Learning**

[![Paper](https://img.shields.io/badge/paper-2603.04597-A42C25?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2603.04597) [![GitHub](https://img.shields.io/badge/GOLF-GitHub-181717?style=flat-square&logo=github)](https://github.com/your-org/GOLF)

</div>

**GOLF** is an RL framework that explicitly exploits **group-level language feedback** to guide targeted exploration through actionable refinements. It aggregates two complementary feedback sources: **(i)** external critiques that pinpoint errors or propose targeted fixes, and **(ii)** intra-group attempts that supply alternative partial ideas and diverse failure patterns. These group-level feedbacks are aggregated to produce high-quality **refinements**, which are **adaptively injected** into training as **off-policy scaffolds** to provide targeted guidance in sparse-reward regions. Meanwhile, GOLF **jointly optimizes generation and refinement** within a unified RL loop, creating a virtuous cycle that continuously improves both capabilities.

This repository supports **fuzzy** tasks (e.g. chat) and **verifiable** tasks (math, code, IF) with task-specific reward and critique pipelines built on [verl](https://github.com/verl-project/verl) and [AMPO](https://github.com/EnigmaYYYY/AMPO)-style hybrid GRPO.

---

## Table of Contents

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

---

## Installation

**Requirements:** Python 3.10, PyTorch, CUDA, vLLM (for rollout). We recommend a dedicated conda environment.

```bash
conda create -n golf python=3.10
conda activate golf
cd golf
cd verl
# For FSDP (no Megatron):
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
# For Megatron-backed training, use: bash scripts/install_vllm_sglang_mcore.sh
cd ..
pip install -r requirements.txt
```

---

## Repository Structure

```
GOLF/
├── golf/                    # Core training code (built on verl)
│   └── verl/
│       └── verl/adaptive_mix_src/   # GOLF trainer, critique refiner, reward
├── data/                    # Data preparation scripts
├── exp_scripts/             # Training launch scripts
│   ├── critique_grpo_hybrid_math.sh      # Verifiable: math
│   ├── critique_grpo_hybrid_if.sh        # Verifiable: instruction following (IF)
│   ├── critique_grpo_hybrid_code.sh      # Verifiable: code
│   └── critique_grpo_hybrid_wildchat.sh  # Fuzzy: wildchat / chat
├── eval_scripts/            # Evaluation and generation
│   ├── eval_math.sh         # Verifiable math eval
│   ├── eval_fuzzy.sh        # Fuzzy benchmark eval (RLMT-style)
│   ├── generate_vllm.py
│   └── ...
└── README.md
```

---

## Data Preparation

Data format and preprocessing per task:

| Task   | Reference |
|--------|-----------|
| **Fuzzy** (chat / instruction following) | [RLMT](https://github.com/princeton-pli/RLMT) |
| **Math** | [critique-GRPO](https://github.com/zhangxy-2019/critique-GRPO) |
| **Code** | [SDPO](https://github.com/lasgroup/SDPO) |
| **IF**   | [allenai/IF_multi_constraints_upto5](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5) |

---

## Training

Scripts assume repo root at `$PROJECT_ROOT/GOLF`. Set `PROJECT_ROOT`, `MODEL_PATH`, `TRAIN_FILE`, `TEST_FILE` (and for IF: `IFEVAL_VAL_FILE`, `IFBENCH_VAL_FILE`) as needed. Optional: `export WANDB_API_KEY=your_key` or `WANDB_MODE=disabled`.

Prepare data per task using the [Data Preparation](#data-preparation) references above, then run:

### Verifiable: Math

```bash
export PROJECT_ROOT=/path/to/your/projects
export MODEL_PATH=/path/to/pretrained_models/Qwen3-8B
export TRAIN_FILE=$PROJECT_ROOT/GOLF/data/openr1_math_4k_train.parquet
export TEST_FILE=$PROJECT_ROOT/GOLF/data/openr1_math_4k_test.parquet
bash exp_scripts/critique_grpo_hybrid_math.sh
```

### Verifiable: Instruction Following (IF)

```bash
export PROJECT_ROOT=/path/to/your/projects
export MODEL_PATH=/path/to/pretrained_models/Qwen3-4B
export TRAIN_FILE=$PROJECT_ROOT/GOLF/data/if_train.parquet
export IFEVAL_VAL_FILE=$PROJECT_ROOT/GOLF/data/ifeval_test.parquet
export IFBENCH_VAL_FILE=$PROJECT_ROOT/GOLF/data/ifbench_test.parquet
bash exp_scripts/critique_grpo_hybrid_if.sh
```

### Verifiable: Code

```bash
export PROJECT_ROOT=/path/to/your/projects
export MODEL_PATH=/path/to/pretrained_models/Qwen3-8B
export TRAIN_FILE=$PROJECT_ROOT/GOLF/data/lcb_v6_train.parquet
export TEST_FILE=$PROJECT_ROOT/GOLF/data/lcb_v6_test.parquet
bash exp_scripts/critique_grpo_hybrid_code.sh
```

### Fuzzy: Wildchat / chat

```bash
export PROJECT_ROOT=/path/to/your/projects
export MODEL_PATH=/path/to/pretrained_models/Llama-3.1-8B-Instruct
export TRAIN_FILE=$PROJECT_ROOT/GOLF/data/wildchat-if_train.parquet
export TEST_FILE=$PROJECT_ROOT/GOLF/data/wildchat-if_val.parquet
bash exp_scripts/critique_grpo_hybrid_wildchat.sh
```

Checkpoints: `$PROJECT_ROOT/GOLF/checkpoints/<model_name>/golf/<exp_name>/`. Merge FSDP shards via `eval_scripts/model_merge.sh` when needed.

---

## Evaluation

### Verifiable: Math

Run inference then score (e.g. with Math-Verify or your validator). Set `PROJECT_ROOT`, `EVAL_DATA`, `EVAL_OUTPUT_DIR`, and the `MODEL_PATHS` array to your merged checkpoints.

```bash
export PROJECT_ROOT=/path/to/your/projects
# Edit eval_scripts/eval_math.sh: set MODEL_PATHS and MODEL_NAMES to your checkpoints

bash eval_scripts/eval_math.sh
```

Then run your preferred math metric on the generated `*.jsonl` under `EVAL_OUTPUT_DIR`.

### Verifiable: Code / IF

Use the same pattern: point the eval scripts to your checkpoint dirs and data, run `generate_vllm.py` (or equivalent), then run task-specific scoring (e.g. pass@k for code, IFEval/IFBench for IF).

### Fuzzy (RLMT-style)

For fuzzy benchmarks (e.g. creative writing, WildBench, arena), use:

```bash
export PROJECT_ROOT=/path/to/your/projects
export OPENAI_BASE_URL=http://your-vllm-server:80/v1   # or local vLLM
# Edit eval_scripts/eval_fuzzy.sh: set MODELS, MODEL_NAMES, BENCHMARKS

bash eval_scripts/eval_fuzzy.sh
```

Benchmark list and scoring follow the same spirit as [RLMT](https://github.com/princeton-pli/RLMT); adjust `BENCHMARKS` and paths as in the script.

---

## Acknowledgement

GOLF builds on the following projects:

- **[AMPO](https://github.com/EnigmaYYYY/AMPO)** — “More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration”; we adopt and extend the adaptive multi-guidance and hybrid training ideas.
- **[verl](https://github.com/verl-project/verl)** — Volcano Engine Reinforcement Learning for LLMs; our training stack is built on verl’s GRPO/PPO and infrastructure.

We also thank [RLMT](https://github.com/princeton-pli/RLMT), [critique-GRPO](https://github.com/zhangxy-2019/critique-GRPO), and [SDPO](https://github.com/lasgroup/SDPO), [Math-Verify](https://github.com/huggingface/Math-Verify) for data, benchmarks, and tooling.

---

## Citation

If you use GOLF or this code, please cite:

```bibtex
@misc{huang2026bootstrappingexplorationgrouplevelnatural,
  title  = {Bootstrapping Exploration with Group-Level Natural Language Feedback in Reinforcement Learning},
  author = {Lei Huang and Xiang Cheng and Chenxiao Zhao and Guobin Shen and Junjie Yang and Xiaocheng Feng and Yuxuan Gu and Xing Yu and Bing Qin},
  year   = {2026},
  eprint = {2603.04597},
  archivePrefix = {arXiv},
  primaryClass   = {cs.CL},
  url    = {https://arxiv.org/abs/2603.04597},
}
```
