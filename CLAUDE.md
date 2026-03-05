# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **Multimodal LLM (MLLM) safety unlearning and alignment**, combining two projects:

- **VLGuard** (ICML 2024): Safety evaluation benchmark/dataset for Vision-Language Models — "Safety Fine-Tuning at (Almost) No Cost" (arXiv:2402.02207)
- **VLM-Safety-Unlearn** ("Safety Mirage"): Machine unlearning framework for VLMs built on LLaVA-1.5 — "How Spurious Correlations Undermine VLM Safety Fine-tuning" (arXiv:2503.11832)

## Repository Structure

- `VLGuard/` — Safety evaluation benchmark with 3 test subsets (`unsafes`, `safe_unsafes`, `safe_safes`) and GPT-4V helpfulness evaluation
- `VLM-Safety-Unlearn/` — Modified LLaVA fork with unlearning capabilities (RMU and NPO algorithms)
- `running_scripts/` — Local experiment runner scripts

## Environment Setup

```bash
conda create -n llava python=3.10 -y && conda activate llava
cd VLM-Safety-Unlearn
pip install -e . && pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Common Commands

### Training (Unlearning)
```bash
# Full-parameter unlearning (8 GPUs, DeepSpeed ZeRO-3)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/finetune_unlearn.sh

# LoRA unlearning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/finetune_unlearn_lora.sh
```

### Evaluation
```bash
# VLGuard eval (LLaVA) — dataset: unsafes | safe_unsafes | safe_safes
cd VLGuard
CUDA_VISIBLE_DEVICES=0 python VLGuard_eval.py --dataset unsafes --engine llava15-7b \
    --metaDir data/test.json --imageDir data/test

# VLGuard eval (Qwen3-VL)
CUDA_VISIBLE_DEVICES=0 python VLGuard_qwen3vl_eval.py --dataset unsafes \
    --engine Qwen3-VL-8B-Instruct --metaDir data/test.json --imageDir data/test

# GPT-4V helpfulness evaluation
OPENAI_API_KEY="..." python gpt4_evaluator.py \
    --file_path results/safe_safes/model.json --image_path data/test \
    --reference_path data/gpt4_safe_safes.json --output_path results/output.json
```

## Architecture

### Unlearning Algorithms (in `VLM-Safety-Unlearn/llava/train/`)

Two algorithms implemented, each with LoRA and full-parameter variants:

- **RMU (Representation Misdirection for Unlearning)**: Steers hidden activations at a target layer toward a random control vector on forget data, while preserving activations on retain data via MSE loss against a frozen reference model. Key args: `rmu_layer_id`, `rmu_steering_coeffs`, `rmu_retain_alpha`, `rmu_llava_loss_weight`.

- **NPO (Negative Preference Optimization)**: Uses log-ratio between current and frozen reference model on forget data with logsigmoid loss. Key args: `npo_beta`, `npo_forget_alpha`, `npo_retain_alpha`, `npo_llava_loss_weight`.

### Key Training Files

| File | Purpose |
|------|---------|
| `train_unlearn.py` / `train_unlearn_full.py` | Entry points for LoRA / full-param unlearning |
| `train_unlearn_mem.py` / `train_unlearn_full_mem.py` | Memory-efficient variants |
| `llava_unlearn_trainer.py` / `llava_unlearn_full_trainer.py` | Custom Trainer classes with RMU/NPO loss |
| `llava_trainer.py` | Standard LLaVA trainer |

### Training Data Flow

Training loads two datasets: **retain_data** (safe data to preserve) and **forget_data** (unsafe data to unlearn). Combined loss = `unlearn_loss + retain_loss + llava_loss_weight * standard_llava_loss`.

### Model Architecture (LLaVA)

LLaMA backbone + CLIP vision encoder (`openai/clip-vit-large-patch14-336`) + MLP projector. Base model: `liuhaotian/llava-v1.5-7b`. A frozen copy is maintained for reference losses.

## Technical Notes

- Uses **Git LFS** for large files (model weights, images, datasets)
- `VLGuard/data/train` and `VLGuard/data/test` are gitignored
- Training uses **DeepSpeed ZeRO-3** across 8 GPUs
- Supports **wandb** reporting (`--report_to wandb`)
- No test suites or CI/CD — this is a research codebase
