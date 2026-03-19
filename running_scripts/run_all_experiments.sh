#!/bin/bash
# =============================================================================
# run_all_experiments.sh
#
# Full pipeline: train L-NPO unlearning, merge LoRA checkpoints, then evaluate
# all 5 models (original + 4 unlearned variants) on:
#   1. Sorry-Bench  (ASR / Attack Success Rate)
#   2. BeaverTails  (unsafe_rate / safe_rate)
#   3. MMLU-Redux   (accuracy)
#
# Available GPUs: 1, 2, 3, 4, 6, 7
#   - Training:   4 GPUs  (1,2,3,4)
#   - Evaluation: 1 GPU each
#
# Models
# -------
#   qwen3-vl-2b           Qwen/Qwen3-VL-2B-Instruct  (original baseline)
#   qwen3-vl-2b-npo       V+L NPO unlearned  (already merged)
#   qwen3-vl-2b-rmu       V+L RMU unlearned  (already merged)
#   qwen3-vl-2b-l-rmu     L   RMU unlearned  (LoRA → needs merge)
#   qwen3-vl-2b-l-npo     L   NPO unlearned  (needs training → merge)
#
# Fast validation mode (set FAST_VALIDATION=1 or pass --fast):
#   - Training:    2 epochs instead of 10
#   - Sorry-Bench: 1st question per category  (44 total, vs 440)
#   - BeaverTails: --max_samples 100          (vs full ~3k)
#   - MMLU-Redux:  --max_samples 20           (vs full ~14k)
#   Results saved under results/fast/ to avoid overwriting full runs.
#
# Usage:
#   bash run_all_experiments.sh          # full run
#   bash run_all_experiments.sh --fast   # fast validation
#   FAST_VALIDATION=1 bash run_all_experiments.sh
# =============================================================================

set -euo pipefail

# ── fast-validation flag ──────────────────────────────────────────────────────
FAST_VALIDATION="${FAST_VALIDATION:-0}"
if [[ "${1:-}" == "--fast" ]]; then FAST_VALIDATION=1; fi

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME=/playpen-shared/pengjie
export NCCL_P2P_DISABLE=1

cd "$PROJ_ROOT"

# ── model paths ───────────────────────────────────────────────────────────────
MODEL_BASE="Qwen/Qwen3-VL-2B-Instruct"

CKPT_VL_NPO="$PROJ_ROOT/VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-lora-merged"
CKPT_VL_RMU="$PROJ_ROOT/VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-lora-rmu-merged"
CKPT_L_RMU_LORA="$PROJ_ROOT/VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-text-lora-rmu"
CKPT_L_RMU="$PROJ_ROOT/VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-text-lora-rmu-merged"
CKPT_L_NPO_LORA="$PROJ_ROOT/VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-text-lora-npo"
CKPT_L_NPO="$PROJ_ROOT/VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-text-lora-npo-merged"

# ── model IDs (used for output file naming) ───────────────────────────────────
ID_ORIG="qwen3-vl-2b"
ID_VL_NPO="qwen3-vl-2b-npo"   # matches existing sorry-bench answer/judgment files
ID_VL_RMU="qwen3-vl-2b-rmu"   # matches existing sorry-bench answer/judgment files
ID_L_RMU="qwen3-vl-2b-l-rmu"
ID_L_NPO="qwen3-vl-2b-l-npo"

ALL_IDS=("$ID_ORIG" "$ID_VL_NPO" "$ID_VL_RMU" "$ID_L_RMU" "$ID_L_NPO")

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── fast-validation settings ──────────────────────────────────────────────────
if [[ "$FAST_VALIDATION" == "1" ]]; then
    RESULTS_DIR="$PROJ_ROOT/results/fast"
    NUM_TRAIN_EPOCHS=1
    # sorry-bench: first question per category = questions 1,11,21,... (step 10)
    # Simplest approximation: take the first 44 questions (indices 0-43)
    SORRY_QBEGIN=0
    SORRY_QEND=44
    # beavertails / mmlu sample caps
    BEAVERTAILS_EXTRA="--max_samples 100"
    MMLU_EXTRA="--max_samples 20"
    log "*** FAST VALIDATION MODE: epochs=${NUM_TRAIN_EPOCHS}, sorry-bench Q[${SORRY_QBEGIN}:${SORRY_QEND}], beavertails/mmlu capped ***"
else
    RESULTS_DIR="$PROJ_ROOT/results"
    NUM_TRAIN_EPOCHS=10
    SORRY_QBEGIN=""   # empty → use full dataset
    SORRY_QEND=""
    BEAVERTAILS_EXTRA=""
    MMLU_EXTRA=""
fi

# ── output directories ────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR/sorry_bench" "$RESULTS_DIR/beavertails" "$RESULTS_DIR/mmlu_redux"

# =============================================================================
# PHASE 1 – Train L-text NPO unlearning
# =============================================================================
log "=== PHASE 1: Training L-text NPO ==="

if [ -f "$CKPT_L_NPO_LORA/adapter_model.safetensors" ] || \
   [ -f "$CKPT_L_NPO_LORA/adapter_model.bin" ]; then
    log "Skipping training – L-NPO LoRA checkpoint already exists."
else
    export CUDA_VISIBLE_DEVICES=1,2,3,4
    cd "$PROJ_ROOT/VLM-Safety-Unlearn"
    deepspeed --master_port 29504 qwen3vl_train/train_unlearn_text.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 \
        --deepspeed ./scripts/zero2.json \
        --model_name_or_path "$MODEL_BASE" \
        --retain_data_path "$PROJ_ROOT/VLGuard/data/retain_data.json" \
        --forget_data_path "$PROJ_ROOT/VLGuard/data/forget_data.json" \
        --caption_folder "$PROJ_ROOT/VLGuard/data" \
        --max_pixels 1003520 \
        --min_pixels 3136 \
        --tune_mm_llm True \
        --tune_mm_vision False \
        --tune_mm_mlp False \
        --bf16 True \
        --output_dir "$CKPT_L_NPO_LORA" \
        --group_by_modality_length False \
        --num_train_epochs "$NUM_TRAIN_EPOCHS" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 1.1e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --report_to none \
        --unlearn_type "npo" \
        --rmu_layer_id 15 \
        --rmu_steering_coeffs "10" \
        --rmu_llava_loss_weight 1.2 \
        --rmu_retain_alpha 0 \
        --npo_beta 0.7 \
        --npo_forget_alpha 1.0 \
        --npo_llava_loss_weight 1.0 \
        --verbose True \
        $( [[ "$FAST_VALIDATION" == "1" ]] && echo "--max_train_samples 20" )
    cd "$PROJ_ROOT"
fi

# =============================================================================
# PHASE 2 – Merge LoRA checkpoints
# =============================================================================
log "=== PHASE 2: Merging LoRA checkpoints ==="

if [ ! -d "$CKPT_L_RMU" ] || [ -z "$(ls -A "$CKPT_L_RMU" 2>/dev/null)" ]; then
    log "Merging L-RMU LoRA → $CKPT_L_RMU"
    python VLM-Safety-Unlearn/scripts/merge_lora_weights.py \
        --model-path "$CKPT_L_RMU_LORA" \
        --model-base "$MODEL_BASE" \
        --save-model-path "$CKPT_L_RMU"
else
    log "Skipping L-RMU merge – already exists."
fi

if [ ! -d "$CKPT_L_NPO" ] || [ -z "$(ls -A "$CKPT_L_NPO" 2>/dev/null)" ]; then
    log "Merging L-NPO LoRA → $CKPT_L_NPO"
    python VLM-Safety-Unlearn/scripts/merge_lora_weights.py \
        --model-path "$CKPT_L_NPO_LORA" \
        --model-base "$MODEL_BASE" \
        --save-model-path "$CKPT_L_NPO"
else
    log "Skipping L-NPO merge – already exists."
fi

# =============================================================================
# PHASE 3 – Sorry-Bench: generate answers (parallel, 5 GPUs)
#   + BeaverTails for original (GPU 7, runs concurrently)
# =============================================================================
log "=== PHASE 3: Sorry-Bench answer generation + BeaverTails (original) ==="

SORRY_Q="$PROJ_ROOT/dataset/sorry_bench/question.jsonl"
SORRY_ANSWER_DIR="$PROJ_ROOT/dataset/sorry_bench/model_answer"
mkdir -p "$SORRY_ANSWER_DIR"

# Build optional sorry-bench range args
_sorry_range_args() {
    if [[ -n "$SORRY_QBEGIN" ]]; then
        echo "--question-begin $SORRY_QBEGIN --question-end $SORRY_QEND"
    fi
}

_gen_sorry_answer() {
    local gpu=$1 model_path=$2 model_id=$3
    local out="$SORRY_ANSWER_DIR/${model_id}.jsonl"
    # In fast mode always regenerate (subset differs from full run)
    if [[ "$FAST_VALIDATION" != "1" ]] && [ -f "$out" ]; then
        log "  [sorry-bench answer] Skipping $model_id – already exists."
        return
    fi
    log "  [sorry-bench answer] GPU $gpu: $model_id"
    # shellcheck disable=SC2046
    CUDA_VISIBLE_DEVICES=$gpu python text_safety_bench/sorry-bench/gen_qwen3vl_answer.py \
        --model-path "$model_path" \
        --model-id "$model_id" \
        --question-file "$SORRY_Q" \
        --answer-file "$out" \
        --max-new-tokens 1024 \
        --num-gpus 1
}

# Original model (GPU 1)
_gen_sorry_answer 1 "$MODEL_BASE" "$ID_ORIG" &
# V+L NPO (GPU 2)
_gen_sorry_answer 2 "$CKPT_VL_NPO" "$ID_VL_NPO" &
# V+L RMU (GPU 3)
_gen_sorry_answer 3 "$CKPT_VL_RMU" "$ID_VL_RMU" &
# L RMU (GPU 4)
_gen_sorry_answer 4 "$CKPT_L_RMU" "$ID_L_RMU" &
# L NPO (GPU 6)
_gen_sorry_answer 6 "$CKPT_L_NPO" "$ID_L_NPO" &

# BeaverTails – original (GPU 7, independent)
log "  [beavertails] GPU 7: $ID_ORIG"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=7 python beavertails_qwen3vl_eval.py \
    --engine "$ID_ORIG" \
    --model_path "$MODEL_BASE" \
    --output_dir "$RESULTS_DIR/beavertails" \
    $BEAVERTAILS_EXTRA &

wait
log "Phase 3 complete."

# =============================================================================
# PHASE 4 – Sorry-Bench: judgment (GPU 1) + BeaverTails for unlearned (GPUs 2-6)
#           + MMLU-Redux for original (GPU 7)
# =============================================================================
log "=== PHASE 4: Sorry-Bench judgment + BeaverTails unlearned + MMLU original ==="

# Sorry-Bench judgment for all 5 models (GPU 1)
log "  [sorry-bench judgment] GPU 1: all models"
(
    cd "$PROJ_ROOT/text_safety_bench/sorry-bench"
    CUDA_VISIBLE_DEVICES=1 python gen_judgment_safety_vllm.py \
        --bench-name sorry_bench \
        --judge-model ft-mistral-7b-instruct-v0.2 \
        --model-list "$ID_ORIG" "$ID_VL_NPO" "$ID_VL_RMU" "$ID_L_RMU" "$ID_L_NPO"
) &

# BeaverTails – unlearned models (GPUs 2, 3, 4, 6)
for entry in "2:$ID_VL_NPO:$CKPT_VL_NPO" "3:$ID_VL_RMU:$CKPT_VL_RMU" \
             "4:$ID_L_RMU:$CKPT_L_RMU" "6:$ID_L_NPO:$CKPT_L_NPO"; do
    IFS=: read -r gpu mid mpath <<< "$entry"
    log "  [beavertails] GPU $gpu: $mid"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu python beavertails_qwen3vl_eval.py \
        --engine "$mid" \
        --model_path "$mpath" \
        --output_dir "$RESULTS_DIR/beavertails" \
        $BEAVERTAILS_EXTRA &
done

# MMLU-Redux – original (GPU 7)
log "  [mmlu-redux] GPU 7: $ID_ORIG"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=7 python mmlu_redux_qwen3vl_eval.py \
    --engine "$ID_ORIG" \
    --model_path "$MODEL_BASE" \
    --output_dir "$RESULTS_DIR/mmlu_redux" \
    $MMLU_EXTRA &

wait
log "Phase 4 complete."

# =============================================================================
# PHASE 5 – Sorry-Bench: compute metrics + MMLU-Redux for unlearned models
# =============================================================================
log "=== PHASE 5: Sorry-Bench metrics + MMLU unlearned ==="

# Compute sorry-bench metrics (CPU)
log "  [sorry-bench metrics] computing..."
python running_scripts/compute_sorry_bench_metrics.py \
    --judgment_file text_safety_bench/sorry-bench/data/sorry_bench/model_judgment/ft-mistral-7b-instruct-v0.2.jsonl \
    --question_file dataset/sorry_bench/question.jsonl \
    --model_ids "${ALL_IDS[@]}" \
    --output_file "$RESULTS_DIR/sorry_bench/sorry_bench_metrics.json" &

# MMLU-Redux – unlearned models (GPUs 1, 2, 3, 4)
# MMLU-Redux – unlearned models (GPUs 1, 2, 3, 4)
for entry in "1:$ID_VL_NPO:$CKPT_VL_NPO" "2:$ID_VL_RMU:$CKPT_VL_RMU" \
             "3:$ID_L_RMU:$CKPT_L_RMU" "4:$ID_L_NPO:$CKPT_L_NPO"; do
    IFS=: read -r gpu mid mpath <<< "$entry"
    log "  [mmlu-redux] GPU $gpu: $mid"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu python mmlu_redux_qwen3vl_eval.py \
        --engine "$mid" \
        --model_path "$mpath" \
        --output_dir "$RESULTS_DIR/mmlu_redux" \
        $MMLU_EXTRA &
done

wait
log "Phase 5 complete."

# =============================================================================
# Done
# =============================================================================
log "=== ALL EXPERIMENTS COMPLETE ==="
log "Results:"
log "  Sorry-Bench:  $RESULTS_DIR/sorry_bench/"
log "  BeaverTails:  $RESULTS_DIR/beavertails/"
log "  MMLU-Redux:   $RESULTS_DIR/mmlu_redux/"
