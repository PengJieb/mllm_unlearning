#!/usr/bin/env bash
# =============================================================================
# run_qwen3vl_rmu_sweep.sh
#
# Train Qwen3-VL-2B with RMU and RMU (Text) over configurable hyper-parameter
# sweeps, merge each LoRA checkpoint, then evaluate every merged checkpoint on:
#   1. Sorry-Bench
#   2. BeaverTails
#   3. MMLU-Redux
#
# Usage:
#   bash running_scripts/run_qwen3vl_rmu_sweep.sh
#   bash running_scripts/run_qwen3vl_rmu_sweep.sh --fast
#
# Useful overrides:
#   TRAIN_GPUS="1,2,3,4" EVAL_GPUS="5 6 7" JUDGE_GPU=7 \
#   TRAIN_EPOCHS="3 5 10" VL_RMU_LEARNING_RATES="5e-6 1e-5" \
#   TEXT_RMU_LEARNING_RATES="5e-6 1e-5" \
#   bash running_scripts/run_qwen3vl_rmu_sweep.sh
#
#   ENABLE_VL_RMU=0 ENABLE_TEXT_RMU=1 \
#   TRAIN_EPOCHS="3 5" TEXT_RMU_LEARNING_RATES="1e-5 2e-5" \
#   bash running_scripts/run_qwen3vl_rmu_sweep.sh
# =============================================================================

set -euo pipefail

FAST_VALIDATION="${FAST_VALIDATION:-0}"
if [[ "${1:-}" == "--fast" ]]; then
    FAST_VALIDATION=1
fi

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME=/playpen-shared/pengjie
export NCCL_P2P_DISABLE=1

cd "$PROJ_ROOT"

MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-VL-2B-Instruct}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
EVAL_GPUS="${EVAL_GPUS:-1}"
JUDGE_GPU="${JUDGE_GPU:-6}"
JUDGE_MODEL="${JUDGE_MODEL:-ft-mistral-7b-instruct-v0.2}"
TRAIN_MASTER_PORT_BASE="${TRAIN_MASTER_PORT_BASE:-29520}"

ENABLE_VL_RMU="${ENABLE_VL_RMU:-1}"
ENABLE_TEXT_RMU="${ENABLE_TEXT_RMU:-1}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJ_ROOT/VLM-Safety-Unlearn/checkpoints/qwen3vl_rmu_sweep}"
if [[ "$FAST_VALIDATION" == "1" ]]; then
    RESULTS_DIR="${RESULTS_DIR:-$PROJ_ROOT/results/fast/qwen3vl_rmu_sweep}"
    TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
    SORRY_QBEGIN="${SORRY_QBEGIN:-0}"
    SORRY_QEND="${SORRY_QEND:-44}"
    BEAVERTAILS_MAX_SAMPLES="${BEAVERTAILS_MAX_SAMPLES:-100}"
    MMLU_MAX_SAMPLES="${MMLU_MAX_SAMPLES:-20}"
    TEXT_MAX_TRAIN_SAMPLES="${TEXT_MAX_TRAIN_SAMPLES:-20}"
else
    RESULTS_DIR="${RESULTS_DIR:-$PROJ_ROOT/results/qwen3vl_rmu_sweep}"
    TRAIN_EPOCHS="2 5"
    SORRY_QBEGIN="${SORRY_QBEGIN:-}"
    SORRY_QEND="${SORRY_QEND:-}"
    BEAVERTAILS_MAX_SAMPLES="${BEAVERTAILS_MAX_SAMPLES:-}"
    MMLU_MAX_SAMPLES="${MMLU_MAX_SAMPLES:-}"
    TEXT_MAX_TRAIN_SAMPLES="${TEXT_MAX_TRAIN_SAMPLES:-}"
fi

VL_RMU_LAYER="${VL_RMU_LAYER:-15}"
VL_RMU_STEERING_COEFF="${VL_RMU_STEERING_COEFF:-10}"
VL_RMU_LLAVA_LOSS_WEIGHT="${VL_RMU_LLAVA_LOSS_WEIGHT:-1.2}"
VL_RMU_RETAIN_ALPHA="${VL_RMU_RETAIN_ALPHA:-0}"
VL_RMU_LEARNING_RATES="${VL_RMU_LEARNING_RATES:-1.1e-5}"

TEXT_RMU_LAYER="${TEXT_RMU_LAYER:-15}"
TEXT_RMU_STEERING_COEFF="${TEXT_RMU_STEERING_COEFF:-10}"
TEXT_RMU_LLAVA_LOSS_WEIGHT="${TEXT_RMU_LLAVA_LOSS_WEIGHT:-1.2}"
TEXT_RMU_RETAIN_ALPHA="${TEXT_RMU_RETAIN_ALPHA:-0}"
TEXT_RMU_LEARNING_RATES="1.1e-5 1e-5 5e-6"

SORRY_Q="${SORRY_Q:-$PROJ_ROOT/dataset/sorry_bench/question.jsonl}"
SORRY_ANSWER_DIR="${SORRY_ANSWER_DIR:-$PROJ_ROOT/dataset/sorry_bench/model_answer}"
SORRY_JUDGMENT_FILE="${SORRY_JUDGMENT_FILE:-$PROJ_ROOT/text_safety_bench/sorry-bench/data/sorry_bench/model_judgment/${JUDGE_MODEL}.jsonl}"
MANIFEST_FILE="$RESULTS_DIR/experiment_manifest.tsv"

mkdir -p \
    "$CHECKPOINT_ROOT" \
    "$RESULTS_DIR/sorry_bench" \
    "$RESULTS_DIR/beavertails" \
    "$RESULTS_DIR/mmlu_redux" \
    "$SORRY_ANSWER_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

slugify() {
    local value="$1"
    value="${value//./p}"
    value="${value//\//-}"
    value="${value//-/m}"
    value="${value//,/x}"
    value="${value// /_}"
    echo "$value"
}

parse_array_var() {
    local var_name="$1"
    local -n out_ref="$2"
    local raw="${!var_name}"
    read -r -a out_ref <<< "$raw"
}

pick_training_port() {
    PICKED_TRAIN_PORT="$TRAIN_MASTER_PORT_BASE"
    TRAIN_MASTER_PORT_BASE=$((TRAIN_MASTER_PORT_BASE + 1))
}

append_experiment() {
    local model_id="$1"
    local model_path="$2"
    local family="$3"
    local epochs="$4"
    local lr="$5"

    ALL_MODEL_IDS+=("$model_id")
    ALL_MODEL_PATHS+=("$model_path")
    printf "%s\t%s\t%s\t%s\t%s\n" \
        "$model_id" "$family" "$epochs" "$lr" "$model_path" >> "$MANIFEST_FILE"
}

train_one_experiment() {
    local family="$1"
    local epochs="$2"
    local lr="$3"

    local family_label train_script output_prefix group_by_modality data_flag data_value
    local layer coeff llava_weight retain_alpha
    if [[ "$family" == "vl-rmu" ]]; then
        family_label="rmu"
        train_script="qwen3vl_train/train_unlearn.py"
        output_prefix="qwen3-vl-2b-rmu"
        group_by_modality="True"
        data_flag="--image_folder"
        data_value="$PROJ_ROOT/VLGuard/data/train/"
        layer="$VL_RMU_LAYER"
        coeff="$VL_RMU_STEERING_COEFF"
        llava_weight="$VL_RMU_LLAVA_LOSS_WEIGHT"
        retain_alpha="$VL_RMU_RETAIN_ALPHA"
    else
        family_label="text-rmu"
        train_script="qwen3vl_train/train_unlearn_text.py"
        output_prefix="qwen3-vl-2b-rmu-text"
        group_by_modality="False"
        data_flag="--caption_folder"
        data_value="$PROJ_ROOT/VLGuard/data"
        layer="$TEXT_RMU_LAYER"
        coeff="$TEXT_RMU_STEERING_COEFF"
        llava_weight="$TEXT_RMU_LLAVA_LOSS_WEIGHT"
        retain_alpha="$TEXT_RMU_RETAIN_ALPHA"
    fi

    local exp_id="${output_prefix}-ep$(slugify "$epochs")-lr$(slugify "$lr")"
    local lora_dir="$CHECKPOINT_ROOT/${exp_id}-lora"
    local merged_dir="$CHECKPOINT_ROOT/${exp_id}"
    local port
    pick_training_port
    port="$PICKED_TRAIN_PORT"

    if [[ -f "$lora_dir/adapter_model.safetensors" || -f "$lora_dir/adapter_model.bin" ]]; then
        log "Skipping training for $exp_id (LoRA checkpoint exists)."
    else
        log "Training $exp_id on GPUs $TRAIN_GPUS"
        (
            cd "$PROJ_ROOT/VLM-Safety-Unlearn"
            cmd=(
                deepspeed --master_port "$port" "$train_script"
                --lora_enable True --lora_r 128 --lora_alpha 256
                --deepspeed ./scripts/zero2.json
                --model_name_or_path "$MODEL_BASE"
                --retain_data_path "$PROJ_ROOT/VLGuard/data/retain_data.json"
                --forget_data_path "$PROJ_ROOT/VLGuard/data/forget_data.json"
                "$data_flag" "$data_value"
                --max_pixels 1003520
                --min_pixels 3136
                --tune_mm_llm True
                --tune_mm_vision False
                --tune_mm_mlp False
                --bf16 True
                --output_dir "$lora_dir"
                --group_by_modality_length "$group_by_modality"
                --num_train_epochs "$epochs"
                --per_device_train_batch_size 4
                --per_device_eval_batch_size 1
                --gradient_accumulation_steps 1
                --save_strategy steps
                --save_steps 50000
                --save_total_limit 1
                --learning_rate "$lr"
                --weight_decay 0.
                --warmup_ratio 0.03
                --lr_scheduler_type cosine
                --logging_steps 1
                --tf32 True
                --model_max_length 2048
                --gradient_checkpointing True
                --dataloader_num_workers 4
                --report_to none
                --unlearn_type rmu
                --rmu_layer_id "$layer"
                --rmu_steering_coeffs "$coeff"
                --rmu_llava_loss_weight "$llava_weight"
                --rmu_retain_alpha "$retain_alpha"
                --npo_beta 0.7
                --npo_forget_alpha 1.0
                --npo_llava_loss_weight 1.0
                --verbose True
                --loss_dir "$lora_dir/losses"
            )
            if [[ "$family" == "text-rmu" && -n "$TEXT_MAX_TRAIN_SAMPLES" ]]; then
                cmd+=(--max_train_samples "$TEXT_MAX_TRAIN_SAMPLES")
            fi
            CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "${cmd[@]}"
        )
    fi

    if [[ -d "$merged_dir" && -n "$(ls -A "$merged_dir" 2>/dev/null)" ]]; then
        log "Skipping merge for $exp_id (merged checkpoint exists)."
    else
        log "Merging LoRA for $exp_id"
        python "$PROJ_ROOT/VLM-Safety-Unlearn/scripts/merge_lora_weights.py" \
            --model-path "$lora_dir" \
            --model-base "$MODEL_BASE" \
            --save-model-path "$merged_dir"
    fi

    append_experiment "$exp_id" "$merged_dir" "$family_label" "$epochs" "$lr"
}

run_sorry_answers_batch() {
    local count="${#ALL_MODEL_IDS[@]}"
    local gpu_count="${#EVAL_GPU_LIST[@]}"
    local start end idx batch_offset gpu model_id model_path out_file

    for ((start = 0; start < count; start += gpu_count)); do
        end=$((start + gpu_count))
        if (( end > count )); then
            end="$count"
        fi

        for ((idx = start; idx < end; idx++)); do
            batch_offset=$((idx - start))
            gpu="${EVAL_GPU_LIST[$batch_offset]}"
            model_id="${ALL_MODEL_IDS[$idx]}"
            model_path="${ALL_MODEL_PATHS[$idx]}"
            out_file="$SORRY_ANSWER_DIR/${model_id}.jsonl"

            log "  [sorry-bench answer] GPU $gpu: $model_id"
            cmd=(
                python "$PROJ_ROOT/text_safety_bench/sorry-bench/gen_qwen3vl_answer.py"
                --model-path "$model_path"
                --model-id "$model_id"
                --question-file "$SORRY_Q"
                --answer-file "$out_file"
                --max-new-tokens 1024
                --num-gpus 1
            )
            if [[ -n "$SORRY_QBEGIN" ]]; then
                cmd+=(--question-begin "$SORRY_QBEGIN" --question-end "$SORRY_QEND")
            fi
            CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
        done
        wait
    done
}

run_beavertails_batch() {
    local count="${#ALL_MODEL_IDS[@]}"
    local gpu_count="${#EVAL_GPU_LIST[@]}"
    local start end idx batch_offset gpu model_id model_path

    for ((start = 0; start < count; start += gpu_count)); do
        end=$((start + gpu_count))
        if (( end > count )); then
            end="$count"
        fi

        for ((idx = start; idx < end; idx++)); do
            batch_offset=$((idx - start))
            gpu="${EVAL_GPU_LIST[$batch_offset]}"
            model_id="${ALL_MODEL_IDS[$idx]}"
            model_path="${ALL_MODEL_PATHS[$idx]}"

            log "  [beavertails] GPU $gpu: $model_id"
            cmd=(
                python "$PROJ_ROOT/beavertails_qwen3vl_eval.py"
                --engine "$model_id"
                --model_path "$model_path"
                --output_dir "$RESULTS_DIR/beavertails"
            )
            if [[ -n "$BEAVERTAILS_MAX_SAMPLES" ]]; then
                cmd+=(--max_samples "$BEAVERTAILS_MAX_SAMPLES")
            fi
            CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
        done
        wait
    done
}

run_mmlu_batch() {
    local count="${#ALL_MODEL_IDS[@]}"
    local gpu_count="${#EVAL_GPU_LIST[@]}"
    local start end idx batch_offset gpu model_id model_path

    for ((start = 0; start < count; start += gpu_count)); do
        end=$((start + gpu_count))
        if (( end > count )); then
            end="$count"
        fi

        for ((idx = start; idx < end; idx++)); do
            batch_offset=$((idx - start))
            gpu="${EVAL_GPU_LIST[$batch_offset]}"
            model_id="${ALL_MODEL_IDS[$idx]}"
            model_path="${ALL_MODEL_PATHS[$idx]}"

            log "  [mmlu-redux] GPU $gpu: $model_id"
            cmd=(
                python "$PROJ_ROOT/mmlu_redux_qwen3vl_eval.py"
                --engine "$model_id"
                --model_path "$model_path"
                --output_dir "$RESULTS_DIR/mmlu_redux"
            )
            if [[ -n "$MMLU_MAX_SAMPLES" ]]; then
                cmd+=(--max_samples "$MMLU_MAX_SAMPLES")
            fi
            CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" &
        done
        wait
    done
}

parse_array_var "TRAIN_EPOCHS" TRAIN_EPOCH_LIST
parse_array_var "VL_RMU_LEARNING_RATES" VL_RMU_LR_LIST

parse_array_var "TEXT_RMU_LEARNING_RATES" TEXT_RMU_LR_LIST
parse_array_var "EVAL_GPUS" EVAL_GPU_LIST

if [[ "${#EVAL_GPU_LIST[@]}" -eq 0 ]]; then
    echo "EVAL_GPUS must contain at least one GPU ID." >&2
    exit 1
fi

: > "$MANIFEST_FILE"
printf "model_id\tfamily\tnum_train_epochs\tlearning_rate\tmodel_path\n" > "$MANIFEST_FILE"

ALL_MODEL_IDS=()
ALL_MODEL_PATHS=()

log "=== Qwen3-VL-2B RMU sweep ==="
log "Training GPUs: $TRAIN_GPUS"
log "Eval GPUs: ${EVAL_GPU_LIST[*]}"
log "Judge GPU: $JUDGE_GPU"
log "Train epochs: ${TRAIN_EPOCH_LIST[*]}"
log "VL-RMU fixed params: layer=$VL_RMU_LAYER coeff=$VL_RMU_STEERING_COEFF llava_weight=$VL_RMU_LLAVA_LOSS_WEIGHT retain_alpha=$VL_RMU_RETAIN_ALPHA"
log "VL-RMU learning rates: ${VL_RMU_LR_LIST[*]}"
log "Text-RMU fixed params: layer=$TEXT_RMU_LAYER coeff=$TEXT_RMU_STEERING_COEFF llava_weight=$TEXT_RMU_LLAVA_LOSS_WEIGHT retain_alpha=$TEXT_RMU_RETAIN_ALPHA"
log "Text-RMU learning rates: ${TEXT_RMU_LR_LIST[*]}"

if [[ "$ENABLE_VL_RMU" == "1" ]]; then
    for epochs in "${TRAIN_EPOCH_LIST[@]}"; do
        for lr in "${VL_RMU_LR_LIST[@]}"; do
            train_one_experiment "vl-rmu" "$epochs" "$lr"
        done
    done
fi

if [[ "$ENABLE_TEXT_RMU" == "1" ]]; then
    for epochs in "${TRAIN_EPOCH_LIST[@]}"; do
        for lr in "${TEXT_RMU_LR_LIST[@]}"; do
            train_one_experiment "text-rmu" "$epochs" "$lr"
        done
    done
fi

if [[ "${#ALL_MODEL_IDS[@]}" -eq 0 ]]; then
    echo "No experiments were scheduled. Set ENABLE_VL_RMU=1 and/or ENABLE_TEXT_RMU=1." >&2
    exit 1
fi

log "=== Phase 1: Sorry-Bench answers ==="
run_sorry_answers_batch

log "=== Phase 2: Sorry-Bench judgment ==="
(
    cd "$PROJ_ROOT/text_safety_bench/sorry-bench"
    CUDA_VISIBLE_DEVICES="$JUDGE_GPU" python gen_judgment_safety_vllm.py \
        --bench-name sorry_bench \
        --judge-model "$JUDGE_MODEL" \
        --model-list "${ALL_MODEL_IDS[@]}"
)

log "=== Phase 3: Sorry-Bench metrics ==="
python "$PROJ_ROOT/running_scripts/compute_sorry_bench_metrics.py" \
    --judgment_file "$SORRY_JUDGMENT_FILE" \
    --question_file "$SORRY_Q" \
    --model_ids "${ALL_MODEL_IDS[@]}" \
    --output_file "$RESULTS_DIR/sorry_bench/sorry_bench_metrics.json"

log "=== Phase 4: BeaverTails ==="
run_beavertails_batch

log "=== Phase 5: MMLU-Redux ==="
run_mmlu_batch

log "=== RMU sweep complete ==="
log "Manifest:      $MANIFEST_FILE"
log "Sorry-Bench:   $RESULTS_DIR/sorry_bench/"
log "BeaverTails:   $RESULTS_DIR/beavertails/"
log "MMLU-Redux:    $RESULTS_DIR/mmlu_redux/"
