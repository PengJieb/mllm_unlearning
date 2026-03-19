#!/usr/bin/env bash
# run_mmmu_pro_eval.sh
# Evaluate Qwen3-VL models on MMMU-Pro Standard (10-option multimodal MCQ).
#
# Usage:
#   # All models
#   bash running_scripts/run_mmmu_pro_eval.sh
#
#   # Specific GPU
#   CUDA_VISIBLE_DEVICES=1 bash running_scripts/run_mmmu_pro_eval.sh
#
#   # Quick smoke-test (debug mode: 2 subjects × 5 samples)
#   DEBUG=1 bash running_scripts/run_mmmu_pro_eval.sh
#
#   # Single model
#   MODELS="Qwen3-VL-8B-Instruct" bash running_scripts/run_mmmu_pro_eval.sh
#
#   # Custom model path (finetuned)
#   MODEL_PATH=/path/to/finetuned/model ENGINE=my-finetuned \
#       bash running_scripts/run_mmmu_pro_eval.sh
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES  — GPU(s) to use             (default: 0)
#   MODELS                — space-separated models     (default: all 6)
#   MODEL_PATH            — explicit model path        (overrides MODELS loop)
#   ENGINE                — engine name for output     (used with MODEL_PATH)
#   SUBJECTS              — space-separated subjects   (default: all)
#   MAX_NEW_TOKENS        — generation budget          (default: 512)
#   MAX_SAMPLES           — total sample cap           (default: none)
#   OUTPUT_DIR            — output directory           (default: results/mmmu_pro)
#   DEBUG                 — set to 1 for smoke test    (default: off)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Conda environment ─────────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-mllm_unlearning}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "${CONDA_ENV}"
echo "[INFO] Using conda env: ${CONDA_ENV}"
SCRIPT="${REPO_ROOT}/mmmu_pro_qwen3vl_eval.py"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/mmmu_pro}"

# ── GPU selection ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ── Hyper-parameters ───────────────────────────────────────────────────────────
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "${REPO_ROOT}"

# ── Helper: run one model ─────────────────────────────────────────────────────
run_eval() {
    local engine="$1"
    local model_path="${2:-}"

    echo ""
    echo "========================================================"
    echo "  Evaluating: ${engine}"
    echo "  max_new_tokens=${MAX_NEW_TOKENS}"
    echo "========================================================"

    EXTRA_ARGS=""

    # Debug mode
    if [[ "${DEBUG:-0}" == "1" ]]; then
        EXTRA_ARGS="${EXTRA_ARGS} --debug"
    fi

    # Optional sample cap
    if [[ -n "${MAX_SAMPLES}" ]]; then
        EXTRA_ARGS="${EXTRA_ARGS} --max_samples ${MAX_SAMPLES}"
    fi

    # Optional subject filter
    if [[ -n "${SUBJECTS:-}" ]]; then
        # shellcheck disable=SC2086
        EXTRA_ARGS="${EXTRA_ARGS} --subjects ${SUBJECTS}"
    fi

    # Optional explicit model path
    local PATH_ARG=""
    if [[ -n "${model_path}" ]]; then
        PATH_ARG="--model_path ${model_path}"
    fi

    # shellcheck disable=SC2086
    python "${SCRIPT}" \
        --engine "${engine}" \
        ${PATH_ARG} \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${OUTPUT_DIR}" \
        ${EXTRA_ARGS}

    echo "[DONE] ${engine}"
}

# ── Main ───────────────────────────────────────────────────────────────────────

if [[ -n "${MODEL_PATH:-}" ]]; then
    # Single finetuned model evaluation
    ENGINE="${ENGINE:-finetuned}"
    run_eval "${ENGINE}" "${MODEL_PATH}"
else
    # Loop over standard models
    if [[ -n "${MODELS:-}" ]]; then
        IFS=' ' read -r -a MODEL_LIST <<< "${MODELS}"
    else
        MODEL_LIST=(
            "Qwen3-VL-2B-Instruct"
            "Qwen3-VL-2B-Thinking"
            "Qwen3-VL-4B-Instruct"
            "Qwen3-VL-4B-Thinking"
            "Qwen3-VL-8B-Instruct"
            "Qwen3-VL-8B-Thinking"
        )
    fi

    for MODEL in "${MODEL_LIST[@]}"; do
        run_eval "${MODEL}"
    done
fi

echo ""
echo "========================================================"
echo "  All MMMU-Pro evaluations complete."
echo "  Results written to: ${OUTPUT_DIR}"
echo "========================================================"
