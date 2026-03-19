#!/usr/bin/env bash
# run_mmlu_redux_eval.sh
# Evaluate Qwen3-VL models on edinburgh-dawg/mmlu-redux-2.0 (text-only MCQ).
#
# Usage:
#   # All models, all 57 subjects
#   bash running_scripts/run_mmlu_redux_eval.sh
#
#   # Specific GPU
#   CUDA_VISIBLE_DEVICES=1 bash running_scripts/run_mmlu_redux_eval.sh
#
#   # Quick smoke-test (debug mode: 2 subjects × 10 samples)
#   DEBUG=1 bash running_scripts/run_mmlu_redux_eval.sh
#
#   # Evaluate a specific subset of subjects
#   SUBJECTS="abstract_algebra clinical_knowledge" bash running_scripts/run_mmlu_redux_eval.sh
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES  — GPU(s) to use          (default: 0)
#   MODELS                — space-separated models  (default: all 6)
#   SUBJECTS              — space-separated subjects (default: all 57)
#   NUM_SHOTS             — few-shot count           (default: 5)
#   MAX_NEW_TOKENS        — generation budget        (default: 256)
#   MAX_SAMPLES           — per-subject sample cap   (default: none)
#   OUTPUT_DIR            — output directory         (default: results/mmlu_redux)
#   DEBUG                 — set to 1 for smoke test  (default: off)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${REPO_ROOT}/mmlu_redux_qwen3vl_eval.py"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/mmlu_redux}"

# ── GPU selection ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ── Models ─────────────────────────────────────────────────────────────────────
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

# ── Hyper-parameters ───────────────────────────────────────────────────────────
NUM_SHOTS="${NUM_SHOTS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "${REPO_ROOT}"

for MODEL in "${MODEL_LIST[@]}"; do
    echo ""
    echo "========================================================"
    echo "  Evaluating: ${MODEL}"
    echo "  num_shots=${NUM_SHOTS}  max_new_tokens=${MAX_NEW_TOKENS}"
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

    # shellcheck disable=SC2086
    python "${SCRIPT}" \
        --engine "${MODEL}" \
        --num_shots "${NUM_SHOTS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${OUTPUT_DIR}" \
        ${EXTRA_ARGS}

    echo "[DONE] ${MODEL}"
done

echo ""
echo "========================================================"
echo "  All MMLU-Redux evaluations complete."
echo "  Results written to: ${OUTPUT_DIR}"
echo "========================================================"
