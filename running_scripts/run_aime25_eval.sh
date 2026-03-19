#!/usr/bin/env bash
# run_aime25_eval.sh
# Evaluate Qwen3-VL models on PrimeIntellect/AIME-25 (open-ended math, text-only).
#
# Usage:
#   # All models (30 problems each)
#   bash running_scripts/run_aime25_eval.sh
#
#   # Specific GPU
#   CUDA_VISIBLE_DEVICES=1 bash running_scripts/run_aime25_eval.sh
#
#   # Thinking models only (recommended for math) with longer generation
#   MODELS="Qwen3-VL-8B-Thinking" MAX_NEW_TOKENS=8192 bash running_scripts/run_aime25_eval.sh
#
#   # Quick smoke-test (first 5 problems)
#   DEBUG=1 bash running_scripts/run_aime25_eval.sh
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES  — GPU(s) to use          (default: 0)
#   MODELS                — space-separated models  (default: all 6)
#   MAX_NEW_TOKENS        — generation budget        (default: 4096)
#   MAX_SAMPLES           — problem count cap        (default: none = all 30)
#   OUTPUT_DIR            — output directory         (default: results/aime25)
#   DEBUG                 — set to 1 for smoke test  (default: off)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${REPO_ROOT}/aime25_qwen3vl_eval.py"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/aime25}"

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
# Thinking variants benefit from more tokens; 4096 is a reasonable default.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "${REPO_ROOT}"

for MODEL in "${MODEL_LIST[@]}"; do
    echo ""
    echo "========================================================"
    echo "  Evaluating: ${MODEL}"
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

    # shellcheck disable=SC2086
    python "${SCRIPT}" \
        --engine "${MODEL}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${OUTPUT_DIR}" \
        ${EXTRA_ARGS}

    echo "[DONE] ${MODEL}"
done

echo ""
echo "========================================================"
echo "  All AIME-25 evaluations complete."
echo "  Results written to: ${OUTPUT_DIR}"
echo "========================================================"
