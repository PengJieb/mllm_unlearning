#!/usr/bin/env bash
# run_beavertails_eval.sh
# Evaluate Qwen3-VL models on PKU-Alignment/BeaverTails-Evaluation.
#
# Usage:
#   bash running_scripts/run_beavertails_eval.sh [CUDA_DEVICE]
#
# Example — single GPU:
#   CUDA_VISIBLE_DEVICES=0 bash running_scripts/run_beavertails_eval.sh
#
# Example — specify device as positional arg:
#   bash running_scripts/run_beavertails_eval.sh 2

set -euo pipefail
export HOME=/playpen-shared/pengjie
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${REPO_ROOT}/beavertails_qwen3vl_eval.py"
OUTPUT_DIR="${REPO_ROOT}/results/beavertails"

# ── GPU selection ──────────────────────────────────────────────────────────────
GPU="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES=6
echo "[INFO] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ── Models to evaluate ─────────────────────────────────────────────────────────
# Adjust this list to run fewer/more models.
MODELS=(
    "Qwen/Qwen3-VL-2B-Instruct"
)

# ── Dataset split ──────────────────────────────────────────────────────────────
# "evaluation" is the canonical test split; "30k" is the larger training-style set.
SPLIT="${SPLIT:-test}"

# ── Max tokens per response ────────────────────────────────────────────────────
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# ── Optional sample cap (leave empty for full split) ──────────────────────────
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "${REPO_ROOT}"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================================"
    echo "  Evaluating: ${MODEL}  (split=${SPLIT})"
    echo "========================================================"

    EXTRA_ARGS=""
    if [[ -n "${MAX_SAMPLES}" ]]; then
        EXTRA_ARGS="--max_samples ${MAX_SAMPLES}"
    fi

    python "${SCRIPT}" \
        --model_path "${MODEL}" \
        --split "${SPLIT}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${OUTPUT_DIR}" \
        ${EXTRA_ARGS}

    echo "[DONE] ${MODEL}"
done

echo ""
echo "========================================================"
echo "  All evaluations complete."
echo "  Results written to: ${OUTPUT_DIR}"
echo "========================================================"
