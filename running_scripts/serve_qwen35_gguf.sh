#!/usr/bin/env bash
# serve_qwen35_gguf.sh
# Serve Qwen3.5-9B-Aggressive GGUF via vLLM OpenAI-compatible API (/v1/chat/completions)
# with API-key authentication.
#
# Usage:
#   bash running_scripts/serve_qwen35_gguf.sh
#
# Override defaults via env vars:
#   CUDA_VISIBLE_DEVICES=0,1  API_KEY=mysecret  PORT=8000  bash running_scripts/serve_qwen35_gguf.sh

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${REPO_ROOT}/dataset/Qwen3.5-9B-Aggresive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
MMPROJ_PATH="${REPO_ROOT}/dataset/Qwen3.5-9B-Aggresive/mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"

# HuggingFace tokenizer for the base model architecture (required for GGUF loading)
TOKENIZER="Qwen/Qwen3.5-9B"

# ── Server settings ───────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="Qwen3.5-9B-Aggressive"

# GPU memory fraction to allocate (0.0 – 1.0)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# Cap context length to avoid OOM (native max: 262144)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# API key — auto-generate a random one if not set via env var
if [[ -z "${API_KEY:-}" ]]; then
    API_KEY="$(openssl rand -hex 32)"
    echo "[INFO] No API_KEY set — generated a random key for this session."
fi

# ── Validate files ────────────────────────────────────────────────────────────
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "[ERROR] Model GGUF not found: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${MMPROJ_PATH}" ]]; then
    echo "[ERROR] Mmproj GGUF not found: ${MMPROJ_PATH}" >&2
    exit 1
fi

# ── Print startup info ────────────────────────────────────────────────────────
echo "========================================================"
echo "  Model:     ${MODEL_PATH}"
echo "  Mmproj:    ${MMPROJ_PATH}"
echo "  Tokenizer: ${TOKENIZER}"
echo "  Endpoint:  http://${HOST}:${PORT}/v1/chat/completions"
echo "  API Key:   ${API_KEY}"
echo "========================================================"
echo ""
echo "  Test with:"
echo "    curl http://localhost:${PORT}/v1/models \\"
echo "         -H \"Authorization: Bearer ${API_KEY}\""
echo ""

# ── Launch vLLM server ────────────────────────────────────────────────────────
vllm serve "${MODEL_PATH}" \
    --tokenizer "${TOKENIZER}" \
    --hf-overrides "{\"vision_model_path\": \"${MMPROJ_PATH}\"}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --dtype bfloat16 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code
