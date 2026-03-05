#!/bin/bash

# Qwen3-VL HarmBench Evaluation Script
export HOME=/playpen-shared/pengjie
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
MODEL_ID="qwen3-vl-2b"
GPU=0
cd text_safety_bench/HarmBench
BEHAVIORS_PATH="data/behavior_datasets/harmbench_behaviors_text_all.csv"
TEST_CASES_PATH="data/test_cases/direct_request.json"
COMPLETIONS_PATH="data/completions/${MODEL_ID}_direct_request.json"

echo "Generating completions for ${MODEL_ID} on HarmBench"
echo "Model path: ${MODEL_PATH}"
echo "Test cases: ${TEST_CASES_PATH}"
echo "Output: ${COMPLETIONS_PATH}"

CUDA_VISIBLE_DEVICES=${GPU} python gen_qwen3vl_completions.py \
    --model-path ${MODEL_PATH} \
    --behaviors-path ${BEHAVIORS_PATH} \
    --test-cases-path ${TEST_CASES_PATH} \
    --save-path ${COMPLETIONS_PATH} \
    --max-new-tokens 512 \
    --num-gpus 1

echo "Generation complete. Results saved to ${COMPLETIONS_PATH}"
