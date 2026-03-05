#!/bin/bash

# Qwen3-VL Sorry-Bench Evaluation Script
export HOME=/playpen-shared/pengjie
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
MODEL_ID="qwen3-vl-2b"
GPU=1

QUESTION_FILE="dataset/sorry_bench/question.jsonl"
ANSWER_FILE="dataset/sorry_bench/model_answer/${MODEL_ID}.jsonl"

echo "Evaluating ${MODEL_ID} on Sorry-Bench"
echo "Model path: ${MODEL_PATH}"
echo "Output: ${ANSWER_FILE}"

CUDA_VISIBLE_DEVICES=${GPU} python text_safety_bench/sorry-bench/gen_qwen3vl_answer.py \
    --model-path ${MODEL_PATH} \
    --model-id ${MODEL_ID} \
    --question-file ${QUESTION_FILE} \
    --answer-file ${ANSWER_FILE} \
    --max-new-tokens 1024 \
    --num-gpus 1

echo "Evaluation complete. Results saved to ${ANSWER_FILE}"
