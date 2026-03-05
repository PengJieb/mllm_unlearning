#!/bin/bash

# Qwen2.5-VL Sorry-Bench Evaluation Script

MODEL_PATH=${1:-"Qwen/Qwen2.5-VL-7B-Instruct"}
MODEL_ID=${2:-"qwen2.5-vl-7b"}
GPU=${3:-0}

QUESTION_FILE="dataset/sorry_bench/question.jsonl"
ANSWER_FILE="dataset/sorry_bench/model_answer/${MODEL_ID}.jsonl"

echo "Evaluating ${MODEL_ID} on Sorry-Bench"
echo "Model path: ${MODEL_PATH}"
echo "Output: ${ANSWER_FILE}"

CUDA_VISIBLE_DEVICES=${GPU} python text_safety_bench/sorry-bench/gen_qwen2vl_answer.py \
    --model-path ${MODEL_PATH} \
    --model-id ${MODEL_ID} \
    --question-file ${QUESTION_FILE} \
    --answer-file ${ANSWER_FILE} \
    --max-new-tokens 1024 \
    --num-gpus 1

echo "Evaluation complete. Results saved to ${ANSWER_FILE}"
