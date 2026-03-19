#!/bin/bash
export HOME=/playpen-shared/pengjie
preprocess_path=../.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203/preprocessor_config.json
MODEL_BASE="Qwen/Qwen3-VL-2B-Instruct"
export CUDA_VISIBLE_DEVICES=1

target_model_path=VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-lora-rmu
# cp $preprocess_path $target_model_path
python VLM-Safety-Unlearn/scripts/merge_lora_weights.py --model-path $target_model_path --model-base $MODEL_BASE --save-model-path $target_model_path-merged
cp $preprocess_path $target_model_path-merged
MODEL_PATH=$target_model_path-merged
MODEL_ID="qwen3-vl-2b-rmu"
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
cd text_safety_bench/sorry-bench
python gen_judgment_safety_vllm.py --model-list qwen3-vl-2b-rmu
cd -

export CUDA_VISIBLE_DEVICES=2,3,4,5

export NCCL_P2P_DISABLE=1
PROJ_ROOT="$PWD"

cd VLM-Safety-Unlearn
# export TRITON_CACHE_DIR
deepspeed --master_port 29502 qwen3vl_train/train_unlearn.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --retain_data_path $PROJ_ROOT/VLGuard/data/retain_data.json \
    --forget_data_path $PROJ_ROOT/VLGuard/data/forget_data.json \
    --image_folder $PROJ_ROOT/VLGuard/data/train/ \
    --max_pixels 1003520 \
    --min_pixels 3136 \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --bf16 True \
    --output_dir ./checkpoints/qwen3vl-unlearn-lora-grad-diff \
    --group_by_modality_length True \
    --num_train_epochs 10 \
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
    --unlearn_type "grad-diff" \
    --rmu_layer_id 15 \
    --rmu_steering_coeffs "10" \
    --rmu_llava_loss_weight 1.2 \
    --rmu_retain_alpha 0 \
    --npo_beta 0.7 \
    --npo_forget_alpha 1.0 \
    --npo_llava_loss_weight 1.0 \
    --verbose True \
    --loss_dir "./checkpoints/qwen3vl-unlearn-lora/losses"



sleep 10

deepspeed --master_port 29502 qwen3vl_train/train_unlearn.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --retain_data_path $PROJ_ROOT/VLGuard/data/retain_data.json \
    --forget_data_path $PROJ_ROOT/VLGuard/data/forget_data.json \
    --image_folder $PROJ_ROOT/VLGuard/data/train/ \
    --max_pixels 1003520 \
    --min_pixels 3136 \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --bf16 True \
    --output_dir ./checkpoints/qwen3vl-unlearn-lora-dpo \
    --group_by_modality_length True \
    --num_train_epochs 10 \
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
    --unlearn_type "dpo" \
    --rmu_layer_id 15 \
    --rmu_steering_coeffs "10" \
    --rmu_llava_loss_weight 1.2 \
    --rmu_retain_alpha 0 \
    --npo_beta 0.7 \
    --npo_forget_alpha 1.0 \
    --npo_llava_loss_weight 1.0 \
    --verbose True \
    --loss_dir "./checkpoints/qwen3vl-unlearn-lora/losses"


cd -

export CUDA_VISIBLE_DEVICES=1
target_model_path=VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-lora-grad-diff
# cp $preprocess_path $target_model_path
python VLM-Safety-Unlearn/scripts/merge_lora_weights.py --model-path $target_model_path --model-base $MODEL_BASE --save-model-path $target_model_path-merged
cp $preprocess_path $target_model_path-merged
MODEL_PATH=$target_model_path-merged
MODEL_ID="qwen3-vl-2b-grad-diff"
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

cd text_safety_bench/sorry-bench
python gen_judgment_safety_vllm.py --model-list qwen3-vl-2b-grad-diff
cd -


export CUDA_VISIBLE_DEVICES=1
target_model_path=VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-lora-dpo

python VLM-Safety-Unlearn/scripts/merge_lora_weights.py --model-path $target_model_path --model-base $MODEL_BASE --save-model-path $target_model_path-merged
cp $preprocess_path $target_model_path-merged
MODEL_PATH=$target_model_path-merged
MODEL_ID="qwen3-vl-2b-dpo"
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

cd text_safety_bench/sorry-bench
python gen_judgment_safety_vllm.py --model-list qwen3-vl-2b-dpo
cd -