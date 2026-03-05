#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3,4,5
export HOME=/playpen-shared/pengjie
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
    --output_dir ./checkpoints/qwen3vl-unlearn-lora-rmu \
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
    --unlearn_type "rmu" \
    --rmu_layer_id 15 \
    --rmu_steering_coeffs "10" \
    --rmu_llava_loss_weight 1.2 \
    --rmu_retain_alpha 0 \
    --npo_beta 0.7 \
    --npo_forget_alpha 1.0 \
    --npo_llava_loss_weight 1.0 \
    --verbose True \
    --loss_dir "./checkpoints/qwen3vl-unlearn-lora/losses"
