export HOME=/playpen-shared/pengjie
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
MODEL_ID="qwen3-vl-2b"
export CUDA_VISIBLE_DEVICES=1

target_model_path=VLM-Safety-Unlearn/checkpoints/qwen3vl-unlearn-lora

python VLM-Safety-Unlearn/scripts/merge_lora_weights.py --model-path $target_model_path --model-base $MODEL_PATH --save-model-path $target_model_path-merged