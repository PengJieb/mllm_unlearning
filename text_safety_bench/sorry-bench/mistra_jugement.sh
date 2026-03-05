export CUDA_VISIBLE_DEVICES=1

cd text_safety_bench/sorry-bench

python gen_judgment_safety_vllm.py --model-list qwen3-vl-2b