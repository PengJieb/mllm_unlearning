export HOME=/playpen-shared/pengjie

cd VLGuard

models="Qwen3-VL-2B-Instruct Qwen3-VL-2B-Thinking Qwen3-VL-4B-Instruct Qwen3-VL-4B-Thinking Qwen3-VL-8B-Instruct Qwen3-VL-8B-Thinking"

for model in $models; do
CUDA_VISIBLE_DEVICES=5 python VLGuard_qwen3vl_eval.py --dataset unsafes --engine $model --metaDir data/test.json --imageDir data/test
CUDA_VISIBLE_DEVICES=5 python VLGuard_qwen3vl_eval.py --dataset safe_unsafes --engine $model --metaDir data/test.json --imageDir data/test
CUDA_VISIBLE_DEVICES=5 python VLGuard_qwen3vl_eval.py --dataset safe_safes --engine $model --metaDir data/test.json --imageDir data/test
done


