"""Generate completions with Qwen3-VL for HarmBench.

Usage:
python gen_qwen3vl_completions.py --model-path Qwen/Qwen3-VL-8B-Instruct \
    --behaviors-path data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test-cases-path data/test_cases.json \
    --save-path results/completions.json
"""
import argparse
import json
import os
import csv
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--behaviors-path", type=str, required=True)
    parser.add_argument("--test-cases-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--incremental-update", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.num_gpus > 1 else "cuda:0"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load test cases
    with open(args.test_cases_path, 'r') as f:
        test_cases_data = json.load(f)

    returned_data = {}

    # Handle incremental update
    if args.incremental_update and os.path.exists(args.save_path):
        with open(args.save_path, 'r') as f:
            returned_data = json.load(f)
        new_test_cases_data = {
            bid: cases for bid, cases in test_cases_data.items()
            if bid not in returned_data or len(returned_data[bid]) != len(cases)
        }
        test_cases_data = new_test_cases_data

    if not test_cases_data:
        print('No test cases to generate completions for')
        return

    # Generate completions
    for behavior_id, test_cases in tqdm(test_cases_data.items()):
        returned_data[behavior_id] = []
        for test_case in test_cases:
            messages = [{"role": "user", "content": [{"type": "text", "text": test_case}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False
                )

            generation = processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            returned_data[behavior_id].append({
                "test_case": test_case,
                "generation": generation
            })

    # Save results
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(returned_data, f, indent=4)
    print(f'Saved to {args.save_path}')


if __name__ == "__main__":
    main()
