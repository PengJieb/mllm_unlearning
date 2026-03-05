"""Generate answers with Qwen3-VL model.

Usage:
python gen_qwen3vl_answer.py --model-path Qwen/Qwen3-VL-8B-Instruct --model-id qwen3-vl-8b
"""
import argparse
import json
import os
import time
import shortuuid
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_questions(question_file, begin=None, end=None):
    questions = []
    with open(question_file, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions[begin:end]


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_tokens,
    num_gpus,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Skip already answered questions
    if os.path.exists(answer_file):
        with open(answer_file, "r") as f:
            answered_ids = {json.loads(line)["question_id"] for line in f if line.strip()}
        questions = [q for q in questions if q["question_id"] not in answered_ids]
        print(f"Remaining questions: {len(questions)}")

    # Load model using AutoModelForVision2Seq to support Qwen3-VL architecture
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if num_gpus > 1 else "cuda:0"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    for question in tqdm(questions):
        turns = []
        for turn_text in question["turns"]:
            messages = [{"role": "user", "content": [{"type": "text", "text": turn_text}]}]

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
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True
                )

            output_text = processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            turns.append(output_text)

        ans_json = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": [{"index": 0, "turns": turns}],
            "tstamp": time.time(),
        }

        with open(answer_file, "a") as f:
            f.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--question-file", type=str, default="dataset/sorry_bench/question.jsonl")
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-gpus", type=int, default=1)

    args = parser.parse_args()

    if args.answer_file is None:
        args.answer_file = f"dataset/sorry_bench/model_answer/{args.model_id}.jsonl"

    print(f"Output to {args.answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=args.question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=args.answer_file,
        max_new_tokens=args.max_new_tokens,
        num_gpus=args.num_gpus,
    )
