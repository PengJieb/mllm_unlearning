"""Evaluate Qwen3-VL on PrimeIntellect/AIME-25 (text-only, open-ended math).

The dataset contains 30 competition math problems from AIME 2025 I & II.
Prompts already instruct the model to mark its final answer with \\boxed{}.
We extract the \\boxed{} answer and compare it to the ground truth (exact match).

Usage:
    CUDA_VISIBLE_DEVICES=0 python aime25_qwen3vl_eval.py \\
        --engine Qwen3-VL-8B-Thinking   # Thinking variant recommended for math

    # Use more tokens for harder problems
    CUDA_VISIBLE_DEVICES=0 python aime25_qwen3vl_eval.py \\
        --engine Qwen3-VL-8B-Thinking --max_new_tokens 8192

    # Quick smoke-test (first 5 problems)
    CUDA_VISIBLE_DEVICES=0 python aime25_qwen3vl_eval.py \\
        --engine Qwen3-VL-8B-Instruct --debug
"""

import argparse
import gc
import json
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── constants ─────────────────────────────────────────────────────────────────

MODEL_MAPPINGS = {
    "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-2B-Thinking": "Qwen/Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking": "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking": "Qwen/Qwen3-VL-8B-Thinking",
}

# Default max tokens — AIME problems require long chain-of-thought.
# Thinking variants may need 4096–16384; Instruct variants use fewer.
DEFAULT_MAX_NEW_TOKENS = 4096

# ── model helpers ─────────────────────────────────────────────────────────────


def load_model(model_path: str):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def generate_response(
    model,
    processor,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    """Run text-only inference on Qwen3-VL."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1,
            min_new_tokens=1,
        )
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]

    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


# ── answer extraction ─────────────────────────────────────────────────────────


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract the innermost \boxed{...} content from model output.

    Handles nested braces, e.g. ``\boxed{\frac{1}{2}}``.
    Returns the last boxed expression found (models often re-state the answer
    at the end) or None if no boxed expression is present.
    """
    # Strip thinking traces first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    pattern = r"\\boxed\{"
    results = []
    for m in re.finditer(pattern, text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1].strip())

    return results[-1] if results else None


def normalize_answer(answer: str | None) -> str | None:
    """Normalize an answer string for comparison.

    Strips whitespace, leading zeros, and common LaTeX formatting so that
    e.g. ``070`` and ``70`` compare as equal.
    """
    if answer is None:
        return None
    # Remove LaTeX commands that don't affect numeric value
    answer = re.sub(r"\\(text|mathrm|mathbf)\{([^}]+)\}", r"\2", answer)
    answer = answer.strip().lstrip("0") or "0"
    return answer


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    """Return True if predicted answer matches ground truth (after normalisation)."""
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(ground_truth)
    if pred_norm is None:
        return False
    return pred_norm == gold_norm


# ── metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(records: list[dict]) -> dict:
    total = len(records)
    correct = sum(r["correct"] for r in records)
    return {
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "correct": correct,
        "total": total,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL on AIME-25 (open-ended math, text-only)"
    )
    parser.add_argument(
        "--engine",
        "-e",
        default="Qwen3-VL-8B-Thinking",
        choices=list(MODEL_MAPPINGS.keys()),
        help="Short model name (mapped to HuggingFace path). "
             "Thinking variants are recommended for math reasoning.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Explicit HuggingFace model path / local dir. Overrides --engine.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=DEFAULT_MAX_NEW_TOKENS,
        type=int,
        help=f"Maximum tokens to generate per response (default: {DEFAULT_MAX_NEW_TOKENS}). "
             "Increase to 8192–16384 for Thinking variants.",
    )
    parser.add_argument(
        "--max_samples",
        default=None,
        type=int,
        help="Cap on the number of problems to evaluate (for quick runs). "
             "Default: all 30 problems.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/aime25",
        type=str,
        help="Directory to write result JSON files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Evaluate first 5 problems only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.max_samples = 5

    model_path = args.model_path or MODEL_MAPPINGS[args.engine]
    engine_name = args.engine
    enable_thinking = "Thinking" in engine_name

    print(f"Loading dataset PrimeIntellect/AIME-25 …", flush=True)
    ds = load_dataset("PrimeIntellect/AIME-25")
    split_key = list(ds.keys())[0]
    data = ds[split_key]
    if args.max_samples:
        data = data.select(range(min(args.max_samples, len(data))))
    print(f"  {len(data)} problems loaded.", flush=True)

    print(f"Loading model: {model_path} …", flush=True)
    model, processor = load_model(model_path)
    print("  Model loaded.\n", flush=True)

    records = []
    with torch.no_grad():
        for sample in tqdm(data, desc=f"Evaluating {engine_name}"):
            problem_id = sample["problem_id"]
            prompt = sample["prompt"]  # already includes "Mark your solution with \boxed."

            # Parse ground truth from verification_info JSON
            try:
                verification = json.loads(sample["verification_info"])
                ground_truth = str(verification["ground_truth"])
            except (json.JSONDecodeError, KeyError):
                ground_truth = str(sample.get("verification_info", ""))

            response = generate_response(
                model, processor, prompt, args.max_new_tokens, enable_thinking
            )
            predicted = extract_boxed_answer(response)
            correct = answers_match(predicted, ground_truth)

            records.append(
                {
                    "problem_id": problem_id,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "correct": correct,
                    "response": response,
                }
            )

            status = "✓" if correct else "✗"
            print(
                f"  [{status}] {problem_id}: gold={ground_truth!r}  pred={predicted!r}",
                flush=True,
            )

    # ── save responses ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    responses_path = os.path.join(args.output_dir, f"{engine_name}_responses.json")
    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nPer-problem responses saved to: {responses_path}")

    # ── compute and save metrics ──────────────────────────────────────────────
    metrics = compute_metrics(records)

    print("\n=== AIME-25 Results ===")
    print(f"  Accuracy : {metrics['accuracy']}%  ({metrics['correct']}/{metrics['total']})")
    print("\n  Per-problem breakdown:")
    for r in records:
        status = "✓" if r["correct"] else "✗"
        print(f"    [{status}] {r['problem_id']:<15s}  gold={r['ground_truth']!r:<8s}  pred={r['predicted']!r}")

    metrics_path = os.path.join(args.output_dir, f"{engine_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"engine": engine_name, **metrics, "per_problem": records}, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
