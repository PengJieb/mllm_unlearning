"""Evaluate Qwen3-VL on edinburgh-dawg/mmlu-redux-2.0 (text-only, multiple choice).

Each of the 57 subject subsets is evaluated independently; overall and
per-subject accuracy are reported and saved to a JSON file.

Usage:
    CUDA_VISIBLE_DEVICES=0 python mmlu_redux_qwen3vl_eval.py \\
        --engine Qwen3-VL-8B-Instruct

    # Evaluate a subset of subjects
    CUDA_VISIBLE_DEVICES=0 python mmlu_redux_qwen3vl_eval.py \\
        --engine Qwen3-VL-8B-Instruct \\
        --subjects abstract_algebra clinical_knowledge

    # Quick smoke-test
    CUDA_VISIBLE_DEVICES=0 python mmlu_redux_qwen3vl_eval.py \\
        --engine Qwen3-VL-8B-Instruct --debug
"""

import argparse
import gc
import json
import os
import re
import sys

import torch
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── constants ─────────────────────────────────────────────────────────────────

ANSWER_CHOICES = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following multiple-choice question "
    "by stating only the letter of the correct option (A, B, C, or D) on the "
    "first line, followed by a brief explanation."
)

MODEL_MAPPINGS = {
    "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-2B-Thinking": "Qwen/Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking": "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking": "Qwen/Qwen3-VL-8B-Thinking",
}

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
            do_sample=True,
            temperature=1,
            min_new_tokens=1,
            top_p=1.0,
            top_k=40,
            repetition_penalty=1.0,
        )
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]

    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


# ── prompt & answer extraction ────────────────────────────────────────────────


def build_prompt(question: str, choices: list[str], num_shots: int, fewshot_examples: list[dict]) -> str:
    """Format a MCQ prompt, optionally prepending few-shot examples."""
    def format_example(q: str, ch: list[str], answer_idx: int | None = None) -> str:
        lines = [f"Question: {q}"]
        for i, c in enumerate(ch):
            lines.append(f"{ANSWER_CHOICES[i]}. {c}")
        if answer_idx is not None:
            lines.append(f"Answer: {ANSWER_CHOICES[answer_idx]}")
        else:
            lines.append("Answer:")
        return "\n".join(lines)

    parts = [SYSTEM_PROMPT, ""]
    for ex in fewshot_examples[:num_shots]:
        parts.append(format_example(ex["question"], ex["choices"], ex["answer"]))
        parts.append("")
    parts.append(format_example(question, choices))
    return "\n".join(parts)


def extract_answer(response: str) -> str | None:
    """Extract the predicted answer letter (A/B/C/D) from model output.

    Strategy (in priority order):
      1. Standalone letter on the first non-empty line.
      2. Pattern «answer is/: X» anywhere in the text.
      3. First occurrence of a standalone A/B/C/D in the text.
    """
    # Strip thinking traces from models that produce <think>...</think>
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # 1. First non-empty line starts with a single letter
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-D])[^a-z]*$", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Allow "A." / "A)" at the start
        m = re.match(r"^([A-D])[.):]", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        break  # only inspect first non-empty line

    # 2. "the answer is X" / "answer: X"
    m = re.search(r"answer\s*(?:is|:)\s*([A-D])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. First standalone letter
    m = re.search(r"\b([A-D])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


# ── evaluation loop ───────────────────────────────────────────────────────────


def evaluate_subject(
    subject: str,
    model,
    processor,
    max_new_tokens: int,
    enable_thinking: bool,
    num_shots: int,
    max_samples: int | None,
) -> list[dict]:
    """Evaluate a single MMLU-Redux subject and return per-sample records."""
    ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject)
    # The dataset has a single unnamed split — grab it regardless of its key.
    split_key = list(ds.keys())[0]
    data = ds[split_key]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    # Build few-shot pool from the same subject (first num_shots samples).
    fewshot_pool = list(data)[:num_shots] if num_shots > 0 else []

    records = []
    for idx, sample in enumerate(tqdm(data, desc=f"  {subject}", leave=False)):
        # Exclude the current sample from few-shot examples.
        fewshot_examples = [ex for ex in fewshot_pool if ex is not sample][:num_shots]

        prompt = build_prompt(
            sample["question"],
            sample["choices"],
            num_shots,
            fewshot_examples,
        )
        response = generate_response(model, processor, prompt, max_new_tokens, enable_thinking)
        predicted = extract_answer(response)
        gold_idx = int(sample["answer"])
        gold_letter = ANSWER_CHOICES[gold_idx]
        correct = predicted == gold_letter

        records.append(
            {
                "subject": subject,
                "question": sample["question"],
                "choices": sample["choices"],
                "gold": gold_letter,
                "predicted": predicted,
                "correct": correct,
                "response": response,
            }
        )

    return records


# ── metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(all_records: list[dict]) -> dict:
    per_subject: dict[str, dict] = {}
    for r in all_records:
        subj = r["subject"]
        per_subject.setdefault(subj, {"correct": 0, "total": 0})
        per_subject[subj]["total"] += 1
        if r["correct"]:
            per_subject[subj]["correct"] += 1

    subject_acc = {
        subj: {
            "accuracy": round(v["correct"] / v["total"] * 100, 2),
            "correct": v["correct"],
            "total": v["total"],
        }
        for subj, v in sorted(per_subject.items())
    }

    total = len(all_records)
    correct = sum(r["correct"] for r in all_records)
    overall_acc = round(correct / total * 100, 2) if total else 0.0

    return {
        "overall": {"accuracy": overall_acc, "correct": correct, "total": total},
        "per_subject": subject_acc,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL on MMLU-Redux 2.0 (multiple-choice, text-only)"
    )
    parser.add_argument(
        "--engine",
        "-e",
        default="Qwen3-VL-8B-Instruct",
        help="Short model name (mapped to HuggingFace path, or used as output ID with --model_path).",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Explicit HuggingFace model path / local dir. Overrides --engine.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="SUBJECT",
        help="Subset of subject names to evaluate. Default: all 57 subjects.",
    )
    parser.add_argument(
        "--num_shots",
        default=5,
        type=int,
        help="Number of few-shot examples prepended to each prompt (default: 5).",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=256,
        type=int,
        help="Maximum tokens to generate per response.",
    )
    parser.add_argument(
        "--max_samples",
        default=None,
        type=int,
        help="Cap on samples per subject (for quick runs).",
    )
    parser.add_argument(
        "--output_dir",
        default="results/mmlu_redux",
        type=str,
        help="Directory to write result JSON files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Evaluate 2 subjects × 10 samples each.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        args.max_samples = 10
        # pick two subjects for a quick smoke test
        all_subjects = get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0")
        args.subjects = all_subjects[:2]

    model_path = args.model_path or MODEL_MAPPINGS.get(args.engine)
    if model_path is None:
        raise ValueError(f"Unknown engine '{args.engine}'. Provide --model_path or use a known engine name.")
    engine_name = args.engine
    enable_thinking = "Thinking" in engine_name

    # Resolve subject list
    if args.subjects is None:
        subjects = get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0")
    else:
        subjects = args.subjects
    print(f"Evaluating {len(subjects)} subject(s): {subjects[:5]}{'...' if len(subjects) > 5 else ''}")

    print(f"Loading model: {model_path} …", flush=True)
    model, processor = load_model(model_path)
    print("  Model loaded.\n", flush=True)

    all_records: list[dict] = []
    for subject in subjects:
        print(f"\n[Subject] {subject}", flush=True)
        records = evaluate_subject(
            subject=subject,
            model=model,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=enable_thinking,
            num_shots=args.num_shots,
            max_samples=args.max_samples,
        )
        all_records.extend(records)
        subj_acc = sum(r["correct"] for r in records) / max(len(records), 1) * 100
        print(f"  {subject}: {subj_acc:.1f}%  ({sum(r['correct'] for r in records)}/{len(records)})")

    # ── save responses ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    responses_path = os.path.join(args.output_dir, f"{engine_name}_responses.json")
    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"\nPer-sample responses saved to: {responses_path}")

    # ── compute and save metrics ──────────────────────────────────────────────
    metrics = compute_metrics(all_records)

    print("\n=== MMLU-Redux 2.0 Results ===")
    print(f"  Overall accuracy: {metrics['overall']['accuracy']}%  "
          f"({metrics['overall']['correct']}/{metrics['overall']['total']})")
    print("\n  Per-subject accuracy:")
    for subj, m in metrics["per_subject"].items():
        print(f"    {subj:<40s} {m['accuracy']:5.1f}%  ({m['correct']}/{m['total']})")

    metrics_path = os.path.join(args.output_dir, f"{engine_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"engine": engine_name, "num_shots": args.num_shots, **metrics}, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
