"""Evaluate Qwen3-VL on MMMU-Pro Standard (10-option multimodal MCQ).

Dataset: https://huggingface.co/datasets/MMMU/MMMU_Pro  (config: "standard (10 options)")
Each question may include up to 7 images and has 10 candidate answers (A–J).

Usage:
    CUDA_VISIBLE_DEVICES=0 python mmmu_pro_qwen3vl_eval.py \
        --engine Qwen3-VL-8B-Instruct

    # Evaluate specific subjects only
    CUDA_VISIBLE_DEVICES=0 python mmmu_pro_qwen3vl_eval.py \
        --engine Qwen3-VL-8B-Instruct --subjects Art Biology

    # Quick smoke-test (2 subjects × 5 samples)
    CUDA_VISIBLE_DEVICES=0 python mmmu_pro_qwen3vl_eval.py \
        --engine Qwen3-VL-8B-Instruct --debug
"""

import argparse
import ast
import gc
import json
import os
import re
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── constants ─────────────────────────────────────────────────────────────────

ANSWER_CHOICES = list("ABCDEFGHIJ")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following multiple-choice question "
    "by stating only the letter of the correct option (A, B, C, D, E, F, G, H, I, or J) "
    "on the first line, followed by a brief explanation."
)

MODEL_MAPPINGS = {
    "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-2B-Thinking": "Qwen/Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking": "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking": "Qwen/Qwen3-VL-8B-Thinking",
}

IMAGE_FIELDS = [f"image_{i}" for i in range(1, 8)]  # image_1 … image_7

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
    content: list[dict],
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    """Run multimodal inference on Qwen3-VL.

    ``content`` is the list of dicts (text / image entries) for the user
    message, e.g. [{"type": "image", "image": <PIL>}, {"type": "text", ...}].
    """
    messages = [{"role": "user", "content": content}]
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
            temperature=0.7,
            min_new_tokens=1,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
        )
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)
        ]

    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


# ── prompt building ───────────────────────────────────────────────────────────


def parse_options(options_raw: str) -> list[str]:
    """Parse the options string into a Python list.

    The dataset stores options as a Python list literal, e.g.
    "['opt1', 'opt2', ...]".
    """
    try:
        parsed = ast.literal_eval(options_raw)
        if isinstance(parsed, list):
            return [str(o) for o in parsed]
    except (ValueError, SyntaxError):
        pass
    try:
        parsed = json.loads(options_raw)
        if isinstance(parsed, list):
            return [str(o) for o in parsed]
    except json.JSONDecodeError:
        pass
    # Fallback: comma-separated
    return [o.strip().strip("'\"") for o in options_raw.split(",")]


def build_content(question: str, options: list[str], images: dict) -> list[dict]:
    """Build the multimodal content list for a user message.

    The question text may contain ``<image N>`` placeholders.  We split on
    these markers and interleave the corresponding PIL images so the model
    receives images at the positions the question refers to.
    """
    # Format options into text
    option_lines = "\n".join(
        f"{ANSWER_CHOICES[i]}. {opt}" for i, opt in enumerate(options)
    )
    full_text = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n{option_lines}\n\nAnswer:"

    # Split the full text on <image N> placeholders
    pattern = re.compile(r"<image\s+(\d+)>", re.IGNORECASE)
    parts = pattern.split(full_text)

    content: list[dict] = []
    # parts alternates: text, image_num, text, image_num, …
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text segment
            if part:
                content.append({"type": "text", "text": part})
        else:
            # Image index (1-based)
            img_idx = int(part)
            img = images.get(img_idx)
            if img is not None:
                content.append({"type": "image", "image": img})
            else:
                # Image not available; insert a placeholder note
                content.append(
                    {"type": "text", "text": f"[image {img_idx} unavailable]"}
                )

    # If question had no <image N> placeholders but images exist, prepend them
    if not pattern.search(full_text):
        img_content: list[dict] = []
        for idx in sorted(images.keys()):
            img_content.append({"type": "image", "image": images[idx]})
        content = img_content + content

    return content


# ── answer extraction ─────────────────────────────────────────────────────────


def extract_answer(response: str) -> str | None:
    """Extract the predicted answer letter (A–J) from model output.

    Strategy (in priority order):
      1. Standalone letter on the first non-empty line.
      2. Pattern «answer is/: X» anywhere in the text.
      3. First occurrence of a standalone A–J letter in the text.
    """
    # Strip thinking traces
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    valid = set(ANSWER_CHOICES)

    # 1. First non-empty line is a single letter
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-J])[^a-z]*$", line, re.IGNORECASE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
        m = re.match(r"^([A-J])[.):]", line, re.IGNORECASE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
        break  # only first non-empty line

    # 2. "the answer is X" / "answer: X"
    m = re.search(r"answer\s*(?:is|:)\s*([A-J])\b", response, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 3. First standalone letter
    m = re.search(r"\b([A-J])\b", response)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    return None


# ── evaluation ────────────────────────────────────────────────────────────────


def collect_images(sample: dict) -> dict[int, "PIL.Image.Image"]:
    """Return {1: img1, 2: img2, …} for non-None image fields."""
    images = {}
    for i, field in enumerate(IMAGE_FIELDS, start=1):
        img = sample.get(field)
        if img is not None:
            images[i] = img
    return images


def evaluate(
    dataset,
    model,
    processor,
    max_new_tokens: int,
    enable_thinking: bool,
    subjects: list[str] | None,
    max_samples: int | None,
) -> list[dict]:
    """Run evaluation over the MMMU-Pro Standard dataset."""
    data = dataset

    # Optional subject filter
    if subjects:
        subject_set = set(subjects)
        data = data.filter(lambda x: x["subject"] in subject_set)
        print(f"  Filtered to {len(data)} samples for subjects: {subjects}")

    if max_samples and max_samples < len(data):
        data = data.select(range(max_samples))

    records = []
    for idx in tqdm(range(len(data)), desc="Evaluating"):
        sample = data[idx]
        options = parse_options(sample["options"])
        images = collect_images(sample)
        content = build_content(sample["question"], options, images)

        response = generate_response(
            model, processor, content, max_new_tokens, enable_thinking
        )
        predicted = extract_answer(response)
        gold = sample["answer"].strip().upper()
        correct = predicted == gold

        records.append(
            {
                "id": sample["id"],
                "subject": sample["subject"],
                "question": sample["question"],
                "options": options,
                "gold": gold,
                "predicted": predicted,
                "correct": correct,
                "num_images": len(images),
                "difficulty": sample.get("topic_difficulty", ""),
                "response": response,
            }
        )

        status = "✓" if correct else "✗"
        if (idx + 1) % 50 == 0 or idx == len(data) - 1:
            running_acc = sum(r["correct"] for r in records) / len(records) * 100
            print(
                f"  [{idx+1}/{len(data)}] Running accuracy: {running_acc:.1f}%",
                flush=True,
            )

    return records


# ── metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(records: list[dict]) -> dict:
    # Per-subject breakdown
    per_subject: dict[str, dict] = {}
    for r in records:
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

    # Per-difficulty breakdown
    per_diff: dict[str, dict] = {}
    for r in records:
        diff = r.get("difficulty", "Unknown") or "Unknown"
        per_diff.setdefault(diff, {"correct": 0, "total": 0})
        per_diff[diff]["total"] += 1
        if r["correct"]:
            per_diff[diff]["correct"] += 1

    difficulty_acc = {
        diff: {
            "accuracy": round(v["correct"] / v["total"] * 100, 2),
            "correct": v["correct"],
            "total": v["total"],
        }
        for diff, v in sorted(per_diff.items())
    }

    total = len(records)
    correct = sum(r["correct"] for r in records)
    overall_acc = round(correct / total * 100, 2) if total else 0.0

    return {
        "overall": {"accuracy": overall_acc, "correct": correct, "total": total},
        "per_subject": subject_acc,
        "per_difficulty": difficulty_acc,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL on MMMU-Pro Standard (10-option multimodal MCQ)"
    )
    parser.add_argument(
        "--engine",
        "-e",
        default="Qwen3-VL-8B-Instruct",
        help="Short model name (mapped to HuggingFace path) or used as output ID with --model_path.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Explicit HuggingFace model path / local dir. Overrides --engine for loading.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="SUBJECT",
        help="Evaluate only these subjects (e.g. Art Biology). Default: all.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=512,
        type=int,
        help="Maximum tokens to generate per response (default: 512).",
    )
    parser.add_argument(
        "--max_samples",
        default=None,
        type=int,
        help="Cap on total samples evaluated (for quick runs).",
    )
    parser.add_argument(
        "--output_dir",
        default="results/mmmu_pro",
        type=str,
        help="Directory to write result JSON files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Quick smoke-test: 2 subjects × 5 samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── resolve model path ────────────────────────────────────────────────────
    model_path = args.model_path or MODEL_MAPPINGS.get(args.engine)
    if model_path is None:
        raise ValueError(
            f"Unknown engine '{args.engine}'. Provide --model_path or use a known engine name."
        )
    engine_name = args.engine
    enable_thinking = "Thinking" in engine_name

    # ── load dataset ──────────────────────────────────────────────────────────
    print("Loading MMMU-Pro Standard (10 options) dataset …", flush=True)
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    print(f"  Loaded {len(ds)} samples.\n", flush=True)

    # ── debug mode ────────────────────────────────────────────────────────────
    if args.debug:
        all_subjects = sorted(set(ds["subject"]))
        args.subjects = all_subjects[:2]
        args.max_samples = 5
        print(f"[DEBUG] Evaluating subjects={args.subjects}, max_samples={args.max_samples}")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {model_path} …", flush=True)
    model, processor = load_model(model_path)
    print("  Model loaded.\n", flush=True)

    # ── run evaluation ────────────────────────────────────────────────────────
    records = evaluate(
        dataset=ds,
        model=model,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=enable_thinking,
        subjects=args.subjects,
        max_samples=args.max_samples,
    )

    # ── save responses ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    responses_path = os.path.join(args.output_dir, f"{engine_name}_responses.json")
    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nPer-sample responses saved to: {responses_path}")

    # ── compute and save metrics ──────────────────────────────────────────────
    metrics = compute_metrics(records)

    print("\n=== MMMU-Pro Standard (10 options) Results ===")
    print(
        f"  Overall accuracy: {metrics['overall']['accuracy']}%  "
        f"({metrics['overall']['correct']}/{metrics['overall']['total']})"
    )

    print("\n  Per-subject accuracy:")
    for subj, m in metrics["per_subject"].items():
        print(f"    {subj:<40s} {m['accuracy']:5.1f}%  ({m['correct']}/{m['total']})")

    print("\n  Per-difficulty accuracy:")
    for diff, m in metrics["per_difficulty"].items():
        print(f"    {diff:<12s} {m['accuracy']:5.1f}%  ({m['correct']}/{m['total']})")

    metrics_path = os.path.join(args.output_dir, f"{engine_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"engine": engine_name, **metrics}, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
