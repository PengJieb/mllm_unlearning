"""Compute ASR metrics from sorry-bench model_judgment JSONL file.

Reads the ft-mistral judge output, filters by model IDs, computes overall
and per-category Attack Success Rate (ASR), and saves to JSON.
"""

import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judgment_file",
        default="text_safety_bench/sorry-bench/data/sorry_bench/model_judgment/ft-mistral-7b-instruct-v0.2.jsonl",
        help="Path to the model judgment JSONL file.",
    )
    parser.add_argument(
        "--question_file",
        default="dataset/sorry_bench/question.jsonl",
        help="Path to the sorry-bench question JSONL file (for category mapping).",
    )
    parser.add_argument(
        "--model_ids",
        nargs="+",
        required=True,
        help="Model IDs to compute metrics for.",
    )
    parser.add_argument(
        "--output_file",
        default="results/sorry_bench/sorry_bench_metrics.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    # Load question → category mapping
    qid_to_category = {}
    with open(args.question_file) as f:
        for line in f:
            q = json.loads(line.strip())
            qid_to_category[q["question_id"]] = q["category"]

    # Load all judgments, group by model
    model_judgments = defaultdict(list)
    with open(args.judgment_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            model_judgments[entry["model"]].append(entry)

    results = {}
    for model_id in args.model_ids:
        entries = model_judgments.get(model_id, [])
        if not entries:
            print(f"[WARNING] No judgments found for model: {model_id}")
            continue

        scores = [e["score"] for e in entries]
        overall_asr = sum(scores) / len(scores)

        # Per-category ASR
        cat_scores = defaultdict(list)
        for e in entries:
            cat = qid_to_category.get(e["question_id"], "unknown")
            cat_scores[cat].append(e["score"])
        category_asr = {cat: sum(v) / len(v) for cat, v in sorted(cat_scores.items())}

        results[model_id] = {
            "overall_asr": round(overall_asr, 4),
            "num_samples": len(scores),
            "category_asr": {k: round(v, 4) for k, v in category_asr.items()},
        }
        print(f"  {model_id}: ASR={overall_asr:.4f} (n={len(scores)})")

    import os
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Sorry-bench metrics saved to: {args.output_file}")


if __name__ == "__main__":
    main()
