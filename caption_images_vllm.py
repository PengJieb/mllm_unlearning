"""
Extract image captions using vLLM with a HuggingFace Qwen-VL model directory.

Output: JSON file mapping image filename -> caption string.

Usage:
    python caption_images_vllm.py \
        --model-path dataset/Qwen3.5-9B \
        --image-dir /path/to/images \
        --output-file captions.json
"""

import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

DEFAULT_PROMPT = '''
Describe this image in detail.
'''


def encode_image_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_messages(image: Image.Image, prompt: str) -> list:
    b64 = encode_image_base64(image)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def collect_images(image_dir: str) -> list[Path]:
    base = Path(image_dir)
    paths = sorted(
        p for p in base.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Batch image captioning with vLLM + HuggingFace Qwen-VL")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the HuggingFace model directory (e.g. dataset/Qwen3.5-9B).",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing images to caption.",
    )
    parser.add_argument(
        "--output-file",
        default="captions.json",
        help="Output JSON file path (default: captions.json).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Caption instruction prompt (default: '{DEFAULT_PROMPT}').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images per vLLM batch (default: 8).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per caption (default: 512).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="vLLM max_model_len (default: 8192).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction (default: 0.9).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1).",
    )
    return parser.parse_args()


def load_llm(args) -> LLM:
    model_path = args.model_path

    kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        # Qwen3.5 hybrid linear-attention is incompatible with CUDA graph capture
        enforce_eager=True,
        # Allow multi-modal inputs
        limit_mm_per_prompt={"image": 1},
    )

    print(f"Loading model: {model_path}")

    return LLM(**kwargs)


def main():
    args = parse_args()

    image_paths = collect_images(args.image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.image_dir}")
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    llm = load_llm(args)
    # Recommended non-thinking (instruct) mode params per Qwen3.5 README
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
    )

    captions: dict[str, str] = {}

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Process in batches
    for batch_start in tqdm(
        range(0, len(image_paths), args.batch_size),
        desc="Captioning",
        unit="batch",
    ):
        batch_paths = image_paths[batch_start : batch_start + args.batch_size]
        conversation_batch = []
        valid_paths = []

        for img_path in batch_paths:
            try:
                image = Image.open(img_path)
                conversation_batch.append(build_messages(image, args.prompt))
                valid_paths.append(img_path)
            except Exception as e:
                print(f"  Warning: could not load {img_path.name}: {e}")

        if not conversation_batch:
            continue

        outputs = llm.chat(
            conversation_batch,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )

        for img_path, output in zip(valid_paths, outputs):
            caption = output.outputs[0].text.strip()
            captions[img_path.name] = caption

    
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(captions)} captions to: {output_path}")


if __name__ == "__main__":
    main()
