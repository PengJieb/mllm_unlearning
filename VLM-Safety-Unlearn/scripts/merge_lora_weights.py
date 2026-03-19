import argparse
import os, sys

sys.path.append('/playpen-shared/pengjie/mllm_unlearning_safety_alignment/VLM-Safety-Unlearn')

from transformers import AutoProcessor
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    if 'llava' in model_name.lower():
        if image_processor is not None:
            image_processor.save_pretrained(args.save_model_path)
    else:
        processor = AutoProcessor.from_pretrained(args.model_base)
        processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
