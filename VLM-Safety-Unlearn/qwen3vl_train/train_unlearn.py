import os
import sys
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import random
import torch

cur_dir=str(pathlib.Path(__file__).resolve().parent.parent)
if cur_dir not in sys.path:
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import transformers
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset
from qwen3vl_train.qwen3vl_unlearn_trainer import Qwen3VLUnlearnTrainer

from PIL import Image
from qwen_vl_utils import process_vision_info

local_rank = None

IGNORE_INDEX = -100


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-VL-8B-Instruct")
    tune_mm_llm: bool = field(default=True)
    tune_mm_vision: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    retain_data_path: str = field(default=None,
                           metadata={"help": "Path to the retain data during unlearning."})
    forget_data_path: str = field(default=None,
                           metadata={"help": "Path to the forget data during unlearning."})
    image_folder: Optional[str] = field(default=None)
    max_pixels: int = field(default=1003520)
    min_pixels: int = field(default=3136)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    # Unlearning arguments
    unlearn_type: str = field(default="rmu",
                              metadata={"help": "rmu, npo, grad-diff, dpo"})
    rmu_layer_id: int = field(default=7,
                          metadata={"help": "layer to unlearn."})
    rmu_steering_coeffs: str = field(default="300",
                       metadata={"help": "Steer vector weight in order of topic"})
    rmu_retain_alpha: float = field(default=0.0,
                       metadata={"help": "retain weight"})
    rmu_llava_loss_weight: float = field(default=1.0,
                          metadata={"help": "standard loss weight for RMU."})
    npo_llava_loss_weight: float = field(default="1.0",
                       metadata={"help": "standard loss weight for NPO."})
    npo_beta: float = field(default="0.1",
                       metadata={"help": "npo beta"})
    npo_retain_alpha: float = field(default="0.0",
                    metadata={"help": "npo retain alpha"})
    npo_forget_alpha: float = field(default="1.0",
                    metadata={"help": "npo forget alpha"})
    verbose: bool = field(default=True,
                            metadata={"help": "Logging the activations norms and cosine at each step"})
    loss_dir: str = field(default="./checkpoints-unlearn/loss",
                           metadata={"help": "Directory for loss logging"})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def _build_messages(conversations, image_path):
    """Build Qwen3-VL chat messages from LLaVA-format conversations.

    Args:
        conversations: list of {"from": "human"/"gpt", "value": "..."}
        image_path: path to image file, or None if text-only

    Returns:
        list of message dicts for Qwen3-VL processor
    """
    messages = []
    for conv in conversations:
        role = "user" if conv["from"] == "human" else "assistant"
        text = conv["value"]
        # Remove <image> placeholder tokens from text
        text = text.replace("<image>", "").strip()

        content = []
        # Add image content for the first user message if image exists
        if role == "user" and image_path is not None and len(messages) == 0:
            content.append({"type": "image", "image": f"file://{image_path}"})
        content.append({"type": "text", "text": text})

        messages.append({"role": role, "content": content})
    return messages


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with Qwen3-VL."""

    def __init__(self, data_path: str, processor, data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]
        conversations = item["conversations"]

        # Build image path
        image_path = None
        if "image" in item:
            image_file = item["image"]
            image_folder = self.data_args.image_folder
            image_path = os.path.join(image_folder, image_file)

        # Build Qwen3-VL messages
        messages = _build_messages(conversations, image_path)

        # Use processor to tokenize with chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Process vision info to get image inputs
        if image_path is not None:
            image_inputs = process_vision_info(messages)[0]
        else:
            image_inputs = None

        # Tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=False,
            max_length=self.processor.tokenizer.model_max_length,
            truncation=True,
        )

        input_ids = inputs["input_ids"].squeeze(0)

        # Create labels: mask everything except assistant responses
        labels = self._create_labels(input_ids, messages)

        result = {
            "input_ids": input_ids,
            "labels": labels,
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0) if inputs["pixel_values"].dim() > 3 else inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0) if inputs["image_grid_thw"].dim() > 1 else inputs["image_grid_thw"]

        return result

    def _create_labels(self, input_ids, messages):
        """Create labels by masking non-assistant tokens with IGNORE_INDEX."""
        labels = input_ids.clone()

        # Get the tokenizer
        tokenizer = self.processor.tokenizer

        # Decode the full sequence to find assistant response boundaries
        # Strategy: tokenize each role's content to find boundaries
        # We mask everything except assistant response content

        # The chat template produces something like:
        # <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
        # We want to keep only the assistant content (after "assistant\n" and before "<|im_end|>")

        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        # Find assistant turn markers
        assistant_start_marker = "<|im_start|>assistant\n"
        assistant_end_marker = "<|im_end|>"

        # Default: mask everything
        labels[:] = IGNORE_INDEX

        # Find all assistant response spans in the text
        search_start = 0
        while True:
            start_pos = full_text.find(assistant_start_marker, search_start)
            if start_pos == -1:
                break
            content_start = start_pos + len(assistant_start_marker)
            end_pos = full_text.find(assistant_end_marker, content_start)
            if end_pos == -1:
                end_pos = len(full_text)

            # Convert character positions to token positions
            # Tokenize prefix to get token offset
            prefix_tokens = tokenizer.encode(full_text[:content_start], add_special_tokens=False)
            content_tokens = tokenizer.encode(full_text[content_start:end_pos], add_special_tokens=False)

            token_start = len(prefix_tokens)
            token_end = token_start + len(content_tokens)

            # Include the end marker token in supervised signal
            end_marker_tokens = tokenizer.encode(assistant_end_marker, add_special_tokens=False)
            token_end_with_marker = token_end + len(end_marker_tokens)

            # Unmask the assistant response tokens (including end marker)
            if token_start < len(labels):
                actual_end = min(token_end_with_marker, len(labels))
                labels[token_start:actual_end] = input_ids[token_start:actual_end]

            search_start = end_pos + len(assistant_end_marker)

        return labels


@dataclass
class DataCollatorForQwen3VL:
    """Collate examples for Qwen3-VL training."""

    processor: object

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        # Pad input_ids and labels
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        # Truncate to max length
        max_length = self.processor.tokenizer.model_max_length
        input_ids = input_ids[:, :max_length]
        labels = labels[:, :max_length]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
        }

        # Handle pixel_values: concatenate along batch dim
        if "pixel_values" in instances[0]:
            pixel_values_list = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
            if pixel_values_list:
                # pixel_values can have variable shapes; concatenate along first dim
                batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        # Handle image_grid_thw: concatenate
        if "image_grid_thw" in instances[0]:
            grid_thw_list = [inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst]
            if grid_thw_list:
                # Ensure each is 2D: (num_images, 3)
                processed = []
                for g in grid_thw_list:
                    if g.dim() == 1:
                        g = g.unsqueeze(0)
                    processed.append(g)
                batch["image_grid_thw"] = torch.cat(processed, dim=0)

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    retain_dataset = LazySupervisedDataset(
        data_path=data_args.retain_data_path,
        processor=processor,
        data_args=data_args,
    )
    forget_dataset = LazySupervisedDataset(
        data_path=data_args.forget_data_path,
        processor=processor,
        data_args=data_args,
    )

    # Sample retain data
    retain_dataset.list_data_dict = random.sample(retain_dataset.list_data_dict, len(retain_dataset.list_data_dict))
    
    # retain_dataset.list_data_dict = random.sample(retain_dataset.list_data_dict, 4)
    # forget_dataset.list_data_dict = random.sample(forget_dataset.list_data_dict, 4)
    
    data_collator = DataCollatorForQwen3VL(processor=processor)
    return dict(
        train_dataset=retain_dataset,
        forget_dataset=forget_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # Load model
    rank0_print(f"Loading model from {model_args.model_name_or_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2",
    )
    frozen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # Component tuning flags
    # Freeze/unfreeze vision encoder
    if not model_args.tune_mm_vision:
        for param in model.model.visual.parameters():
            param.requires_grad = False
    # Freeze/unfreeze MLP projector (merger in Qwen3-VL)
    if not model_args.tune_mm_mlp:
        for param in model.model.visual.merger.parameters():
            param.requires_grad = False
    # Freeze/unfreeze LLM backbone
    if not model_args.tune_mm_llm:
        for param in model.model.language_model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        max_pixels=data_args.max_pixels,
        min_pixels=data_args.min_pixels,
    )
    processor.tokenizer.model_max_length = training_args.model_max_length
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Freeze the frozen model parameters
    for param in frozen_model.parameters():
        param.requires_grad = False
    frozen_model.eval()

    # Create data module
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
    )

    training_args.rmu_steering_coeff_list = [float(c) for c in training_args.rmu_steering_coeffs.split(",")]
    trainer = Qwen3VLUnlearnTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        args=training_args,
        frozen_model=frozen_model,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        # Full parameter save
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
