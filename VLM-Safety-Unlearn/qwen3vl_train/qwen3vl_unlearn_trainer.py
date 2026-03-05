import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from peft import PeftModel
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import itertools
import json
import datasets
from accelerate import Accelerator
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    logger,
)
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    TrainerCallback,
)
from transformers.utils import is_datasets_available
from transformers.trainer_utils import (
    EvalPrediction,
)


def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


def get_base_model(model):
    while hasattr(model, 'module'):
        model = model.module
    return model


def _get_decoder_layers(model):
    """Get decoder layers from Qwen3-VL model, handling PEFT wrapping."""
    base = model
    # Unwrap PEFT
    if isinstance(base, PeftModel):
        base = base.base_model.model
    # Unwrap DDP/DeepSpeed
    while hasattr(base, 'module'):
        base = base.module
    # Qwen3VLForConditionalGeneration -> .model -> .language_model -> .layers
    return base.model.language_model.layers


def _get_hidden_size(model):
    """Get hidden size from Qwen3-VL config."""
    config = model.config
    if hasattr(config, 'text_config'):
        return config.text_config.hidden_size
    return config.hidden_size


class Qwen3VLUnlearnTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        forget_dataset: Optional[Dataset] = None,
        frozen_model: Union[PreTrainedModel, nn.Module] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        hidden_size = _get_hidden_size(model)


        self.tokenizer = tokenizer
        # Create frozen model
        self.frozen_model = frozen_model
        self.frozen_model.to(self.model.device)
        self.frozen_model = self.frozen_model.to(dtype=self.model.dtype)

        if self.args.unlearn_type == "rmu":
            model_layers = _get_decoder_layers(model)
            frozen_layers = _get_decoder_layers(frozen_model)

            total_layers = len(model_layers)
            if not (0 <= self.args.rmu_layer_id < total_layers):
                raise ValueError(f"layer_id {self.args.rmu_layer_id} is out of range. The model has {total_layers} layers.")

            self.updated_lora_modules = model_layers[self.args.rmu_layer_id]
            self.frozen_lora_modules = frozen_layers[self.args.rmu_layer_id]

        # Forget dataset parameters
        self.forget_dataset = forget_dataset

        # Process forget_dataset & retain_dataset
        if self.forget_dataset:
            logger.info("Initialize unlearning related parameters")

            self.forget_dataloaders = [self.get_dataloader(self.forget_dataset)]
            self.forget_iterators = [iter(dl) for dl in self.forget_dataloaders]

            if self.args.unlearn_type == "rmu":
                random_vector = torch.rand(1, 1, hidden_size, dtype=self.model.dtype, device=self.model.device)
                control_vec = random_vector / torch.norm(random_vector) * args.rmu_steering_coeff_list[0]
                self.control_vector = control_vec

                if self.args.rmu_retain_alpha != 0:
                    self.retain_dataset = train_dataset
                    self.retain_dataloaders = [self.get_dataloader(self.retain_dataset)]
                    self.retain_iterators = [iter(dl) for dl in self.retain_dataloaders]

            if self.args.unlearn_type == "npo" and self.args.npo_retain_alpha != 0:
                self.retain_dataset = train_dataset
                self.retain_dataloaders = [self.get_dataloader(self.retain_dataset)]
                self.retain_iterators = [iter(dl) for dl in self.retain_dataloaders]
        else:
            self.forget_dataloaders = None
            self.retain_dataloaders = None
            self.forget_iterators = None
            self.retain_iterators = None
            self.control_vectors = None
            logger.info("Skip unlearning related parameters initialization.")

        # Frozen model is only used for inference (no training), so no need
        # for a separate Accelerator. Just keep it on the right device/dtype.
        self.frozen_acc_model = self.frozen_model

        self.loss_dir = self.args.loss_dir if hasattr(self.args, 'loss_dir') else None
        if self.loss_dir:
            os.makedirs(self.loss_dir, exist_ok=True)
            self.loss_file_path = os.path.join(self.loss_dir, 'loss.json')
            with open(self.loss_file_path, 'w') as f:
                pass
            logger.info(f"Initialized loss logging at {self.loss_file_path}")
        else:
            logger.warning("No 'loss_dir' specified in TrainingArguments. Loss will not be logged.")

    def _get_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(train_dataset)

    def get_dataloader(self, dataset) -> DataLoader:
        if dataset is None:
            raise ValueError("Trainer: training requires a dataset.")

        train_dataset = dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_sampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = lambda worker_id: torch.manual_seed(torch.initial_seed() + worker_id)

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _safe_to_device(self, inputs, device):
        """Move inputs to device, handling None values (e.g. pixel_values when no image)."""
        result = {}
        for k, v in inputs.items():
            if v is None:
                result[k] = None
            elif isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            else:
                result[k] = v
        return result

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        if self.frozen_model:
            model.train()

            # Lazy move: under DeepSpeed, self.model.device during __init__
            # may be CPU before the model is placed on GPU. Move frozen_model
            # to the correct device on the first training step.
            frozen_device = next(self.frozen_model.parameters()).device
            if frozen_device != model.device:
                self.frozen_model = self.frozen_model.to(model.device)
                self.frozen_acc_model = self.frozen_model

            if self.args.unlearn_type == "rmu":
                total_loss = torch.tensor(0.0, device=model.device)

                if self.args.rmu_retain_alpha != 0:
                    frozen_model = self.frozen_acc_model

                try:
                    forget_batch = next(self.forget_iterators[0])
                    forget_inputs = self._prepare_inputs(forget_batch)
                    forget_inputs = self._safe_to_device(forget_inputs, model.device)
                    unlearn_loss = self.compute_unlearn_loss(model, forget_inputs)

                    if self.args.rmu_retain_alpha != 0:
                        retain_batch = next(self.retain_iterators[0])
                        retain_inputs = self._prepare_inputs(retain_batch)
                        retain_inputs = self._safe_to_device(retain_inputs, model.device)
                        retain_loss = self.compute_retain_loss(model, frozen_model, retain_inputs)
                    else:
                        retain_loss = torch.tensor(0.0, device=model.device)

                    total_loss += unlearn_loss + retain_loss

                except StopIteration:
                    logger.info("forget_dataloader or retain_dataloader exhausted. Resetting iterators.")
                    self.forget_iterators = [iter(dl) for dl in self.forget_dataloaders]

                    try:
                        forget_batch = next(self.forget_iterators[0])
                        forget_inputs = self._prepare_inputs(forget_batch)
                        forget_inputs = self._safe_to_device(forget_inputs, model.device)
                        unlearn_loss = self.compute_unlearn_loss(model, forget_inputs)

                        if self.args.rmu_retain_alpha != 0:
                            retain_batch = next(self.retain_iterators[0])
                            retain_inputs = self._prepare_inputs(retain_batch)
                            retain_inputs = self._safe_to_device(retain_inputs, model.device)
                            retain_loss = self.compute_retain_loss(model, frozen_model, retain_inputs)
                        else:
                            retain_loss = torch.tensor(0.0, device=model.device)

                        total_loss += unlearn_loss + retain_loss

                    except StopIteration:
                        logger.warning("After resetting, dataloader is still exhausted. Skipping loss computation.")
                        return torch.tensor(0.0, device=model.device)

                # Calculate standard loss
                inputs = self._prepare_inputs(inputs)
                llava_loss = self.compute_loss(model, inputs)

                loss = total_loss + self.args.rmu_llava_loss_weight * llava_loss

                # Track per-forward-pass components for separate backward calls
                # (avoids DeepSpeed ZeRO-2 "gradient computed twice" with LoRA)
                backward_components = [unlearn_loss]
                if self.args.rmu_retain_alpha != 0:
                    backward_components.append(retain_loss)
                if self.args.rmu_llava_loss_weight != 0:
                    backward_components.append(self.args.rmu_llava_loss_weight * llava_loss)

                if self.loss_dir:
                    loss_entry = {
                        "step": self.state.global_step,
                        "loss": loss.item(),
                        "llava_loss": llava_loss.item(),
                        "unlearn_loss": unlearn_loss.item(),
                        "retain_loss": retain_loss.item(),
                        "learning_rate": self._get_learning_rate(),
                        "epoch": self.state.epoch
                    }
                    try:
                        with open(self.loss_file_path, 'a') as f:
                            f.write(json.dumps(loss_entry) + '\n\n')
                    except Exception as e:
                        logger.error(f"Failed to write loss entry to {self.loss_file_path}: {e}")

                if self.args.verbose:
                    print(f"Step {self.state.global_step}: Total Loss={loss}, Standard Loss={llava_loss}, Retain Loss={retain_loss}, Unlearn Loss={unlearn_loss}")

            elif self.args.unlearn_type == "npo":
                npo_loss = torch.tensor(0.0, device=model.device)
                frozen_model = self.frozen_acc_model

                try:
                    forget_batch = next(self.forget_iterators[0])
                    forget_inputs = self._prepare_inputs(forget_batch)
                    forget_inputs = self._safe_to_device(forget_inputs, model.device)

                    prepared_forget_inputs = self._prepare_inputs(forget_inputs)

                    forget_loss_current = self.compute_loss(model, prepared_forget_inputs)

                    with torch.no_grad():
                        forget_loss_oracle = self.compute_loss(frozen_model, prepared_forget_inputs)

                    neg_log_ratios = forget_loss_current - forget_loss_oracle

                    forget_npo_loss = self.args.npo_forget_alpha * (-F.logsigmoid(self.args.npo_beta * neg_log_ratios).mean() * 2 / self.args.npo_beta)
                    retain_npo_loss = torch.tensor(0.0, device=model.device)

                    if self.args.npo_retain_alpha != 0:
                        retain_batch = next(self.retain_iterators[0])
                        retain_inputs = self._prepare_inputs(retain_batch)
                        retain_inputs = self._safe_to_device(retain_inputs, model.device)

                        prepared_retain_inputs = self._prepare_inputs(retain_inputs)

                        retain_loss_current = self.compute_loss(model, prepared_retain_inputs)

                        with torch.no_grad():
                            retain_loss_oracle = self.compute_loss(frozen_model, prepared_retain_inputs)

                        retain_neg_log_ratios = retain_loss_current - retain_loss_oracle

                        retain_npo_loss = self.args.npo_retain_alpha * (F.logsigmoid(self.args.npo_beta * retain_neg_log_ratios).mean() * 2 / self.args.npo_beta)

                except StopIteration:
                    logger.info("forget_dataloader or retain_dataloader exhausted. Resetting iterators.")
                    self.forget_iterators = [iter(dl) for dl in self.forget_dataloaders]

                    if self.args.npo_retain_alpha != 0:
                        self.retain_iterators = [iter(dl) for dl in self.retain_dataloaders]

                    try:
                        forget_batch = next(self.forget_iterators[0])
                        forget_inputs = self._prepare_inputs(forget_batch)
                        forget_inputs = self._safe_to_device(forget_inputs, model.device)

                        prepared_forget_inputs = self._prepare_inputs(forget_inputs)

                        forget_loss_current = self.compute_loss(model, prepared_forget_inputs)

                        with torch.no_grad():
                            forget_loss_oracle = self.compute_loss(frozen_model, prepared_forget_inputs)

                        neg_log_ratios = forget_loss_current - forget_loss_oracle

                        forget_npo_loss = self.args.npo_forget_alpha * (-F.logsigmoid(self.args.npo_beta * neg_log_ratios).mean() * 2 / self.args.npo_beta)
                        retain_npo_loss = torch.tensor(0.0, device=model.device)

                        if self.args.npo_retain_alpha != 0:
                            retain_batch = next(self.retain_iterators[0])
                            retain_inputs = self._prepare_inputs(retain_batch)
                            retain_inputs = self._safe_to_device(retain_inputs, model.device)

                            prepared_retain_inputs = self._prepare_inputs(retain_inputs)

                            retain_loss_current = self.compute_loss(model, prepared_retain_inputs)

                            with torch.no_grad():
                                retain_loss_oracle = self.compute_loss(frozen_model, prepared_retain_inputs)

                            retain_neg_log_ratios = retain_loss_current - retain_loss_oracle

                            retain_npo_loss = self.args.npo_retain_alpha * (F.logsigmoid(self.args.npo_beta * retain_neg_log_ratios).mean() * 2 / self.args.npo_beta)

                    except StopIteration:
                        logger.warning("After resetting, dataloader is still exhausted. Skipping loss computation.")
                        return torch.tensor(0.0, device=model.device)

                # Calculate standard loss
                inputs = self._prepare_inputs(inputs)
                llava_loss = self.compute_loss(model, inputs)
                npo_loss = forget_npo_loss + retain_npo_loss

                loss = npo_loss + self.args.npo_llava_loss_weight * llava_loss

                # Track per-forward-pass components for separate backward calls
                # (avoids DeepSpeed ZeRO-2 "gradient computed twice" with LoRA)
                backward_components = [forget_npo_loss]
                if self.args.npo_retain_alpha != 0:
                    backward_components.append(retain_npo_loss)
                if self.args.npo_llava_loss_weight != 0:
                    backward_components.append(self.args.npo_llava_loss_weight * llava_loss)

                if self.loss_dir:
                    loss_entry = {
                        "step": self.state.global_step,
                        "loss": loss.item(),
                        "llava_loss": llava_loss.item(),
                        "forget_po_loss": forget_npo_loss.item(),
                        "retain_po_loss": retain_npo_loss.item(),
                        "learning_rate": self._get_learning_rate(),
                        "epoch": self.state.epoch
                    }
                    try:
                        with open(self.loss_file_path, 'a') as f:
                            f.write(json.dumps(loss_entry) + '\n\n')
                    except Exception as e:
                        logger.error(f"Failed to write loss entry to {self.loss_file_path}: {e}")

                if self.args.verbose:
                    print(f"Step {self.state.global_step}: Total Loss={loss}, Standard Loss={llava_loss}, Unlearn PO Loss={forget_npo_loss}, Retain PO Loss={retain_npo_loss}")

            elif self.args.unlearn_type == "grad-diff":
                backward_components = []

            # Call backward separately for each forward-pass component.
            # With LoRA + DeepSpeed ZeRO-2 (overlap_comm=True), a single
            # backward() through a combined loss that touches the same LoRA
            # parameters in multiple forward-pass sub-graphs causes
            # "gradient computed twice" assertion errors. Separate backward()
            # calls reset params_already_reduced between each call.
            output_loss = loss.detach() / self.args.gradient_accumulation_steps
            for component in backward_components:
                if self.args.n_gpu > 1:
                    component = component.mean()
                self.accelerator.backward(component)
            return output_loss
        else:
            return super().training_step(model, inputs)

    def compute_unlearn_loss(self, model: nn.Module, forget_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        updated_forget_activations = self.forward_with_cache(model, forget_batch, module=self.updated_lora_modules, no_grad=False).to(self.model.device)

        batch_size, seq_len, _ = updated_forget_activations.shape
        expanded_control_vector = self.control_vector.to(updated_forget_activations.device).broadcast_to(batch_size, seq_len, -1)

        unlearn_loss = torch.nn.functional.mse_loss(updated_forget_activations, expanded_control_vector)
        return unlearn_loss

    def compute_retain_loss(self, model: nn.Module, frozen_model: nn.Module, retain_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        updated_retain_activations = self.forward_with_cache(model, retain_batch, module=self.updated_lora_modules, no_grad=False).to(self.model.device)
        updated_forget_activations = self.forward_with_cache(frozen_model, retain_batch, module=self.frozen_lora_modules, no_grad=True).to(self.model.device)

        batch_size_retain, seq_len_retain, feature_dim_retain = updated_retain_activations.shape
        batch_size_forget, seq_len_forget, feature_dim_forget = updated_forget_activations.shape

        assert batch_size_retain == batch_size_forget, "batch size is not consistent"

        min_seq_len = min(seq_len_retain, seq_len_forget)
        updated_retain_activations = updated_retain_activations[:, :min_seq_len, :]
        updated_forget_activations = updated_forget_activations[:, :min_seq_len, :]

        retain_loss = torch.nn.functional.mse_loss(updated_retain_activations, updated_forget_activations)

        alpha = float(self.args.rmu_retain_alpha)
        return alpha * retain_loss

    def forward_with_cache(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        module: nn.Module,
        no_grad: bool
    ) -> torch.Tensor:
        activations = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                activations.append(output[0])
            else:
                activations.append(output)
            return None

        handle = module.register_forward_hook(hook)

        prepared_inputs = self._prepare_inputs(inputs)

        base_model = get_base_model(model)

        if no_grad:
            with torch.no_grad():
                base_model(**prepared_inputs)
        else:
            base_model(**prepared_inputs)

        handle.remove()

        if len(activations) == 0:
            print(f"No activations captured for module: {module}")
            raise ValueError("No activations were captured. Please check if the hook is correctly registered.")

        return activations[0]

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, **kwargs):
        super(Qwen3VLUnlearnTrainer, self)._save_checkpoint(model, trial, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(Qwen3VLUnlearnTrainer, self)._save(output_dir, state_dict)
