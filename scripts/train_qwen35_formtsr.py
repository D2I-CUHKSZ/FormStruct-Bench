#!/usr/bin/env python3
"""LoRA-SFT a Qwen multimodal model on an index-built FormTSR conversation file."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA module suffixes.",
    )
    parser.add_argument(
        "--target-modules-regex",
        default="",
        help="Optional full module-name regex. Overrides --target-modules.",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument(
        "--fsdp",
        default="",
        help='Optional Transformers FSDP mode, for example "full_shard auto_wrap".',
    )
    parser.add_argument(
        "--fsdp-transformer-layer-cls-to-wrap",
        default="",
        help="Comma-separated layer class names for FSDP auto wrapping.",
    )
    parser.add_argument(
        "--fsdp-backward-prefetch",
        choices=("no_prefetch", "backward_pre", "backward_post"),
        default="no_prefetch",
    )
    parser.add_argument("--fsdp-limit-all-gathers", action="store_true")
    parser.add_argument(
        "--fsdp-activation-checkpointing",
        action="store_true",
        help="Use FSDP's outer checkpoint wrapper instead of model-native checkpointing.",
    )
    parser.add_argument(
        "--fsdp-adapter-only-save",
        action="store_true",
        help="Gather and save only LoRA tensors, avoiding a full FSDP state dict.",
    )
    parser.add_argument(
        "--fsdp-adapter-save-steps",
        type=int,
        default=0,
        help="Optional adapter-only checkpoint interval in optimizer steps.",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


class JsonlDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path, limit: int | None = None) -> None:
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if limit is not None:
            self.rows = self.rows[: max(0, limit)]
        if not self.rows:
            raise ValueError(f"empty dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _chat_text(processor: Any, messages: list[dict[str, Any]], *, generation: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": generation,
        "enable_thinking": False,
    }
    try:
        return str(processor.apply_chat_template(messages, **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        return str(processor.apply_chat_template(messages, **kwargs))


class FormTSRCollator:
    def __init__(
        self,
        processor: Any,
        *,
        max_seq_length: int,
        min_pixels: int,
        max_pixels: int,
    ) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = "right"
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        self.image_token = getattr(processor, "image_token", "<|image_pad|>")
        self.merge_length = int(getattr(processor.image_processor, "merge_size", 2)) ** 2

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_texts: list[str] = []
        prefix_texts: list[str] = []
        images: list[Image.Image] = []
        ids: list[str] = []
        for feature in features:
            messages = feature["messages"]
            full_texts.append(_chat_text(self.processor, messages, generation=False))
            prefix_texts.append(_chat_text(self.processor, messages[:1], generation=True))
            image_path = feature.get("image") or feature.get("image_path")
            if not image_path:
                raise ValueError(f"missing image path for {feature.get('id')}")
            with Image.open(str(image_path)) as image:
                images.append(image.convert("RGB"))
            ids.append(str(feature.get("id", "")))

        encoded = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_mm_token_type_ids=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        for index, (prefix, sample_id) in enumerate(zip(prefix_texts, ids)):
            grid = encoded["image_grid_thw"][index]
            image_tokens = int(grid.prod().item() // self.merge_length)
            expanded_prefix = prefix.replace(self.image_token, self.image_token * image_tokens, 1)
            prefix_ids = self.tokenizer(expanded_prefix, add_special_tokens=False)["input_ids"]
            prefix_length = len(prefix_ids)
            sample_length = int(attention_mask[index].sum().item())
            if prefix_length >= sample_length:
                raise ValueError(
                    f"assistant target is empty for {sample_id}: prefix={prefix_length}, length={sample_length}"
                )
            if sample_length > self.max_seq_length:
                raise ValueError(
                    f"sample {sample_id} has {sample_length} tokens, exceeding --max-seq-length {self.max_seq_length}; "
                    "increase the limit rather than truncating JSON targets"
                )
            labels[index, :prefix_length] = -100
            labels[index, sample_length:] = -100

        # Keep only model inputs.  BatchFeature may contain image metadata that
        # Trainer would otherwise try to move as an unsupported object.
        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
        }
        if "mm_token_type_ids" in encoded:
            batch["mm_token_type_ids"] = encoded["mm_token_type_ids"]
        return batch


def build_training_args(args: argparse.Namespace, output_dir: Path, *, has_eval: bool) -> Any:
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    fsdp_enabled = bool(args.fsdp.strip())
    model_gradient_checkpointing = (
        not args.no_gradient_checkpointing and not args.fsdp_activation_checkpointing
    )
    values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 2,
        "bf16": not args.no_bf16,
        "tf32": True,
        "gradient_checkpointing": model_gradient_checkpointing,
        "use_cache": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "ddp_find_unused_parameters": False,
        "optim": "adamw_torch_fused",
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": [],
        "logging_strategy": "steps",
        "save_strategy": "no" if args.fsdp_adapter_only_save else "steps",
        "save_safetensors": True,
    }
    if model_gradient_checkpointing:
        values["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if fsdp_enabled:
        layer_classes = [
            item.strip()
            for item in args.fsdp_transformer_layer_cls_to_wrap.split(",")
            if item.strip()
        ]
        if "auto_wrap" in args.fsdp.split() and not layer_classes:
            raise ValueError(
                "--fsdp auto_wrap requires --fsdp-transformer-layer-cls-to-wrap"
            )
        fsdp_config: dict[str, Any] = {
            "use_orig_params": True,
            "sync_module_states": True,
            "cpu_ram_efficient_loading": False,
            "activation_checkpointing": args.fsdp_activation_checkpointing,
            "limit_all_gathers": args.fsdp_limit_all_gathers,
            "forward_prefetch": False,
            "backward_prefetch": args.fsdp_backward_prefetch,
        }
        if layer_classes:
            fsdp_config["transformer_layer_cls_to_wrap"] = layer_classes
        values["fsdp"] = args.fsdp
        values["fsdp_config"] = fsdp_config
    if "eval_strategy" in params:
        values["eval_strategy"] = "steps" if has_eval else "no"
    elif "evaluation_strategy" in params:
        values["evaluation_strategy"] = "steps" if has_eval else "no"
    if "use_liger_kernel" in params:
        values["use_liger_kernel"] = False
    return TrainingArguments(**{key: value for key, value in values.items() if key in params})


def _normalize_fsdp_name(name: str) -> str:
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


def _lora_weight_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    """Recover logical LoRA matrix shapes hidden by FSDP parameter views."""
    shapes: dict[str, tuple[int, ...]] = {}
    for module_name, module in model.named_modules():
        normalized_name = _normalize_fsdp_name(module_name)
        for container_name in ("lora_A", "lora_B"):
            container = getattr(module, container_name, None)
            if container is None or not hasattr(container, "items"):
                continue
            for adapter_name, projection in container.items():
                weight = getattr(projection, "weight", None)
                if weight is None:
                    continue
                key = f"{normalized_name}.{container_name}.{adapter_name}.weight"
                shapes[key] = (int(projection.out_features), int(projection.in_features))
    return shapes


def save_fsdp_adapter_only(trainer: Any, output_dir: Path) -> dict[str, Any]:
    """Materialize one FSDP unit at a time and persist only its LoRA tensors."""
    import gc

    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    wrapped_model = trainer.model_wrapped
    if not isinstance(wrapped_model, FSDP):
        raise TypeError("--fsdp-adapter-only-save requires an FSDP-wrapped model")

    rank = dist.get_rank() if dist.is_initialized() else 0
    raw_adapter_state: dict[str, torch.Tensor] = {}
    logical_shapes = _lora_weight_shapes(wrapped_model)
    gathered_units = 0
    fsdp_units = [
        (name, module)
        for name, module in wrapped_model.named_modules()
        if isinstance(module, FSDP) and not module.check_is_root()
    ]
    for unit_name, fsdp_unit in fsdp_units:
        if not any("lora_" in name for name, _ in fsdp_unit.named_parameters(recurse=True)):
            continue
        gathered_units += 1
        full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            fsdp_unit,
            StateDictType.FULL_STATE_DICT,
            full_state_config,
        ):
            unit_state = fsdp_unit.state_dict()
        if rank == 0:
            prefix = _normalize_fsdp_name(unit_name)
            for local_name, tensor in unit_state.items():
                if "lora_" not in local_name:
                    continue
                normalized_local = _normalize_fsdp_name(local_name)
                if prefix and not normalized_local.startswith(f"{prefix}."):
                    key = f"{prefix}.{normalized_local}"
                else:
                    key = normalized_local
                if key in raw_adapter_state:
                    raise RuntimeError(f"duplicate gathered LoRA key: {key}")
                logical_shape = logical_shapes.get(key)
                if logical_shape is None:
                    raise RuntimeError(f"missing logical LoRA shape for gathered key: {key}")
                if tuple(tensor.shape) != logical_shape:
                    raise RuntimeError(
                        f"LoRA tensor shape mismatch for {key}: "
                        f"gathered={tuple(tensor.shape)}, expected={logical_shape}"
                    )
                raw_adapter_state[key] = tensor.detach().cpu().clone()
        del unit_state
        gc.collect()

    if rank == 0:
        expected = sum(1 for name, _ in trainer.model.named_parameters() if "lora_" in name)
        if len(raw_adapter_state) != expected:
            sample = ", ".join(sorted(raw_adapter_state)[:5])
            raise RuntimeError(
                f"gathered {len(raw_adapter_state)} LoRA tensors, expected {expected}; sample: {sample}"
            )
        trainer.model.save_pretrained(
            str(output_dir),
            state_dict=raw_adapter_state,
            safe_serialization=True,
            is_main_process=True,
            save_embedding_layers=False,
        )
    if dist.is_initialized():
        dist.barrier()
    return {
        "fsdp_units_with_lora": gathered_units,
        "adapter_tensor_count": len(raw_adapter_state) if rank == 0 else None,
        "save_mode": "per-unit FSDP full state dict; LoRA tensors only",
    }


def add_fsdp_adapter_checkpoint_callback(
    trainer: Any,
    output_dir: Path,
    processor: Any,
    save_steps: int,
) -> None:
    import torch.distributed as dist
    from transformers import TrainerCallback

    if save_steps < 1:
        raise ValueError("--fsdp-adapter-save-steps must be positive")

    class AdapterOnlyCheckpointCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.global_step <= 0 or state.global_step % save_steps:
                return control
            checkpoint_dir = output_dir / f"checkpoint-{state.global_step}"
            save_metadata = save_fsdp_adapter_only(trainer, checkpoint_dir)
            if trainer.is_world_process_zero():
                processor.save_pretrained(str(checkpoint_dir))
                state.save_to_json(str(checkpoint_dir / "trainer_state.json"))
                (checkpoint_dir / "adapter_save.json").write_text(
                    json.dumps(save_metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            if dist.is_initialized():
                dist.barrier()
            return control

    trainer.add_callback(AdapterOnlyCheckpointCallback())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen multimodal SFT requires CUDA")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, set_seed
    from transformers import AutoModelForImageTextToText
    from transformers import Trainer

    set_seed(args.seed)
    model_path = Path(args.model_path).expanduser().resolve()
    train_path = Path(args.train_jsonl).expanduser().resolve()
    val_path = Path(args.val_jsonl).expanduser().resolve() if args.val_jsonl else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = JsonlDataset(train_path, args.limit)
    val_dataset = JsonlDataset(val_path, args.val_limit) if val_path and val_path.exists() else None
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    collator = FormTSRCollator(
        processor,
        max_seq_length=args.max_seq_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    dtype = torch.bfloat16 if not args.no_bf16 else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if not args.no_gradient_checkpointing and not args.fsdp_activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    # Linear-attention in_proj modules remain frozen for portable adapter serving.
    target_modules: list[str]
    if args.target_modules_regex:
        pattern = re.compile(args.target_modules_regex)
        matched_modules = [
            (name, module)
            for name, module in model.named_modules()
            if pattern.fullmatch(name)
        ]
        if not matched_modules:
            raise ValueError(
                f"--target-modules-regex matched no modules: {args.target_modules_regex}"
            )
        unsupported = [
            f"{name} ({type(module).__name__})"
            for name, module in matched_modules
            if not isinstance(module, torch.nn.Linear)
        ]
        if unsupported:
            raise ValueError(
                "--target-modules-regex matched unsupported modules: "
                + ", ".join(unsupported[:10])
            )
        target_modules = [name for name, _ in matched_modules]
    else:
        target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
        if not target_modules:
            raise ValueError("--target-modules cannot be empty")
    print(
        f"resolved {len(target_modules)} LoRA target modules; "
        f"first={target_modules[0]!r}, last={target_modules[-1]!r}"
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = build_training_args(args, output_dir, has_eval=val_dataset is not None)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    if args.fsdp_adapter_save_steps:
        if not args.fsdp_adapter_only_save:
            raise ValueError(
                "--fsdp-adapter-save-steps requires --fsdp-adapter-only-save"
            )
        add_fsdp_adapter_checkpoint_callback(
            trainer,
            output_dir,
            processor,
            args.fsdp_adapter_save_steps,
        )
    resume = args.resume_from_checkpoint or None
    result = trainer.train(resume_from_checkpoint=resume)
    adapter_save: dict[str, Any] | None = None
    if args.fsdp_adapter_only_save:
        adapter_save = save_fsdp_adapter_only(trainer, output_dir)
    else:
        trainer.save_model(str(output_dir))
    trainer.save_state()
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(output_dir))
        summary = {
            "model_path": str(model_path),
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path) if val_path else "",
            "output_dir": str(output_dir),
            "train_count": len(train_dataset),
            "val_count": len(val_dataset) if val_dataset else 0,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_length": args.max_seq_length,
            "max_pixels": args.max_pixels,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_modules,
            "target_modules_regex": args.target_modules_regex,
            "fsdp": args.fsdp,
            "fsdp_transformer_layer_cls_to_wrap": args.fsdp_transformer_layer_cls_to_wrap,
            "fsdp_backward_prefetch": args.fsdp_backward_prefetch,
            "fsdp_limit_all_gathers": args.fsdp_limit_all_gathers,
            "fsdp_activation_checkpointing": args.fsdp_activation_checkpointing,
            "fsdp_adapter_only_save": args.fsdp_adapter_only_save,
            "fsdp_adapter_save_steps": args.fsdp_adapter_save_steps,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "bf16": not args.no_bf16,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "adapter_save": adapter_save,
            "train_metrics": {
                key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))
            },
        }
        (output_dir / "train_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
