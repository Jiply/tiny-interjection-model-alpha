from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Any

from .format import build_prompt
from .io import read_jsonl
from .parser import target_to_text
from .schema import Target, event_from_dict

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TEXT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_prompt_completion_rows(train_file: Path, max_samples: int | None):
    rows = []
    for row in read_jsonl(train_file):
        prompt_raw = row.get("prompt")
        completion_raw = row.get("completion")
        if (
            isinstance(prompt_raw, str)
            and prompt_raw
            and isinstance(completion_raw, str)
            and completion_raw
        ):
            rows.append({"prompt": prompt_raw, "completion": completion_raw})
            continue

        events_raw = row.get("events")
        target_raw = row.get("target")
        if not isinstance(events_raw, list) or not isinstance(target_raw, dict):
            continue
        events = tuple(event_from_dict(event) for event in events_raw)
        target = Target(
            action=target_raw["action"],
            messages=tuple(target_raw.get("messages", [])),
        )
        rows.append(
            {
                "prompt": f"{build_prompt(events)}\n",
                "completion": target_to_text(target),
            }
        )
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(
            f"No training rows found in {train_file}. Run `make data` first."
        )
    return rows


def load_text_dataset(train_file: Path, max_samples: int | None):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies with `make setup` before training."
        ) from exc

    rows = load_prompt_completion_rows(train_file, max_samples)
    return Dataset.from_list(rows)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str, use_4bit: bool, *, purpose: str = "training"):
    import torch
    from transformers import AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }

    if use_4bit:
        if platform.system() == "Darwin":
            raise RuntimeError(
                f"4-bit {purpose} via bitsandbytes is not available on macOS. Run on a CUDA Linux host or pass --no-4bit."
            )
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes/transformers quantization support is required for 4-bit QLoRA."
            ) from exc

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    dataset = load_text_dataset(args.train_file, args.max_samples)
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, use_4bit=not args.no_4bit)

    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = None
    if args.initial_adapter:
        model = PeftModel.from_pretrained(
            model, str(args.initial_adapter), is_trainable=True
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(TEXT_TARGET_MODULES),
            bias="none",
            task_type="CAUSAL_LM",
        )

    train_config = SFTConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        max_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16 if args.bf16 is not None else torch.cuda.is_available(),
        fp16=args.fp16,
        report_to=[],
        packing=False,
        completion_only_loss=True,
    )

    trainer_kwargs = {
        "model": model,
        "args": train_config,
        "train_dataset": dataset,
    }
    if lora_config is not None:
        trainer_kwargs["peft_config"] = lora_config
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    resume_from_checkpoint = (
        str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a text interaction model with LoRA."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--train-file", type=Path, default=Path("data/processed/train.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/qwen3-4b-instruct-2507")
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("adapters/qwen3-4b-instruct-2507")
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--initial-adapter", type=Path, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
