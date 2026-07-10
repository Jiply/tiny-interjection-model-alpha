from __future__ import annotations

import argparse
from pathlib import Path


def merge_adapter(base_model: str, adapter_dir: Path, output_dir: Path) -> None:
    """When training passes, merge the LoRA into portable base-model weights."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a TIMA LoRA adapter into its base model."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    merge_adapter(args.base_model, args.adapter_dir, args.output_dir)


if __name__ == "__main__":
    main()
