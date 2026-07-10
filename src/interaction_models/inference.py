from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import is_backchannel, is_correction_or_redirect, is_incomplete_text
from .format import build_prompt
from .parser import parse_target_or_wait, target_to_text
from .schema import Event, Target
from .train import DEFAULT_MODEL, load_model, load_tokenizer


@dataclass(frozen=True)
class Decision:
    target: Target
    raw: str
    mode: str


def call_llama_cli(
    *,
    executable: Path,
    model_path: Path,
    adapter_path: Path,
    prompt: str,
    max_new_tokens: int,
):
    """When local GGUF LoRA inference is selected, request one llama.cpp decision."""
    result = subprocess.run(
        [
            str(executable),
            "-m",
            str(model_path),
            "--lora",
            str(adapter_path),
            "-ngl",
            "99",
            "-c",
            "4096",
            "-b",
            "256",
            "-ub",
            "256",
            "--simple-io",
            "--log-disable",
            "--no-display-prompt",
            "-st",
            "-n",
            str(max_new_tokens),
            "--temp",
            "0",
            "--seed",
            "1",
            "-p",
            prompt,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    start = result.stdout.rfind("<act>")
    end = result.stdout.find("<done/>", start)
    if start < 0 or end < 0:
        raise RuntimeError("llama.cpp returned no TIMA decision.")
    return result.stdout[start : end + len("<done/>")]


def heuristic_decision(events: tuple[Event, ...]) -> Decision:
    if not events:
        target = Target(action="wait", messages=())
        return Decision(target=target, raw=target_to_text(target), mode="heuristic")

    latest = events[-1]
    previous = events[-2] if len(events) >= 2 else None

    if latest.role == "user":
        if previous and previous.role == "assistant":
            if is_backchannel(latest.text):
                target = Target(
                    action="continue", messages=("Continuing from the previous point.",)
                )
            elif is_correction_or_redirect(latest.text):
                target = Target(
                    action="interject",
                    messages=("Got it. I will adjust to the latest message.",),
                )
            else:
                target = Target(
                    action="respond",
                    messages=("Got it. I will respond to the latest message.",),
                )
        elif is_incomplete_text(latest.text):
            target = Target(action="wait", messages=())
        else:
            target = Target(
                action="respond",
                messages=("Got it. Here is a focused response to the latest message.",),
            )
    elif latest.role == "assistant":
        target = Target(
            action="continue", messages=("Continuing with the next useful point.",)
        )
    else:
        target = Target(action="wait", messages=())

    return Decision(target=target, raw=target_to_text(target), mode="heuristic")


class InteractionEngine:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        adapter_dir: Path | None = None,
        base_only: bool = False,
        heuristic: bool = False,
        use_4bit: bool = True,
        max_new_tokens: int = 192,
        llama_cli_path: Path | None = None,
        llama_model_path: Path | None = None,
        llama_adapter_path: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.adapter_dir = adapter_dir
        self.base_only = base_only
        self.llama_cli_path = llama_cli_path
        self.llama_model_path = llama_model_path
        self.llama_adapter_path = llama_adapter_path
        llama_paths = (llama_cli_path, llama_model_path, llama_adapter_path)
        if any(llama_paths) and not all(llama_paths):
            raise ValueError("llama.cpp requires CLI, model, and adapter paths.")
        self.heuristic = heuristic
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        if not self.heuristic and self.llama_model_path is None:
            self._load_model()

    def _load_model(self) -> None:
        tokenizer = load_tokenizer(self.model_name)
        model = load_model(self.model_name, use_4bit=self.use_4bit, purpose="inference")

        if self.adapter_dir and not self.base_only:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_dir)

        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def decide(self, events: tuple[Event, ...]) -> Decision:
        if self.heuristic:
            return heuristic_decision(events)

        if (
            self.llama_cli_path is not None
            and self.llama_model_path is not None
            and self.llama_adapter_path is not None
        ):
            raw = call_llama_cli(
                executable=self.llama_cli_path,
                model_path=self.llama_model_path,
                adapter_path=self.llama_adapter_path,
                prompt=build_prompt(events),
                max_new_tokens=self.max_new_tokens,
            )
            return Decision(target=parse_target_or_wait(raw), raw=raw, mode="llama.cpp")

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded.")

        import torch

        prompt = build_prompt(events)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[-1]
        raw = self.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        target = parse_target_or_wait(raw)
        return Decision(
            target=target, raw=raw, mode="base" if self.base_only else "adapter"
        )
