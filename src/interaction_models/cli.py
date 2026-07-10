from __future__ import annotations

import argparse
import time
from pathlib import Path

from .inference import InteractionEngine
from .schema import Event
from .train import DEFAULT_MODEL


def print_decision(engine: InteractionEngine, events: tuple[Event, ...]) -> None:
    decision = engine.decide(events)
    print(f"\n[{decision.mode}] action={decision.target.action}")
    for message in decision.target.messages:
        print(f"assistant> {message}")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive typed-chat demo for the interaction model."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("adapters/qwen3-4b-instruct-2507")
    )
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--llama-cli-path", type=Path, default=None)
    parser.add_argument("--llama-model-path", type=Path, default=None)
    parser.add_argument("--llama-adapter-path", type=Path, default=None)
    args = parser.parse_args()

    engine = InteractionEngine(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        base_only=args.base_only,
        heuristic=args.heuristic,
        use_4bit=not args.no_4bit,
        llama_cli_path=args.llama_cli_path,
        llama_model_path=args.llama_model_path,
        llama_adapter_path=args.llama_adapter_path,
    )

    events: list[Event] = []
    last_at = time.monotonic()
    print(
        "Type user messages. Blank line forces a decision. Use /assistant text, /reset, or /quit."
    )

    while True:
        raw = input("you> ")
        now = time.monotonic()
        dt_ms = max(0, int((now - last_at) * 1000))
        last_at = now

        if raw.strip() == "":
            if events:
                print_decision(engine, tuple(events))
            continue
        if raw.strip() == "/quit":
            break
        if raw.strip() == "/reset":
            events.clear()
            print("reset")
            continue
        if raw.startswith("/assistant "):
            events.append(
                Event(
                    role="assistant",
                    text=raw[len("/assistant ") :].strip(),
                    dt_ms=dt_ms,
                )
            )
        else:
            events.append(Event(role="user", text=raw.strip(), dt_ms=dt_ms))


if __name__ == "__main__":
    main()
