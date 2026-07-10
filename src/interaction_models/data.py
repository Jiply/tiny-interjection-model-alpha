from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .benchmarks import DOUBLE_TEXT_BENCH
from .format import build_prompt
from .io import read_jsonl, write_jsonl
from .parser import TargetParseError, parse_target, target_to_text
from .schema import Event, Example, Target, event_from_dict, example_to_dict
from .synthetic import synthetic_examples, synthetic_question_examples

DEFAULT_PUBLIC_DATASETS = (
    ("marcodsn/SOC-2602", "train"),
    ("marcodsn/SOC-2508", "train"),
    ("lmsys/lmsys-chat-1m", "train"),
)

INCOMPLETE_ENDINGS = (
    "and",
    "but",
    "or",
    "because",
    "where",
    "when",
    "the",
    "a",
    "an",
    "to",
    "for",
    "with",
    "like",
    "says",
    "about",
)

CORRECTION_MARKERS = (
    "actually",
    "wait",
    "no ",
    "not ",
    "instead",
    "never mind",
    "nevermind",
    "scratch that",
    "i mean",
)

BACKCHANNELS = {"yeah", "yep", "ok", "okay", "mm", "mhm", "uh huh", "uh-huh", "right"}


def is_incomplete_text(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return True
    if stripped.endswith(("...", ",", ":", "-", "(")):
        return True
    if stripped.endswith((".", "?", "!")):
        return False
    words = stripped.split()
    return bool(words and words[-1] in INCOMPLETE_ENDINGS)


def is_correction_or_redirect(text: str) -> bool:
    lowered = f" {text.strip().lower()} "
    return any(marker in lowered for marker in CORRECTION_MARKERS)


def is_backchannel(text: str) -> bool:
    normalized = text.strip().lower().strip(".!,")
    return normalized in BACKCHANNELS


def collect_messages(
    events: tuple[Event, ...], start: int, role: str = "assistant", limit: int = 2
) -> tuple[str, ...]:
    messages: list[str] = []
    for event in events[start:]:
        if event.role != role:
            break
        if event.text:
            messages.append(event.text)
        if len(messages) >= limit:
            break
    return tuple(messages)


def examples_from_events(
    events: Iterable[Event],
    *,
    source: str,
    case_prefix: str,
    burst_ms: int = 3_000,
    interruption_ms: int = 2_500,
) -> list[Example]:
    normalized = tuple(event for event in events if event.text)
    examples: list[Example] = []

    for index, event in enumerate(normalized):
        prefix = normalized[: index + 1]
        next_event = normalized[index + 1] if index + 1 < len(normalized) else None
        prev_event = normalized[index - 1] if index > 0 else None

        if event.role == "user":
            if (
                next_event
                and next_event.role == "user"
                and next_event.dt_ms <= burst_ms
                and is_incomplete_text(event.text)
            ):
                examples.append(
                    Example(
                        events=prefix,
                        target=Target(action="wait", messages=()),
                        source=source,
                        case_id=f"{case_prefix}-{index}-wait",
                    )
                )
                continue

            assistant_messages = collect_messages(normalized, index + 1)
            if (
                prev_event
                and prev_event.role == "assistant"
                and event.dt_ms <= interruption_ms
            ):
                if is_backchannel(event.text):
                    next_assistant = assistant_messages or (
                        "Continuing from the previous point.",
                    )
                    examples.append(
                        Example(
                            events=prefix,
                            target=Target(
                                action="continue", messages=next_assistant[:1]
                            ),
                            source=source,
                            case_id=f"{case_prefix}-{index}-continue",
                        )
                    )
                elif is_correction_or_redirect(event.text):
                    next_assistant = assistant_messages or (
                        "Got it. I will adjust to the latest correction.",
                    )
                    examples.append(
                        Example(
                            events=prefix,
                            target=Target(
                                action="interject", messages=next_assistant[:1]
                            ),
                            source=source,
                            case_id=f"{case_prefix}-{index}-interject",
                        )
                    )
                continue

            if assistant_messages:
                examples.append(
                    Example(
                        events=prefix,
                        target=Target(action="respond", messages=assistant_messages),
                        source=source,
                        case_id=f"{case_prefix}-{index}-respond",
                    )
                )

        if event.role == "assistant" and next_event and next_event.role == "assistant":
            examples.append(
                Example(
                    events=prefix,
                    target=Target(action="continue", messages=(next_event.text,)),
                    source=source,
                    case_id=f"{case_prefix}-{index}-continue",
                )
            )

    return examples


def extract_messages_container(record: dict[str, Any]) -> Any:
    for key in (
        "events",
        "messages",
        "conversation",
        "conversations",
        "chat",
        "dialogue",
        "turns",
    ):
        if key in record:
            return record[key]
    return None


def events_from_record(record: dict[str, Any]) -> tuple[Event, ...]:
    container = extract_messages_container(record)
    if not isinstance(container, list):
        return ()

    events: list[Event] = []
    speaker_role_map: dict[str, str] = {}
    next_role = "user"

    for item in container:
        if not isinstance(item, dict):
            continue

        raw_speaker = (
            item.get("role")
            or item.get("speaker")
            or item.get("author")
            or item.get("from")
            or item.get("name")
        )
        fallback = None
        if raw_speaker:
            speaker_key = str(raw_speaker).strip().lower()
            if speaker_key not in speaker_role_map:
                speaker_role_map[speaker_key] = next_role
                next_role = "assistant" if next_role == "user" else "user"
            fallback = speaker_role_map[speaker_key]

        try:
            event = event_from_dict(item, fallback_role=fallback)  # type: ignore[arg-type]
        except ValueError:
            continue
        if event.role == "system":
            continue
        events.append(event)

    return tuple(events)


def seed_examples() -> list[Example]:
    return [
        Example(
            events=case.events,
            target=case.target,
            source="doubletextbench",
            case_id=case.case_id,
        )
        for case in DOUBLE_TEXT_BENCH
    ]


def public_examples(limit_per_dataset: int) -> list[Example]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install project dependencies before using --include-public."
        ) from exc

    examples: list[Example] = []
    for dataset_name, split in DEFAULT_PUBLIC_DATASETS:
        try:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        except Exception:
            continue

        for row_index, record in enumerate(
            itertools.islice(dataset, limit_per_dataset)
        ):
            if not isinstance(record, dict):
                continue
            events = events_from_record(record)
            if len(events) < 2:
                continue
            examples.extend(
                examples_from_events(
                    events,
                    source=dataset_name,
                    case_prefix=f"{dataset_name.replace('/', '-')}-{row_index}",
                )
            )

    return examples


def rows_from_examples(examples: Iterable[Example]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        prompt = f"{build_prompt(example.events)}\n"
        completion = target_to_text(example.target)
        row = example_to_dict(example, text=f"{prompt}{completion}")
        row["prompt"] = prompt
        row["completion"] = completion
        rows.append(row)
    return rows


def rows_from_bench_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in DOUBLE_TEXT_BENCH:
        example = Example(
            events=case.events,
            target=case.target,
            source="doubletextbench",
            case_id=case.case_id,
        )
        prompt = f"{build_prompt(example.events)}\n"
        completion = target_to_text(example.target)
        row = example_to_dict(example, text=f"{prompt}{completion}")
        row["prompt"] = prompt
        row["completion"] = completion
        row["expected_contains"] = list(case.expected_contains)
        rows.append(row)
    return rows


def rows_from_adaptive_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    holdout_ids = {case.case_id for case in DOUBLE_TEXT_BENCH}
    for row in read_jsonl(path):
        prompt = row.get("prompt")
        completion = row.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            continue
        if row.get("case_id") in holdout_ids:
            continue

        try:
            trusted_target = parse_target(completion)
        except TargetParseError:
            continue

        selected_completion = completion
        enhanced_completion = row.get("enhanced_completion")
        if trusted_target.action != "wait" and isinstance(enhanced_completion, str):
            try:
                enhanced_target = parse_target(enhanced_completion)
            except TargetParseError:
                enhanced_target = None
            if (
                enhanced_target is not None
                and enhanced_target.action == trusted_target.action
            ):
                selected_completion = enhanced_completion

        rows.append(
            {
                "case_id": row.get("case_id"),
                "completion": selected_completion,
                "prompt": prompt,
                "source": row.get("source", "adaptive"),
                "text": f"{prompt}{selected_completion}",
            }
        )
    return rows


def split_examples(
    examples: list[Example], val_ratio: float, seed: int
) -> tuple[list[Example], list[Example]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def action_counts(examples: Iterable[Example]) -> dict[str, int]:
    return dict(sorted(Counter(example.target.action for example in examples).items()))


def build_dataset(
    *,
    output_dir: Path,
    include_public: bool,
    public_limit: int,
    synthetic_count: int,
    val_ratio: float,
    seed: int,
    additional_jsonl: Path | None = None,
    synthetic_question_boost: int = 0,
) -> dict[str, int]:
    examples = synthetic_examples(synthetic_count, seed=seed)
    examples.extend(
        synthetic_question_examples(synthetic_question_boost, seed=seed + 1)
    )
    if include_public:
        examples.extend(public_examples(public_limit))
    if not examples:
        raise ValueError(
            "No training examples built. Use --synthetic-count > 0 or --include-public."
        )

    train_examples, val_examples = split_examples(
        examples, val_ratio=val_ratio, seed=seed
    )
    train_rows = rows_from_examples(train_examples)
    additional_rows = (
        rows_from_adaptive_file(additional_jsonl) if additional_jsonl else []
    )
    train_rows.extend(additional_rows)
    train_count = write_jsonl(output_dir / "train.jsonl", train_rows)
    val_count = write_jsonl(output_dir / "val.jsonl", rows_from_examples(val_examples))
    bench_count = write_jsonl(
        output_dir / "doubletextbench.jsonl",
        rows_from_bench_cases(),
    )
    manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "synthetic_count": synthetic_count,
        "synthetic_question_boost": synthetic_question_boost,
        "include_public": include_public,
        "public_limit": public_limit,
        "train": train_count,
        "additional_jsonl": str(additional_jsonl) if additional_jsonl else None,
        "additional_rows": len(additional_rows),
        "val": val_count,
        "doubletextbench": bench_count,
        "train_action_counts": action_counts(train_examples),
        "val_action_counts": action_counts(val_examples),
        "holdout": "doubletextbench",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"train": train_count, "val": val_count, "doubletextbench": bench_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TIM training data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--include-public", action="store_true")
    parser.add_argument("--public-limit", type=int, default=1_000)
    parser.add_argument("--synthetic-count", type=int, default=2_000)
    parser.add_argument("--synthetic-question-boost", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--additional-jsonl", type=Path, default=None)
    args = parser.parse_args()

    counts = build_dataset(
        output_dir=args.output_dir,
        include_public=args.include_public,
        public_limit=args.public_limit,
        synthetic_count=args.synthetic_count,
        val_ratio=args.val_ratio,
        seed=args.seed,
        additional_jsonl=args.additional_jsonl,
        synthetic_question_boost=args.synthetic_question_boost,
    )
    print(
        f"Wrote {counts['train']} train, {counts['val']} val, "
        f"{counts['doubletextbench']} DoubleTextBench examples to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
