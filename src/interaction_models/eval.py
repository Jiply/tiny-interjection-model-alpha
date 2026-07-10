from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmarks import DOUBLE_TEXT_BENCH, BenchCase, map_semantic_turn_action
from .inference import InteractionEngine
from .io import read_jsonl
from .parser import TargetParseError, parse_target
from .schema import Event, Target, event_from_dict
from .train import DEFAULT_MODEL


def bench_cases_from_jsonl(path: Path) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for row in read_jsonl(path):
        target_raw = row.get("target") or {}
        events = tuple(event_from_dict(event) for event in row.get("events", []))
        target = Target(
            action=target_raw["action"],
            messages=tuple(target_raw.get("messages", [])),
        )
        cases.append(
            BenchCase(
                case_id=str(row.get("case_id") or len(cases)),
                events=events,
                target=target,
                expected_contains=tuple(row.get("expected_contains", [])),
            )
        )
    return cases


def load_doubletext_cases(path: Path | None) -> list[BenchCase]:
    if path and path.exists():
        return bench_cases_from_jsonl(path)
    return list(DOUBLE_TEXT_BENCH)


def score_cases(engine: InteractionEngine, cases: list[BenchCase]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    schema_valid = 0
    action_correct = 0
    contains_correct = 0
    contains_total = 0
    wait_cases = 0
    premature_responses = 0

    for case in cases:
        decision = engine.decide(case.events)
        try:
            parse_target(decision.raw, strict=True)
            valid = True
            schema_valid += 1
        except TargetParseError:
            valid = False

        predicted = decision.target.action
        if predicted == case.target.action:
            action_correct += 1

        if case.target.action == "wait":
            wait_cases += 1
            if predicted != "wait":
                premature_responses += 1

        joined_messages = " ".join(decision.target.messages).lower()
        expected_contains = tuple(item.lower() for item in case.expected_contains)
        contains_ok = all(item in joined_messages for item in expected_contains)
        if expected_contains:
            contains_total += 1
            if contains_ok:
                contains_correct += 1

        results.append(
            {
                "case_id": case.case_id,
                "expected_action": case.target.action,
                "predicted_action": predicted,
                "schema_valid": valid,
                "contains_ok": contains_ok if expected_contains else None,
                "raw": decision.raw,
            }
        )

    total = max(1, len(cases))
    return {
        "total": len(cases),
        "schema_valid_rate": schema_valid / total,
        "action_accuracy": action_correct / total,
        "expected_contains_accuracy": contains_correct / max(1, contains_total),
        "premature_response_rate": premature_responses / max(1, wait_cases),
        "results": results,
    }


def semantic_turn_examples(limit: int) -> list[BenchCase]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies before semantic turn-taking evaluation."
        ) from exc

    dataset = load_dataset(
        "anyreach-ai/semantic-turn-taking-benchmark", split="test", streaming=True
    )
    cases: list[BenchCase] = []
    for row in dataset:
        if len(cases) >= limit:
            break
        if not isinstance(row, dict):
            continue
        action_raw = row.get("action") or row.get("label") or row.get("target")
        text_raw = (
            row.get("text")
            or row.get("prompt")
            or row.get("conversation")
            or row.get("context")
        )
        if action_raw is None or text_raw is None:
            continue
        try:
            action = map_semantic_turn_action(str(action_raw))
        except ValueError:
            continue
        events = (Event(role="user", text=str(text_raw), dt_ms=0),)
        messages = () if action == "wait" else ("Continuing.",)
        cases.append(
            BenchCase(
                case_id=f"semantic-{len(cases)}",
                events=events,
                target=Target(action=action, messages=messages),  # type: ignore[arg-type]
            )
        )
    return cases


def build_engine(args: argparse.Namespace, *, base_only: bool) -> InteractionEngine:
    return InteractionEngine(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        base_only=base_only,
        heuristic=args.heuristic,
        use_4bit=not args.no_4bit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TIMA behavior.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("adapters/qwen3-4b-instruct-2507")
    )
    parser.add_argument(
        "--bench-file", type=Path, default=Path("data/processed/doubletextbench.jsonl")
    )
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--compare-base", action="store_true")
    parser.add_argument("--semantic-turn", action="store_true")
    parser.add_argument("--semantic-limit", type=int, default=200)
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    cases = load_doubletext_cases(args.bench_file)
    report: dict[str, Any] = {"doubletextbench": {}}

    if args.base_only or args.compare_base:
        report["doubletextbench"]["base"] = score_cases(
            build_engine(args, base_only=True), cases
        )

    if not args.base_only:
        report["doubletextbench"]["adapter"] = score_cases(
            build_engine(args, base_only=False), cases
        )

    if args.semantic_turn:
        semantic_cases = semantic_turn_examples(args.semantic_limit)
        report["semantic_turn"] = score_cases(
            build_engine(args, base_only=False), semantic_cases
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
