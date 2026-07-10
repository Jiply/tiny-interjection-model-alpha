from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from candidate_utils import (
    ACTIONS,
    extract_json_object,
    fingerprint,
    normalize_example,
    validation_errors,
)

PLACEHOLDER_RE = re.compile(
    r"\b(?:lorem ipsum|placeholder|example text|sample text|foo(?:\s+bar(?:\s+baz)?)?|xxx|todo)\b",
    re.IGNORECASE,
)
META_RE = re.compile(
    r"\b(?:as an ai|json object|target action|training example)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def event_texts(example: dict[str, Any]) -> list[str]:
    return [str(event.get("text") or "") for event in example.get("events", [])]


def target_texts(example: dict[str, Any]) -> list[str]:
    target = example.get("target", {})
    messages = target.get("messages", []) if isinstance(target, dict) else []
    return [str(message or "") for message in messages]


def last_user_text(example: dict[str, Any]) -> str:
    for event in reversed(example.get("events", [])):
        if event.get("role") == "user":
            return str(event.get("text") or "")
    return ""


def quality_report(path: Path) -> dict[str, Any]:
    rows = 0
    valid = 0
    invalid = 0
    action_counts: Counter[str] = Counter()
    timing_counts: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    last_users: Counter[str] = Counter()
    placeholder_rows = 0
    meta_rows = 0
    short_event_rows = 0
    short_target_rows = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows += 1
            try:
                raw = extract_json_object(stripped)
                example = normalize_example(
                    raw,
                    source=str(raw.get("source") or "quality"),
                    case_id=str(raw.get("case_id") or f"quality-{line_number:06d}"),
                )
                errors = validation_errors(example)
                if errors:
                    invalid += 1
                    reject_counts.update(errors)
                    continue

                valid += 1
                action_counts[example["target"]["action"]] += 1
                timing_bucket = raw.get("timing_bucket")
                if isinstance(timing_bucket, str) and timing_bucket:
                    timing_counts[timing_bucket] += 1
                seen[fingerprint(example)] += 1
                last_users[norm(last_user_text(example))] += 1

                texts = event_texts(example) + target_texts(example)
                if any(PLACEHOLDER_RE.search(text) for text in texts):
                    placeholder_rows += 1
                if any(META_RE.search(text) for text in texts):
                    meta_rows += 1
                if any(len(words(text)) < 3 for text in event_texts(example)):
                    short_event_rows += 1
                non_wait_targets = target_texts(example)
                if non_wait_targets and any(
                    len(words(text)) < 3 for text in non_wait_targets
                ):
                    short_target_rows += 1
            except Exception as exc:
                invalid += 1
                reject_counts.update([str(exc)])

    duplicate_rows = sum(count - 1 for count in seen.values() if count > 1)
    duplicate_rate = duplicate_rows / valid if valid else 1.0
    unique_last_user_ratio = len(last_users) / valid if valid else 0.0
    short_event_rate = short_event_rows / valid if valid else 1.0
    short_target_rate = short_target_rows / valid if valid else 1.0

    return {
        "rows": rows,
        "valid": valid,
        "invalid": invalid,
        "action_counts": dict(action_counts),
        "timing_counts": dict(timing_counts),
        "reject_counts": dict(reject_counts),
        "duplicate_rate": duplicate_rate,
        "unique_last_user_ratio": unique_last_user_ratio,
        "placeholder_rows": placeholder_rows,
        "meta_rows": meta_rows,
        "short_event_rate": short_event_rate,
        "short_target_rate": short_target_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Heuristic quality gate for generated TIM candidates."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--min-per-action", type=int, default=20)
    parser.add_argument("--max-duplicate-rate", type=float, default=0.05)
    parser.add_argument("--min-unique-last-user-ratio", type=float, default=0.8)
    parser.add_argument("--max-short-event-rate", type=float, default=0.25)
    parser.add_argument("--max-short-target-rate", type=float, default=0.10)
    args = parser.parse_args()

    report = quality_report(Path(args.input))
    failures: list[str] = []

    if report["invalid"]:
        failures.append(f"{report['invalid']} invalid rows")
    if report["valid"] < args.min_count:
        failures.append(f"valid count {report['valid']} below {args.min_count}")
    for action in ACTIONS:
        count = report["action_counts"].get(action, 0)
        if count < args.min_per_action:
            failures.append(f"{action} count {count} below {args.min_per_action}")
    if report["duplicate_rate"] > args.max_duplicate_rate:
        failures.append(
            f"duplicate rate {report['duplicate_rate']:.3f} above {args.max_duplicate_rate:.3f}"
        )
    if report["unique_last_user_ratio"] < args.min_unique_last_user_ratio:
        failures.append(
            f"unique last-user ratio {report['unique_last_user_ratio']:.3f} below "
            f"{args.min_unique_last_user_ratio:.3f}"
        )
    if report["placeholder_rows"]:
        failures.append(f"{report['placeholder_rows']} rows contain placeholder text")
    if report["meta_rows"]:
        failures.append(f"{report['meta_rows']} rows contain meta text")
    if report["short_event_rate"] > args.max_short_event_rate:
        failures.append(
            f"short event rate {report['short_event_rate']:.3f} above {args.max_short_event_rate:.3f}"
        )
    if report["short_target_rate"] > args.max_short_target_rate:
        failures.append(
            f"short target rate {report['short_target_rate']:.3f} above {args.max_short_target_rate:.3f}"
        )

    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
