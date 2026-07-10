from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from candidate_utils import (
    ACTIONS,
    add_training_text,
    extract_json_object,
    fingerprint,
    normalize_example,
    validation_errors,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and dedupe TIMA candidate JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/candidates.valid.jsonl")
    parser.add_argument("--rejects", default="data/candidates.invalid.jsonl")
    parser.add_argument("--source", default="do-validated")
    parser.add_argument("--case-prefix", default="do-valid")
    parser.add_argument("--max-per-action", type=int, default=0)
    parser.add_argument("--min-per-action", type=int, default=1)
    parser.add_argument("--no-text", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rejects_path = Path(args.rejects)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()

    with (
        input_path.open(encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as out,
        rejects_path.open("w", encoding="utf-8") as rej,
    ):
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = extract_json_object(stripped)
                source = str(raw.get("source") or args.source)
                case_id = str(
                    raw.get("case_id") or f"{args.case_prefix}-{line_number:06d}"
                )
                example = normalize_example(raw, source=source, case_id=case_id)
                errors = validation_errors(example)
                key = fingerprint(example) if not errors else ""
                action = example.get("target", {}).get("action")
                if key in seen:
                    errors.append("duplicate")
                if (
                    args.max_per_action
                    and action in ACTIONS
                    and counts[action] >= args.max_per_action
                ):
                    errors.append(f"max-per-action reached for {action}")
                if errors:
                    rejected.update(errors)
                    rej.write(
                        json.dumps(
                            {"line": line_number, "errors": errors, "raw": raw},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                seen.add(key)
                counts[action] += 1
                accepted += 1
                row = example if args.no_text else add_training_text(example)
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception as exc:
                rejected.update([str(exc)])
                rej.write(
                    json.dumps(
                        {"line": line_number, "errors": [str(exc)]}, ensure_ascii=False
                    )
                    + "\n"
                )

    balance_errors = []
    if args.min_per_action:
        for action in ACTIONS:
            if counts[action] < args.min_per_action:
                balance_errors.append(
                    f"{action} has {counts[action]} accepted examples; expected at least {args.min_per_action}"
                )

    summary = {
        "accepted": accepted,
        "action_counts": dict(counts),
        "balance_errors": balance_errors,
        "reject_counts": dict(rejected),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
    return 0 if accepted and not balance_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
