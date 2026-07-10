from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class VerificationError(AssertionError):
    """Raised when an evaluation report fails the quality gate."""


def _metric(section: dict[str, Any], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, int | float):
        raise VerificationError(f"Missing numeric metric: {name}")
    return float(value)


def _failed_cases(section: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for result in section.get("results", []):
        case_id = str(result.get("case_id", "unknown"))
        reasons: list[str] = []
        if not result.get("schema_valid"):
            reasons.append("schema")
        if result.get("expected_action") != result.get("predicted_action"):
            reasons.append(
                f"action {result.get('predicted_action')} != {result.get('expected_action')}"
            )
        if result.get("contains_ok") is False:
            reasons.append("expected_contains")
        if reasons:
            failures.append(f"{case_id}: {', '.join(reasons)}")
    return failures


def verify_report(
    report: dict[str, Any],
    *,
    suite: str,
    model_key: str,
    min_total: int,
    min_schema_valid_rate: float,
    min_action_accuracy: float,
    min_expected_contains_accuracy: float,
    max_premature_response_rate: float,
) -> dict[str, Any]:
    try:
        section = report[suite][model_key]
    except KeyError as exc:
        raise VerificationError(f"Missing report section: {suite}.{model_key}") from exc
    if not isinstance(section, dict):
        raise VerificationError(f"Report section is not an object: {suite}.{model_key}")

    total = int(section.get("total", 0))
    failures: list[str] = []
    if total < min_total:
        failures.append(f"total {total} < {min_total}")

    checks = (
        (
            "schema_valid_rate",
            _metric(section, "schema_valid_rate"),
            ">=",
            min_schema_valid_rate,
        ),
        (
            "action_accuracy",
            _metric(section, "action_accuracy"),
            ">=",
            min_action_accuracy,
        ),
        (
            "expected_contains_accuracy",
            _metric(section, "expected_contains_accuracy"),
            ">=",
            min_expected_contains_accuracy,
        ),
        (
            "premature_response_rate",
            _metric(section, "premature_response_rate"),
            "<=",
            max_premature_response_rate,
        ),
    )
    for name, observed, operator, threshold in checks:
        if operator == ">=" and observed < threshold:
            failures.append(f"{name} {observed:.4f} < {threshold:.4f}")
        if operator == "<=" and observed > threshold:
            failures.append(f"{name} {observed:.4f} > {threshold:.4f}")

    case_failures = _failed_cases(section)
    if case_failures:
        failures.append("failed_cases: " + "; ".join(case_failures))

    if failures:
        raise VerificationError("\n".join(failures))

    return {
        "suite": suite,
        "model_key": model_key,
        "total": total,
        "schema_valid_rate": _metric(section, "schema_valid_rate"),
        "action_accuracy": _metric(section, "action_accuracy"),
        "expected_contains_accuracy": _metric(section, "expected_contains_accuracy"),
        "premature_response_rate": _metric(section, "premature_response_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify TIMA eval report thresholds.")
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("runs/qwen3-4b-instruct-2507/eval.json"),
    )
    parser.add_argument("--suite", default="doubletextbench")
    parser.add_argument("--model-key", default="adapter")
    parser.add_argument("--min-total", type=int, default=40)
    parser.add_argument("--min-schema-valid-rate", type=float, default=1.0)
    parser.add_argument("--min-action-accuracy", type=float, default=0.95)
    parser.add_argument("--min-expected-contains-accuracy", type=float, default=0.95)
    parser.add_argument("--max-premature-response-rate", type=float, default=0.0)
    args = parser.parse_args()

    report = json.loads(args.report_file.read_text(encoding="utf-8"))
    try:
        summary = verify_report(
            report,
            suite=args.suite,
            model_key=args.model_key,
            min_total=args.min_total,
            min_schema_valid_rate=args.min_schema_valid_rate,
            min_action_accuracy=args.min_action_accuracy,
            min_expected_contains_accuracy=args.min_expected_contains_accuracy,
            max_premature_response_rate=args.max_premature_response_rate,
        )
    except VerificationError as exc:
        raise SystemExit(f"Verification failed:\n{exc}") from None
    print(
        "Verification passed: "
        f"{summary['suite']}.{summary['model_key']} "
        f"total={summary['total']} "
        f"schema_valid_rate={summary['schema_valid_rate']:.3f} "
        f"action_accuracy={summary['action_accuracy']:.3f} "
        f"expected_contains_accuracy={summary['expected_contains_accuracy']:.3f} "
        f"premature_response_rate={summary['premature_response_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
