from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from candidate_utils import (
    add_training_text,
    extract_json_object,
    fingerprint,
    normalize_example,
    quality_errors,
    validation_errors,
)
from quality_candidates import quality_report


def sha256(path: Path) -> str:
    """When a dataset artifact needs provenance, calculate its SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    """When validated candidates are packaged, load and verify every source row."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = extract_json_object(line)
            example = normalize_example(
                raw,
                source=str(raw.get("source") or "do-serverless-qwen35"),
                case_id=str(raw.get("case_id") or f"qwen35-{line_number:06d}"),
            )
            errors = validation_errors(example)
            errors.extend(quality_errors(example))
            if errors:
                raise ValueError(f"Invalid row {line_number}: {', '.join(errors)}")
            example = add_training_text(example)
            example["case_id"] = f"qwen35-{fingerprint(example)[:20]}"
            rows.append(example)
    if not rows:
        raise ValueError(f"No candidates found in {path}")
    return rows


def split_rows(
    rows: list[dict[str, Any]], validation_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """When a dataset is published, create stable action-stratified splits."""
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        action = str(row["target"]["action"])
        by_action.setdefault(action, []).append(row)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for action_rows in by_action.values():
        ordered = sorted(action_rows, key=fingerprint)
        validation_count = (
            max(1, int(len(ordered) * validation_ratio)) if len(ordered) > 1 else 0
        )
        validation.extend(ordered[:validation_count])
        train.extend(ordered[validation_count:])
    return sorted(train, key=fingerprint), sorted(validation, key=fingerprint)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """When rows are ready for Hugging Face, write deterministic JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    """When a manifest is built, count a named categorical field."""
    values: Counter[str] = Counter()
    for row in rows:
        if field == "action":
            values[str(row["target"]["action"])] += 1
        elif isinstance(row.get(field), str):
            values[str(row[field])] += 1
    return dict(sorted(values.items()))


def dataset_card(teacher_model: str, train_count: int, validation_count: int) -> str:
    """When a package is created, describe its provenance and safe initial use."""
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
language:
- en
license: mit
task_categories:
- text-generation
pretty_name: Tiny Interjection Model Alpha Data
---

# Tiny Interjection Model Alpha Data

Synthetic typed-chat interaction examples generated with `{teacher_model}` through DigitalOcean Serverless Inference.

The dataset contains {train_count} training rows and {validation_count} validation rows. Each row includes an event stream with millisecond delays, a floor-control action (`wait`, `respond`, `interject`, or `continue`), and optional assistant messages.

## Intended use

Training and evaluating small interaction models that decide when to wait, respond, interject, or continue in typed chat.

## Data policy

The data is synthetic and contains no private conversations or personal data. Only validated rows are published. Raw model responses, rejected rows, credentials, and the DoubleTextBench holdout are excluded.

## Limitations

Synthetic data can contain unrealistic dialogue, model biases, or timing artifacts. Review samples before training or deploying a model trained on this dataset.
"""


def package_dataset(
    input_path: Path,
    output_dir: Path,
    teacher_model: str,
    validation_ratio: float,
) -> dict[str, Any]:
    """When candidates pass quality checks, package them for a private Hub repo."""
    rows = load_rows(input_path)
    train, validation = split_rows(rows, validation_ratio)
    train_path = output_dir / "data" / "train.jsonl"
    validation_path = output_dir / "data" / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_model": teacher_model,
        "source_sha256": sha256(input_path),
        "validation_ratio": validation_ratio,
        "train": len(train),
        "validation": len(validation),
        "action_counts": counts(rows, "action"),
        "timing_counts": counts(rows, "timing_bucket"),
        "quality": quality_report(input_path),
        "files": {
            "data/train.jsonl": sha256(train_path),
            "data/validation.jsonl": sha256(validation_path),
        },
    }
    metadata_path = output_dir / "metadata" / "manifest.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        dataset_card(teacher_model, len(train), len(validation)), encoding="utf-8"
    )
    return manifest


def verify_package(output_dir: Path) -> dict[str, Any]:
    """When a packaged dataset is published, verify every recorded file hash."""
    manifest_path = output_dir / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"No file hashes found in {manifest_path}")
    for relative_path, expected in files.items():
        path = output_dir / relative_path
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"Hash mismatch for {relative_path}: expected {expected}, got {actual}"
            )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package validated TIMA data for Hugging Face."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default="qwen3.5-397b-a17b")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_package(args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.input is None:
        parser.error("--input is required unless --verify-only is set")
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")
    manifest = package_dataset(
        args.input,
        args.output_dir,
        args.teacher_model,
        args.validation_ratio,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
