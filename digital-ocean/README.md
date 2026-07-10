# Synthetic Teacher Data

This directory generates synthetic TIM training candidates with Qwen3.5 through DigitalOcean Serverless Inference. The generator validates each response, balances action labels, supports resumable runs, and packages approved rows as deterministic Hugging Face dataset splits.

## Requirements

- Python 3.10 or newer
- an authenticated `doctl` installation with Serverless Inference access
- an authenticated Hugging Face CLI and existing private dataset repository only when uploading the packaged dataset

No credentials are stored by these scripts.

## Generate and validate

```bash
make generate COUNT=100
make validate COUNT=100
make quality COUNT=100
```

Generated files are written under `data/raw/qwen35/` from the project root and are ignored by Git.

Execution provenance is retained separately from raw model output. See [`traces/serverless-smoke-manifest.json`](traces/serverless-smoke-manifest.json) and the project [experiment results](../docs/experiment-results.md).

To emphasize difficult boundary cases:

```bash
make generate COUNT=100 FAILURE_FOCUSED=1
```

## Package and upload

```bash
make package COUNT=100
make verify-package
make upload HF_DATASET_REPO=ORGANIZATION/DATASET
```

`make upload` verifies the hashes in the existing package and uses the official Hugging Face CLI. It does not regenerate the dataset or require raw endpoint responses. Set `HF_CLI` when the executable is not named `hf`. Dataset source files and packages are retained after upload; the `clean` target removes caches only.

Only validated rows are packaged. Raw responses, rejected rows, credentials, and the DoubleTextBench holdout are excluded.

## Local validation smoke test

```bash
make sample
```

Expected result: four accepted examples, one for each action.
