# Tiny Interjection Model Alpha

Tiny Interjection Model Alpha (TIMA) explores a small floor-control model for typed chat. Given a timestamped event stream, the model decides whether an assistant should wait, respond, interject, or continue.

```text
<act>wait|respond|interject|continue</act>
<msg>optional assistant message</msg>
<done/>
```

The repository contains a reproducible data pipeline, Qwen3 4B QLoRA training and evaluation code, a FastAPI decision endpoint, and a local `llama.cpp` runtime.

## Challenge status

This project is being prepared for the Nebius Serverless AI Builders Challenge. It packages QLoRA training as a Nebius Serverless AI Job and includes a secret-safe submission wrapper. The serverless job still needs to be executed and its measured results recorded before submission.

The challenge also requires a public repository, proof of execution, and a technical blog post. Those submission materials should be added only after the serverless run is complete.

## Architecture

1. Generate synthetic typed-chat candidates with Qwen3.5 through DigitalOcean Serverless Inference.
2. Validate schema, semantics, balance, duplicates, and prompt diversity locally.
3. Package approved rows as deterministic Hugging Face dataset splits.
4. Fine-tune `Qwen/Qwen3-4B-Instruct-2507` with QLoRA in a Nebius Serverless AI Job.
5. Evaluate against the fixed 40-case DoubleTextBench holdout and block publication unless the quality gate passes.
6. Merge and quantize every completed experiment for archival; promote only adapters that pass the release gate.

## Nebius training evidence

The first QLoRA experiment ran on a Nebius L40S virtual machine. The redacted screenshots below preserve the compute configuration and utilization without exposing resource names or identifiers. A Nebius Serverless AI Job run remains part of the challenge work in [TODO.md](TODO.md).

![Redacted Nebius L40S virtual machine configuration](assets/nebius-virtual-machine.png)

![Redacted Nebius compute utilization](assets/nebius-compute-utilisation.png)

Raw responses, rejected candidates, datasets, checkpoints, adapters, reports, and model binaries are generated artifacts and are intentionally excluded from Git.

Measured provenance and non-sensitive result summaries are retained in [docs/experiment-results.md](docs/experiment-results.md), including the DigitalOcean Serverless Inference teacher run and its smoke manifest.

Ignored does not mean disposable. Training datasets, adapters, evaluation reports, and GGUF executables are retained until their Hugging Face uploads and hashes are verified. `make clean` removes caches only and never removes anything under `data/`, `adapters/`, or `runs/`.

The retained model artifact hashes and pinned public base source are recorded in [docs/artifact-manifest.json](docs/artifact-manifest.json). A filename ending in `.partial` is never accepted as an executable model.

## Local setup

Python 3.11 is recommended.

```bash
make setup
make test
```

Generate the deterministic offline dataset:

```bash
make data
```

This writes training, validation, and DoubleTextBench JSONL files under `data/processed/`.

## Teacher data

The validated 702-row synthetic dataset is published through Hugging Face. Use its public repository ID and immutable revision to reproduce a run.

Download published train and validation splits at a pinned revision:

```bash
hf download \
  ORGANIZATION/DATASET \
  --repo-type dataset \
  --revision DATASET_COMMIT_SHA \
  --local-dir data/processed/qwen35-hf
```

The hosted teacher-generation path requires an authenticated `doctl` installation. It creates a new synthetic dataset and does not read private conversations or personal data.

```bash
cd digital-ocean
make generate COUNT=100
make package COUNT=100
make verify-package
make upload HF_DATASET_REPO=ORGANIZATION/DATASET
```

The upload target uses the official Hugging Face CLI. Set `HF_CLI` only when the executable is not named `hf`.

For a newly generated dataset, review the card and samples before making its repository public:

```bash
hf repos settings ORGANIZATION/DATASET --public
```

See [digital-ocean/README.md](digital-ocean/README.md) for details.

## Training and evaluation

The generic CUDA path is:

```bash
make data
make eval-base
make train
make eval-adapter
make verify
```

Defaults:

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- quantization: 4-bit NF4
- LoRA rank and alpha: `16`
- maximum sequence length: `4096`
- quality gate: 100% schema validity, 95% action accuracy, 95% expected-content accuracy, and no premature response on holdout `wait` cases

## Nebius Serverless AI Job

Build the Linux GPU image and publish it to a registry accessible by Nebius:

```bash
docker build --platform linux/amd64 -t REGISTRY/tiny-interjection-model-alpha:VERSION .
docker push REGISTRY/tiny-interjection-model-alpha:VERSION
```

Store the Hugging Face token in Nebius MysteryBox, then submit the one-off job:

```bash
nebius/submit_job.sh \
  REGISTRY/tiny-interjection-model-alpha:VERSION \
  ORGANIZATION/DATASET \
  DATASET_COMMIT_SHA \
  ORGANIZATION/MODEL \
  HF_SECRET_SELECTOR \
  SUBNET_ID
```

The submission uses the `gpu-l40s-a` platform with the `1gpu-8vcpu-32gb` preset, a 450 GiB ephemeral disk, 16 GiB shared memory, and a 12-hour timeout. The container downloads an immutable dataset revision, evaluates the base model, trains and evaluates the adapter, applies both quality gates, creates a Q4 GGUF, and privately archives the experiment. Only a passing adapter is promoted to the model repository root.

Expected durable outputs in the target Hugging Face model repository are the PEFT adapter, base evaluation, adapter evaluation, generated-validation evaluation, and quantized GGUF. Job status and logs remain available through the Nebius CLI.

For planning, allow approximately 1–2 hours for a 200-step run, including model downloads, evaluation, training, merge, and quantization. The configured L40S preset costs about $1.55 per running hour before ephemeral disk charges, or roughly $1.55–$3.10 for that range at prices checked on July 10, 2026. Replace these estimates with measured job duration and billed cost before submission.

See [nebius/README.md](nebius/README.md) for setup, monitoring, and security details.

## Demo

Run the explicit heuristic smoke demo without model weights:

```bash
make cli-heuristic
```

Run a trained GGUF base and LoRA adapter with `llama.cpp`:

```bash
make cli-llama \
  LLAMA_CLI=/path/to/llama-cli \
  LLAMA_MODEL=/path/to/base.gguf \
  LLAMA_ADAPTER=/path/to/adapter.gguf
```

To download the pinned public base used with the retained adapter:

```bash
curl -L --fail --retry 5 \
  --output runs/local-llama/Qwen3-4B-Instruct-2507-Q4_K_M.gguf.partial \
  https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/resolve/4edb920b6f14e3b9284d4502a6485103d72cde05/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

Keep incomplete downloads suffixed with `.partial` until their full size and hash have been verified, then rename the verified file to remove that suffix.
The expected SHA-256 is `8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc`.

Start the FastAPI service after placing a trained PEFT adapter in `adapters/qwen3-4b-instruct-2507`:

```bash
make serve
```

Then request a decision:

```bash
curl -s http://127.0.0.1:8000/decide \
  -H 'content-type: application/json' \
  -d '{"events":[{"role":"user","text":"can you help write this","dt_ms":0},{"role":"user","text":"make it warmer actually","dt_ms":1300}]}'
```

## Data and security

- Only synthetic or public data belongs in the pipeline.
- Never commit credentials, private data, raw model responses, rejected rows, model weights, or unreviewed cloud logs.
- Hugging Face tokens must be provided at runtime and scoped to the required dataset or model repository.
- Review generated samples before publishing a dataset or changing repository visibility.

## License

MIT. See [LICENSE](LICENSE).
