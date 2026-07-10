# Nebius Serverless AI Job

This directory packages the Qwen3 4B QLoRA pipeline as a one-off Nebius Serverless AI Job. Serverless AI provisions the configured GPU resources, runs the container to completion, and releases the underlying compute automatically.

The implementation follows the official Nebius job interface:

- public or authenticated container image
- `gpu-l40s-a` platform
- `1gpu-8vcpu-32gb` preset
- MysteryBox secret injection for `HF_TOKEN`
- immutable Hugging Face dataset revision
- no persistent cloud disk required after successful upload

## Files

- `submit_job.sh`: creates the Serverless AI Job through the Nebius CLI.
- `job_entrypoint.sh`: validates injected configuration and starts the pipeline.
- `train_qwen_lora.sh`: downloads data, trains, evaluates, verifies, merges, quantizes, and uploads.
- `merge_adapter.py`: merges the verified PEFT adapter into the base model.

## Prerequisites

1. Install and authenticate the Nebius CLI.
2. Configure a Nebius project and subnet.
3. Build the project image for `linux/amd64` and push it to a registry accessible by Nebius.
4. Create private Hugging Face dataset and model repositories.
5. Store a fine-grained Hugging Face token in Nebius MysteryBox. The token should read only the source dataset and write only the target model repository.

Do not pass the Hugging Face token through `--env` or commit it to a file.

## Build and publish the image

```bash
docker build --platform linux/amd64 -t REGISTRY/tiny-interjection-model-alpha:VERSION .
docker push REGISTRY/tiny-interjection-model-alpha:VERSION
```

Use an immutable version tag or digest for a reproducible run.

## Submit

```bash
nebius/submit_job.sh \
  REGISTRY/tiny-interjection-model-alpha:VERSION \
  ORGANIZATION/DATASET \
  DATASET_COMMIT_SHA \
  ORGANIZATION/MODEL \
  HF_SECRET_SELECTOR \
  SUBNET_ID
```

Optional environment overrides:

```bash
JOB_NAME=tima-qwen3-lora \
MAX_STEPS=200 \
TIMEOUT=12h \
nebius/submit_job.sh IMAGE DATASET REVISION MODEL HF_SECRET SUBNET_ID
```

The default resource configuration is:

- platform: `gpu-l40s-a`
- preset: `1gpu-8vcpu-32gb`
- container disk: `450Gi`
- shared memory: `16Gi`
- timeout: `12h`

## Monitor

```bash
nebius ai job list
nebius ai job get JOB_ID
nebius ai job logs JOB_ID --follow --timestamps
```

The job must finish with `COMPLETED`. Capture the final status, timestamps, and logs as proof of execution for the challenge submission.

## Expected outputs

The target Hugging Face model repository receives:

- PEFT adapter and tokenizer files
- `eval/base-eval.json`
- `eval/adapter-eval.json`
- `eval/generated-validation-eval.json`
- `tima-q4_k_m.gguf`

Every completed experiment is archived privately under `experiments/`, including its adapter, evaluation reports, and quantized GGUF. A failed quality gate prevents promotion to the model repository root, but does not prevent archival or discard the executable experiment artifacts.

## Runtime and cost

Plan for approximately 1–2 hours for the default 200-step run, including downloads, two evaluation suites, training, merge, and quantization. At pricing checked on July 10, 2026, the configured preset is approximately $1.55 per running hour before ephemeral disk charges, or roughly $1.55–$3.10 for that planning range.

These are estimates. Replace them with the measured Serverless AI job duration and billed cost before submission.

## Cleanup

Successful jobs release their compute automatically. After saving proof of execution, delete completed job metadata if it is no longer useful:

```bash
nebius ai job delete JOB_ID
```
