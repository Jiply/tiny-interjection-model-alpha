# Experiment Results

This record preserves the useful provenance and measured results from the first teacher-data and QLoRA experiment without committing raw responses, credentials, private repository names, model weights, or machine-specific paths.

## DigitalOcean Serverless teacher data

- service: DigitalOcean Serverless Inference
- teacher model: `qwen3.5-397b-a17b`
- generation method: JSON chat completions through `doctl`
- validation: schema, action-specific semantics, balance, duplicate rate, prompt diversity, timing buckets, and failure-focused cases
- durable storage: approved rows are published in a public Hugging Face dataset at an immutable revision

The final focused dataset contained 702 approved examples:

| Action      | Count |
| ----------- | ----: |
| `wait`      |   176 |
| `respond`   |   176 |
| `interject` |   175 |
| `continue`  |   175 |

The full immutable dataset manifest is available at [`digital-ocean/traces/serverless-dataset-manifest.json`](../digital-ocean/traces/serverless-dataset-manifest.json). It records the 598-row training split, 104-row validation split, source hash, packaged file hashes, quality metrics, and public Hugging Face revision. The earlier 16-row endpoint smoke manifest is retained separately at [`digital-ocean/traces/serverless-smoke-manifest.json`](../digital-ocean/traces/serverless-smoke-manifest.json).

Raw endpoint responses and rejected rows are intentionally excluded because they are not required to reproduce the pipeline and may contain unreviewed model output. The 702 approved rows used for this run are retained locally and published as the canonical single-root Hugging Face dataset.

## QLoRA training

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- method: completion-only 4-bit QLoRA
- LoRA rank and alpha: `16` and `16`
- training hardware: one NVIDIA L40S GPU
- focused run: 702 examples and 300 steps
- evaluation: unchanged 40-case DoubleTextBench holdout

The 500-example run reached:

- schema validity: `1.000`
- action accuracy: `0.925`
- expected-content accuracy: `0.46875`

The focused 702-example VM run improved expected-content accuracy to `0.6875`. It did not meet the required `0.9500` threshold, so the quality gate correctly stopped publication of a merged model.

The 40 cases are now treated as a fixed development suite, not an unbiased holdout. They were not included in the 598 training rows, but repeated evaluation informed data and training decisions, so performance on them cannot establish generalization by itself. The earlier `1.000` score was also only schema validity, not end-to-end quality.

The recovered step-300 PEFT adapter, converted F16 GGUF adapter, evaluation reports, and training log are retained outside Git. Their hashes are recorded in [`docs/artifact-manifest.json`](artifact-manifest.json).

## Nebius Serverless AI Job

The final challenge proof ran the complete pipeline as a one-off Serverless AI Job:

- job ID: `aijob-e00kxw7qp6kg6z18w2`
- final state: `COMPLETED`
- created: `2026-07-15T06:15:09.442048Z`
- started: `2026-07-15T06:17:44.136195655Z`
- finished: `2026-07-15T06:55:08.362173769Z`
- measured running time: 37 minutes 24.226 seconds
- image: `REGISTRY/tiny-interjection-model-alpha@sha256:284888e80ca9d0e23884c35a10ddd864ebf990f636535d78e3cf9d2244833ee5` (account-scoped registry namespace redacted)
- canonical training dataset recorded by the Nebius job: `jeremysoojk/tiny-interjection-model-alpha@f38db36f56efecb46b84854484afa99287401a41` (superseded after execution and no longer downloadable after repository replacement)
- canonical public dataset: `jeremysoojk/tiny-interjection-model-alpha@7eab2028563f17bae3a66c392d0dd9bbf1fe389f`
- base model: `Qwen/Qwen3-4B-Instruct-2507`
- resources: `gpu-l40s-a`, `1gpu-8vcpu-32gb`, 450 GiB network SSD, 16 GiB shared memory
- training: 598 examples, 200 steps, final train loss `0.6487`
- artifact upload: skipped because this public-data proof run intentionally injected no Hugging Face token
- quality status: failed; no release was promoted

Measured evaluation:

| Suite                         |  Schema |    Action | Expected content | Premature response | Cases |
| ----------------------------- | ------: | --------: | ---------------: | -----------------: | ----: |
| Base model, current dev suite | `0.075` |   `0.200` |          `0.000` |            `0.000` |    40 |
| LoRA, current dev suite       | `1.000` |   `0.975` |        `0.65625` |            `0.000` |    40 |
| LoRA, generated validation    | `1.000` | `0.98077` |    not annotated |          `0.03846` |   104 |

The generated validation aggregate passed its floor-control gate. The release remained blocked because current-dev expected-content accuracy was below `0.9500`. This is evidence against the interpretation that the model achieved perfect end-to-end performance: action selection is strong on two non-training suites, while completion quality is still materially below threshold.

Literal redacted Nebius CLI log lines:

```text
[2026-07-14T23:42:35-07:00] {'train_runtime': '942.3', 'train_samples_per_second': '1.698', 'train_steps_per_second': '0.212', 'train_loss': '0.6487', 'epoch': '2.669'}
[2026-07-14T23:44:56-07:00]       "action_accuracy": 0.975,
[2026-07-14T23:44:56-07:00]       "expected_contains_accuracy": 0.65625,
[2026-07-14T23:44:56-07:00]       "premature_response_rate": 0.0,
[2026-07-14T23:44:56-07:00]       "schema_valid_rate": 1.0,
[2026-07-14T23:44:56-07:00]       "total": 40
[2026-07-14T23:50:17-07:00]       "action_accuracy": 0.9807692307692307,
[2026-07-14T23:50:17-07:00]       "expected_contains_accuracy": 0.0,
[2026-07-14T23:50:17-07:00]       "premature_response_rate": 0.038461538461538464,
[2026-07-14T23:50:17-07:00]       "schema_valid_rate": 1.0,
[2026-07-14T23:50:17-07:00]       "total": 104
[2026-07-14T23:50:18-07:00] ARTIFACT_UPLOAD_STATUS=skipped
[2026-07-14T23:55:08-07:00] llama_model_quantize_impl: quant size  =  2375.91 MiB (4.95 BPW)
[2026-07-14T23:55:08-07:00] QUALITY_GATE_STATUS=failed
```

At the previously recorded planning rate of `$1.55` per running hour, 37m 24s is a compute-only estimate of approximately `$0.97`, plus ephemeral disk. This is not represented as billed cost; the settled Nebius billing export was not yet available immediately after job completion.
