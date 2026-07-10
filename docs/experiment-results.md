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

Raw endpoint responses and rejected rows are intentionally excluded because they are not required to reproduce the pipeline and may contain unreviewed model output. The 702 approved rows are retained locally and at the immutable public Hugging Face revision.

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

The focused 702-example run improved expected-content accuracy to `0.6875`. It did not meet the required `0.9500` threshold, so the quality gate correctly stopped publication of a merged model.

These results are negative but important: the pipeline worked, while the model remained below the release threshold. Future Serverless AI Job runs must preserve their job ID, timestamps, logs, dataset revision, image digest, evaluation reports, and billed runtime before updating this record.

The recovered step-300 PEFT adapter, converted F16 GGUF adapter, evaluation reports, and training log are retained outside Git. Their hashes are recorded in [`docs/artifact-manifest.json`](artifact-manifest.json).
