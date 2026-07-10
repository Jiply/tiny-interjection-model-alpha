#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 DATASET_REPO DATASET_REVISION ADAPTER_REPO [MAX_STEPS]" >&2
  exit 2
fi

DATASET_REPO="$1"
DATASET_REVISION="$2"
ADAPTER_REPO="$3"
MAX_STEPS="${4:-200}"
DATASET_DIR="data/processed/qwen35-hf"
RUN_DIR="runs/nebius-qwen3-4b-instruct-2507"
ADAPTER_DIR="adapters/nebius-qwen3-4b-instruct-2507"
MODEL_DIR="$RUN_DIR/base-model"
MERGED_DIR="$RUN_DIR/merged"
LLAMA_CPP_DIR="$RUN_DIR/llama.cpp"
GGUF_F16="$RUN_DIR/tim-f16.gguf"
GGUF_Q4="$RUN_DIR/tim-q4_k_m.gguf"
MODEL_REPO="Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION="cdbee75f17c01a7cc42f958dc650907174af0554"
LLAMA_CPP_REVISION="505b1ed15ca80e2a19f12ff4ac365e40fb374053"
PYTHON="${PYTHON:-python3}"
HF="${HF_CLI:-hf}"

"$HF" download "$DATASET_REPO" --repo-type dataset --revision "$DATASET_REVISION" --local-dir "$DATASET_DIR"
"$HF" download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
make data PYTHON="$PYTHON"
if [[ ! -f "$RUN_DIR/base-eval.json" ]]; then
  "$PYTHON" -m interaction_models.eval --model-name "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" --base-only --report-file "$RUN_DIR/base-eval.json"
fi
"$PYTHON" -m interaction_models.train --model-name "$MODEL_DIR" --train-file "$DATASET_DIR/data/train.jsonl" --output-dir "$RUN_DIR" --adapter-dir "$ADAPTER_DIR" --max-steps "$MAX_STEPS"
"$PYTHON" -m interaction_models.eval --model-name "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" --report-file "$RUN_DIR/eval.json"
"$PYTHON" -m interaction_models.eval --model-name "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" --bench-file "$DATASET_DIR/data/validation.jsonl" --report-file "$RUN_DIR/generated-validation-eval.json"

VERIFICATION_FAILED=0
if ! "$PYTHON" -m interaction_models.verify --report-file "$RUN_DIR/eval.json" --min-total 40 --min-schema-valid-rate 1.0 --min-action-accuracy 0.95 --min-expected-contains-accuracy 0.95 --max-premature-response-rate 0.0; then
  VERIFICATION_FAILED=1
fi
if ! "$PYTHON" -m interaction_models.verify --report-file "$RUN_DIR/generated-validation-eval.json" --min-total 60 --min-schema-valid-rate 1.0 --min-action-accuracy 0.95 --min-expected-contains-accuracy 0.0 --max-premature-response-rate 0.05; then
  VERIFICATION_FAILED=1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  "$HF" upload "$ADAPTER_REPO" "$ADAPTER_DIR" experiments/adapter --repo-type model --commit-message "archive tim qwen3 4b lora experiment" --commit-description "dataset: $DATASET_REPO@$DATASET_REVISION"
  "$HF" upload "$ADAPTER_REPO" "$RUN_DIR/base-eval.json" experiments/eval/base-eval.json --repo-type model --commit-message "archive base evaluation"
  "$HF" upload "$ADAPTER_REPO" "$RUN_DIR/eval.json" experiments/eval/adapter-eval.json --repo-type model --commit-message "archive adapter evaluation"
  "$HF" upload "$ADAPTER_REPO" "$RUN_DIR/generated-validation-eval.json" experiments/eval/generated-validation-eval.json --repo-type model --commit-message "archive generated validation evaluation"
else
  echo "ARTIFACT_UPLOAD_STATUS=skipped"
fi

"$PYTHON" nebius/merge_adapter.py --base-model "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" --output-dir "$MERGED_DIR"
if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi
git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_REVISION"
git -C "$LLAMA_CPP_DIR" checkout --detach FETCH_HEAD
cmake -B "$LLAMA_CPP_DIR/build" -S "$LLAMA_CPP_DIR"
cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-quantize -j
"$PYTHON" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "$GGUF_F16" --outtype f16
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_F16" "$GGUF_Q4" Q4_K_M
if [[ -n "${HF_TOKEN:-}" ]]; then
  "$HF" upload "$ADAPTER_REPO" "$GGUF_Q4" experiments/tim-q4_k_m.gguf --repo-type model --commit-message "archive quantized gguf experiment"
fi

if [[ "$VERIFICATION_FAILED" -ne 0 ]]; then
  echo "QUALITY_GATE_STATUS=failed"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Experiment completed and artifacts were archived, but nothing was promoted as a release." >&2
  else
    echo "Experiment completed without artifact upload, and nothing was promoted as a release." >&2
  fi
  exit 0
fi

echo "QUALITY_GATE_STATUS=passed"
if [[ -n "${HF_TOKEN:-}" ]]; then
  "$HF" upload "$ADAPTER_REPO" "$ADAPTER_DIR" . --repo-type model --commit-message "publish verified tim qwen3 4b lora adapter" --commit-description "dataset: $DATASET_REPO@$DATASET_REVISION"
  "$HF" upload "$ADAPTER_REPO" "$RUN_DIR/base-eval.json" eval/base-eval.json --repo-type model --commit-message "publish base evaluation"
  "$HF" upload "$ADAPTER_REPO" "$RUN_DIR/eval.json" eval/adapter-eval.json --repo-type model --commit-message "publish adapter evaluation"
  "$HF" upload "$ADAPTER_REPO" "$RUN_DIR/generated-validation-eval.json" eval/generated-validation-eval.json --repo-type model --commit-message "publish generated validation evaluation"
  "$HF" upload "$ADAPTER_REPO" "$GGUF_Q4" tim-q4_k_m.gguf --repo-type model --commit-message "publish verified quantized gguf"
fi
