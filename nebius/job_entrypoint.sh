#!/usr/bin/env bash
set -euo pipefail

required=(DATASET_REPO DATASET_REVISION ADAPTER_REPO)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required." >&2
    exit 2
  fi
done

exec nebius/train_qwen_lora.sh \
  "$DATASET_REPO" \
  "$DATASET_REVISION" \
  "$ADAPTER_REPO" \
  "${MAX_STEPS:-200}"
