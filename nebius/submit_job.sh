#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 IMAGE DATASET_REPO DATASET_REVISION ADAPTER_REPO HF_SECRET SUBNET_ID" >&2
  exit 2
fi

IMAGE="$1"
DATASET_REPO="$2"
DATASET_REVISION="$3"
ADAPTER_REPO="$4"
HF_SECRET="$5"
SUBNET_ID="$6"

nebius ai job create \
  --name "${JOB_NAME:-tima-qwen3-lora}" \
  --image "$IMAGE" \
  --working-dir /workspace \
  --env "DATASET_REPO=$DATASET_REPO" \
  --env "DATASET_REVISION=$DATASET_REVISION" \
  --env "ADAPTER_REPO=$ADAPTER_REPO" \
  --env "MAX_STEPS=${MAX_STEPS:-200}" \
  --env-secret "HF_TOKEN=$HF_SECRET" \
  --platform "${PLATFORM:-gpu-l40s-a}" \
  --preset "${PRESET:-1gpu-8vcpu-32gb}" \
  --disk-size "${DISK_SIZE:-450Gi}" \
  --shm-size "${SHM_SIZE:-16Gi}" \
  --timeout "${TIMEOUT:-12h}" \
  --subnet-id "$SUBNET_ID"
