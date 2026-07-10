PYTHON ?= python3
DOCTL_BIN ?= doctl
MODEL ?= qwen3.5-397b-a17b
COUNT ?= 100
NUM_PREDICT ?= 512
RETRIES ?= 3
TIMEOUT ?= 120
FAILURE_FOCUSED ?= 0
DATA_DIR ?= ../data/raw/qwen35
PACKAGE_DIR ?= ../data/processed/qwen35-hf
HF_DATASET_REPO ?=
HF_CLI ?= hf

.PHONY: generate validate quality package verify-package upload sample clean

generate:
	$(PYTHON) generate_candidates.py --doctl-bin "$(DOCTL_BIN)" --model "$(MODEL)" --count "$(COUNT)" --num-predict "$(NUM_PREDICT)" --retries "$(RETRIES)" --timeout "$(TIMEOUT)" --balanced-timing --resume $(if $(filter 1 true yes,$(FAILURE_FOCUSED)),--failure-focused,) --output "$(DATA_DIR)/candidates.raw.jsonl" --rejects "$(DATA_DIR)/candidates.rejects.jsonl"

validate:
	$(PYTHON) validate_candidates.py --input "$(DATA_DIR)/candidates.raw.jsonl" --output "$(DATA_DIR)/candidates.valid.jsonl" --rejects "$(DATA_DIR)/candidates.invalid.jsonl" --min-per-action 1

quality:
	$(PYTHON) quality_candidates.py --input "$(DATA_DIR)/candidates.valid.jsonl" --min-count "$(COUNT)" --min-per-action 1

package: validate quality
	$(PYTHON) package_hf_dataset.py --input "$(DATA_DIR)/candidates.valid.jsonl" --output-dir "$(PACKAGE_DIR)" --teacher-model "$(MODEL)"

verify-package:
	$(PYTHON) package_hf_dataset.py --output-dir "$(PACKAGE_DIR)" --verify-only

upload: verify-package
	@test -n "$(HF_DATASET_REPO)" || (echo "Set HF_DATASET_REPO to the private Hugging Face dataset repository." >&2; exit 2)
	@command -v "$(HF_CLI)" >/dev/null || (echo "Hugging Face CLI not found. Install it or set HF_CLI explicitly." >&2; exit 2)
	$(HF_CLI) upload "$(HF_DATASET_REPO)" "$(PACKAGE_DIR)" . --repo-type dataset --exclude ".cache/**" --commit-message "publish validated interaction dataset"

sample:
	$(PYTHON) validate_candidates.py --input samples/valid_candidates.jsonl --output /tmp/tim-sample.valid.jsonl --rejects /tmp/tim-sample.invalid.jsonl
	$(PYTHON) quality_candidates.py --input /tmp/tim-sample.valid.jsonl --min-count 4 --min-per-action 1

clean:
	rm -rf __pycache__ .pytest_cache "$(PACKAGE_DIR)/.cache"
