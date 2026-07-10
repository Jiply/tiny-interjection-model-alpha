PYTHON ?= python3.11
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MODEL ?= Qwen/Qwen3-4B-Instruct-2507
RUN_DIR ?= runs/qwen3-4b-instruct-2507
ADAPTER_DIR ?= adapters/qwen3-4b-instruct-2507
LLAMA_CLI ?= llama-cli
LLAMA_MODEL ?= runs/local-llama/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
LLAMA_ADAPTER ?= runs/local-llama/tim-lora-f16.gguf

.PHONY: setup data data-public train eval-base eval-adapter verify serve cli cli-heuristic cli-llama test clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

data:
	PYTHONPATH=src $(PYTHON) -m interaction_models.data --synthetic-count 2000 --output-dir data/processed

data-public:
	$(PY) -m interaction_models.data --synthetic-count 2000 --include-public --output-dir data/processed

train:
	$(PY) -m interaction_models.train --model-name $(MODEL) --train-file data/processed/train.jsonl --output-dir $(RUN_DIR) --adapter-dir $(ADAPTER_DIR)

eval-base:
	$(PY) -m interaction_models.eval --model-name $(MODEL) --adapter-dir $(ADAPTER_DIR) --base-only --report-file $(RUN_DIR)/base-eval.json

eval-adapter:
	$(PY) -m interaction_models.eval --model-name $(MODEL) --adapter-dir $(ADAPTER_DIR) --report-file $(RUN_DIR)/eval.json

verify:
	$(PY) -m interaction_models.verify --report-file $(RUN_DIR)/eval.json --min-total 40 --min-schema-valid-rate 1.0 --min-action-accuracy 0.95 --min-expected-contains-accuracy 0.95 --max-premature-response-rate 0.0

serve:
	$(PY) -m interaction_models.server --model-name $(MODEL) --adapter-dir $(ADAPTER_DIR)

cli:
	$(PY) -m interaction_models.cli --model-name $(MODEL) --adapter-dir $(ADAPTER_DIR)

cli-heuristic:
	PYTHONPATH=src $(PYTHON) -m interaction_models.cli --heuristic

cli-llama:
	PYTHONPATH=src $(PYTHON) -m interaction_models.cli --llama-cli-path "$(LLAMA_CLI)" --llama-model-path "$(LLAMA_MODEL)" --llama-adapter-path "$(LLAMA_ADAPTER)"

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
