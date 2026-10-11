# One-command entry points. `make all` sets up the venv, lints, tests, runs the eval and a demo.
PY      ?= python3.11
VENV    := .venv
BIN     := $(VENV)/bin
DEMO_DB := ./.qdrant-demo

.PHONY: all install lint test eval demo clean

all: install lint test eval demo

$(BIN)/python:
	$(PY) -m venv $(VENV)

install: $(BIN)/python
	@# uv-created venvs ship without pip; bootstrap it offline if missing.
	@$(BIN)/python -m pip --version >/dev/null 2>&1 || $(BIN)/python -m ensurepip >/dev/null
	$(BIN)/python -m pip install -q -e ".[dev]"
	@test -f .env || cp .env.example .env

lint:
	$(BIN)/python -m ruff check .

test:
	$(BIN)/python -m pytest -q

eval:
	$(BIN)/python -m guardrail_rag.eval.run

demo:
	rm -rf $(DEMO_DB)
	GR_QDRANT_PATH=$(DEMO_DB) $(BIN)/python -m guardrail_rag.cli ingest --tenant acme --path tests/data/cli_docs
	GR_QDRANT_PATH=$(DEMO_DB) $(BIN)/python -m guardrail_rag.cli ask --tenant acme \
		"How many days of paid annual leave does Acme offer?"
	rm -rf $(DEMO_DB)

clean:
	rm -rf $(DEMO_DB) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
