.PHONY: setup test lint firewall case-study judge eval ablate placebo api web clean

PYTHON ?= .venv/bin/python
UVICORN ?= .venv/bin/uvicorn
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy

setup:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev,ai,report]"

test:
	$(PYTEST) tests/unit tests/firewall tests/smoke -v

lint:
	$(RUFF) check .
	$(MYPY) trishul

firewall:
	$(PYTEST) tests/firewall -v

case-study:
	$(PYTHON) -m trishul.pipeline \
		--case-study \
		--output runs/trishuli_aug2026

judge:
	@if [ -z "$(AOI)" ] || [ -z "$(DATE)" ]; then \
		echo "Usage: make judge AOI='[[[lon,lat],...]]' DATE=YYYY-MM-DD"; exit 1; \
	fi
	$(PYTHON) -m trishul.pipeline --aoi '$(AOI)' --event-date $(DATE) --output runs/judge_run

eval:
	$(PYTHON) -m eval.run_evaluation

ablate:
	$(PYTHON) -m eval.ablation_ladder

placebo:
	$(PYTHON) -m eval.placebo_test

api:
	$(UVICORN) trishul.api.app:app --host 0.0.0.0 --port 8000 --reload

web:
	cd web && npm install && npm run dev

docs:
	@echo "Documentation generated in docs/ and README.md"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
