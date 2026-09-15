VENV := .venv

.PHONY: venv install lint typecheck test coverage build run-local clean format ci

venv:
	python3 -m venv $(VENV)

install:
	$(VENV)/bin/pip install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/ruff format src/ tests/

typecheck:
	$(VENV)/bin/mypy src/

test:
	$(VENV)/bin/pytest -q

coverage:
	$(VENV)/bin/pytest --cov=gate --cov-fail-under=80

ci:
	bash scripts/local-ci.sh

build:
	docker build -t vllm-gate .

run-local:
	$(VENV)/bin/uvicorn gate.app:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build *.egg-info src/*.egg-info
