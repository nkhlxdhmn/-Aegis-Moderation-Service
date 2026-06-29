.PHONY: install test run docker lint format benchmark

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install ruff black mypy pytest pre-commit

test:
	python -m pytest

run:
	python -m uvicorn main:app --host 0.0.0.0 --port 8000

docker:
	docker build -t aegis-moderation:latest .

lint:
	ruff check .
	black --check .
	mypy .

format:
	ruff check . --fix
	black .

benchmark:
	python scripts/benchmark.py --image examples/image.jpg
