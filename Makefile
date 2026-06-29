.PHONY: install test run docker docker-gpu lint format benchmark

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install ruff black mypy pytest pre-commit

test:
	python -m pytest

run:
	python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

docker-gpu:
	docker compose -f docker-compose.gpu.yml up --build

lint:
	ruff check .
	black --check .
	mypy .

format:
	ruff check . --fix
	black .

benchmark:
	python scripts/benchmark.py --image examples/image.jpg
