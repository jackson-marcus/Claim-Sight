.PHONY: install lint format test api ui mlflow docker-up docker-down

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest --cov

api:
	uv run uvicorn claimsight.api.main:app --reload --port 8250

ui:
	CLAIMSIGHT_API_URL=http://localhost:8250 uv run streamlit run src/claimsight/ui/app.py --server.port 8751

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5026

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
