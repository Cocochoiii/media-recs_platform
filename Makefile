.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down train serve

PYTHON := python3
PIP := pip3

help:
	@echo "Media Recommender System - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev-install    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make lint           Run linting"
	@echo "  make format         Format code with black"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo ""
	@echo "ML:"
	@echo "  make train          Train models"
	@echo "  make serve          Start API server"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Clean build artifacts"

# Setup
install:
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

dev-install: install
	$(PIP) install -e ".[dev]"

# Testing
test:
	PYTHONPATH=. pytest tests/ -v

test-cov:
	pip install pytest-cov
	PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	PYTHONPATH=. pytest tests/ -v -m "not slow"

# Code Quality
lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

check: lint test

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ML Training
train:
	PYTHONPATH=. $(PYTHON) -m src.training.train_collaborative --epochs 10

train-seq:
	PYTHONPATH=. $(PYTHON) -m src.training.train_sequential --model lstm --epochs 10

train-all:
	PYTHONPATH=. $(PYTHON) scripts/train_all.py --epochs 10

demo:
	PYTHONPATH=. $(PYTHON) scripts/demo.py

# API
serve:
	PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	PYTHONPATH=. gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Database
db-init:
	psql -h localhost -U postgres -d recommender -f docker/init.sql

db-migrate:
	alembic upgrade head

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage

clean-all: clean
	rm -rf checkpoints/ mlruns/ wandb/

# Documentation
docs:
	cd docs && make html

# Deployment
deploy-staging:
	$(PYTHON) scripts/deploy_sagemaker.py --action deploy --env staging

deploy-prod:
	$(PYTHON) scripts/deploy_sagemaker.py --action deploy --env production
