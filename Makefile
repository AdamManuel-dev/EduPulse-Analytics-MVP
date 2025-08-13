.PHONY: help install install-dev lint format test clean run docker-up docker-down setup

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	pre-commit install

setup: install-dev ## Complete development setup
	@echo "$(GREEN)Setting up development environment...$(NC)"
	cp .env.example .env 2>/dev/null || true
	mkdir -p data/raw data/processed data/cache logs models
	@echo "$(GREEN)Setup complete! Edit .env file with your configuration.$(NC)"

lint: ## Run all linters
	@echo "$(YELLOW)Running linters...$(NC)"
	$(BLACK) --check src tests
	$(ISORT) --check-only src tests
	$(FLAKE8) src tests
	$(MYPY) src --ignore-missing-imports
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(BLACK) src tests
	$(ISORT) src tests
	@echo "$(GREEN)Formatting complete!$(NC)"

test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v --cov=src --cov-report=term-missing

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	$(PYTEST) tests/e2e/ -v

coverage: ## Generate coverage report
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "$(GREEN)Cleanup complete!$(NC)"

run: ## Run the API server locally
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-worker: ## Run Celery worker
	celery -A src.tasks.worker worker --loglevel=info

run-beat: ## Run Celery beat scheduler
	celery -A src.tasks.worker beat --loglevel=info

run-flower: ## Run Flower monitoring
	celery -A src.tasks.worker flower

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services with Docker Compose
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Flower: http://localhost:5555"
	@echo "Docs: http://localhost:8000/docs"

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

docker-clean: ## Clean Docker resources
	docker-compose down -v
	docker system prune -f

db-init: ## Initialize database
	$(PYTHON) -m src.db.database init

db-migrate: ## Run database migrations
	alembic upgrade head

db-rollback: ## Rollback database migration
	alembic downgrade -1

train: ## Train the ML model
	$(PYTHON) -m src.training.trainer train --epochs 50

predict: ## Make a test prediction
	$(PYTHON) -m src.services.prediction_service predict --student-id STU001

check-security: ## Run security checks
	bandit -r src -ll
	safety check

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

update-deps: ## Update dependencies
	$(PIP) list --outdated
	@echo "$(YELLOW)Run 'pip install --upgrade <package>' to update specific packages$(NC)"

docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	$(PYTHON) -m mkdocs build 2>/dev/null || echo "MkDocs not installed"

serve-docs: ## Serve documentation locally
	$(PYTHON) -m mkdocs serve 2>/dev/null || echo "MkDocs not installed"

version: ## Show version information
	@echo "Python version:"
	@$(PYTHON) --version
	@echo "\nPackage versions:"
	@$(PIP) show fastapi torch pandas | grep "Version:" || true

all: clean install-dev lint test ## Run all checks

.DEFAULT_GOAL := help