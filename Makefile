# Makefile for Sloposcope AI Text Analysis Project

.PHONY: help install install-dev test test-unit lint format type-check pre-commit clean docker-build docker-run docker-clean dev-setup ci deploy-fly deploy-cf

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and Setup
install: ## Install the package
	python3 -m pip install -e .
	python3 -m spacy download en_core_web_sm

install-dev: ## Install development dependencies
	python3 -m pip install -e ".[dev]"
	python3 -m spacy download en_core_web_sm

sync: ## Sync dependencies with uv (if using uv)
	@if command -v uv >/dev/null 2>&1; then \
		uv sync; \
	else \
		echo "uv not found, using pip instead"; \
		python3 -m pip install -e .; \
	fi

sync-dev: ## Sync development dependencies with uv (if using uv)
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --dev; \
	else \
		echo "uv not found, using pip instead"; \
		make install-dev; \
	fi

# Testing
test: ## Run all tests
	python3 -m pytest tests/ -v

test-unit: ## Run unit tests only (fast)
	python3 -m pytest tests/test_cli.py -v --tb=short

# Code Quality
lint: ## Run linting
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check sloplint/ tests/ scripts/; \
	else \
		echo "ruff not found, install with: pip install ruff"; \
	fi
	@if command -v black >/dev/null 2>&1; then \
		black --check sloplint/ tests/ scripts/; \
	else \
		echo "black not found, install with: pip install black"; \
	fi

format: ## Format code
	@if command -v black >/dev/null 2>&1; then \
		black sloplint/ tests/ scripts/; \
	else \
		echo "black not found, install with: pip install black"; \
	fi
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check --fix sloplint/ tests/ scripts/; \
	else \
		echo "ruff not found, install with: pip install ruff"; \
	fi

type-check: ## Run type checking
	@if command -v mypy >/dev/null 2>&1; then \
		mypy sloplint/; \
	else \
		echo "mypy not found, install with: pip install mypy"; \
	fi

pre-commit: ## Install pre-commit hooks
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
	else \
		echo "pre-commit not found, install with: pip install pre-commit"; \
	fi

pre-commit-run: ## Run pre-commit hooks on all files
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit run --all-files; \
	else \
		echo "pre-commit not found, install with: pip install pre-commit"; \
	fi

# Cleanup
clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

# Docker commands
docker-build: ## Build Docker image
	docker build -t sloposcope .

docker-run: ## Run Docker container
	cd docker && docker-compose up --build

docker-clean: ## Clean up Docker containers and images
	cd docker && docker-compose down
	docker rmi sloposcope 2>/dev/null || true
	docker system prune -f

# Development workflow
dev-setup: install-dev pre-commit ## Set up development environment
	@echo "Development environment set up complete!"
	@echo "Run 'make test' to run all tests"
	@echo "Run 'make docker-run' to start with Docker"

# CI Pipeline
ci: lint type-check test ## Run CI pipeline locally
	@echo "CI pipeline completed successfully!"

# Deployment
deploy-fly: ## Deploy to Fly.io
	@if [ -f "./deploy-fly.sh" ]; then \
		chmod +x ./deploy-fly.sh && ./deploy-fly.sh; \
	else \
		echo "deploy-fly.sh not found"; \
	fi

# Local development server
run: ## Run the development server
	uvicorn app:app --reload --host 0.0.0.0 --port 8000
