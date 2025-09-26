# Makefile for AI Slop project

.PHONY: help install install-dev test test-unit test-aws test-localstack clean-localstack lint format type-check pre-commit

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run all tests
	python scripts/run_tests_with_localstack.py

test-unit: ## Run unit tests only (fast)
	python -m pytest tests/test_cli.py -v --tb=short

test-aws: ## Run AWS Worker tests with LocalStack
	python scripts/run_tests_with_localstack.py tests/test_aws_worker.py -v --tb=short

test-localstack: ## Start LocalStack for testing
	python scripts/start_localstack_for_tests.py start

clean-localstack: ## Stop and remove LocalStack container
	python scripts/start_localstack_for_tests.py stop

lint: ## Run linting
	ruff check sloplint/ tests/ scripts/
	black --check sloplint/ tests/ scripts/

format: ## Format code
	black sloplint/ tests/ scripts/
	ruff check --fix sloplint/ tests/ scripts/

type-check: ## Run type checking
	mypy sloplint/

pre-commit: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Docker commands
docker-build: ## Build Docker image
	cd docker && docker-compose build sloplint-worker

docker-test: ## Test with Docker and LocalStack
	cd docker && docker-compose up -d localstack
	sleep 10
	python docker/setup_localstack.py
	cd docker && docker-compose up sloplint-worker

docker-clean: ## Clean up Docker containers
	cd docker && docker-compose down
	docker stop localstack-test 2>/dev/null || true
	docker rm localstack-test 2>/dev/null || true

# Development workflow
dev-setup: install-dev pre-commit ## Set up development environment
	@echo "Development environment set up complete!"
	@echo "Run 'make test' to run all tests with LocalStack"

ci: lint type-check test ## Run CI pipeline locally
