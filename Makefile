# Makefile for Sloposcope AI Text Analysis Project

.PHONY: help install test lint format clean run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Main Setup
install: ## Install dependencies and models (one command setup)
	uv sync --dev
	uv run python -m spacy download en_core_web_sm
	@echo "Verifying model installation..."
	uv run python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy model verified and working')"
	@echo "Setting up pre-commit hooks..."
	uv run pre-commit install --install-hooks
	@echo "✅ Setup complete! Ready to use."

# Development
test: ## Run tests
	TRANSFORMERS_VERBOSITY=error uv run pytest tests/ -v

lint: ## Run linting
	uv run ruff check sloplint/ tests/ && uv run black --check sloplint/ tests/

format: ## Format code
	uv run black sloplint/ tests/ && uv run ruff check --fix sloplint/ tests/

setup-hooks: ## Set up pre-commit hooks
	uv run pre-commit install --install-hooks

# Utilities
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/ htmlcov/ .coverage dist/ build/

run: ## Run the development server
	uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
