.PHONY: help dev test cov lint typecheck fmt serve clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

test:  ## Run the test suite
	pytest

cov:  ## Run tests with coverage report
	pytest --cov=conduit --cov-report=term-missing --cov-report=xml

lint:  ## Lint with ruff
	ruff check conduit tests

typecheck:  ## Static type-check with mypy
	mypy

fmt:  ## Auto-format / fix with ruff
	ruff check --fix conduit tests
	ruff format conduit tests

serve:  ## Run the gateway locally
	uvicorn conduit.server.app:app --reload --host 0.0.0.0 --port 8080

clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
