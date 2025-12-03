.PHONY: help install sync clean test test-coverage test-parallel test-unit test-integration test-slow test-config test-data test-models test-training test-utils test-cli test-api lint format typecheck build build-config build-utils build-data build-models build-training build-cli build-api serve-api install-hooks pre-commit pre-commit-fix ci

# Platform detection
ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
	SHELL := cmd.exe
else
	DETECTED_OS := $(shell uname -s)
endif


PY_PACKAGES = "packages/config","packages/utils","packages/data","packages/models","packages/training","packages/cli","apps/api"

help:
	@echo "Detected OS: $(DETECTED_OS)"
	@echo "Alzheimer's MRI CNN - Python/UV Monorepo"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install UV and sync dependencies"
	@echo "  make install-hooks  - Install Git pre-commit hooks"
	@echo "  make sync           - Sync all UV dependencies"
	@echo "  make build          - Build all packages/apps (uv build)"
	@echo "  make test           - Run tests with coverage (parallel)"
	@echo "  make lint           - Lint all code with ruff"
	@echo "  make format         - Format all code with ruff"
	@echo "  make typecheck      - Run pyright across the workspace"
	@echo "  make clean          - Clean build/test artifacts"
	@echo ""
	@echo "Testing commands:"
	@echo "  make test           - Run all tests in parallel (auto CPU detection)"
	@echo "  make test-coverage  - Run tests with full coverage reports"
	@echo "  make test-parallel  - Run tests with custom worker count"
	@echo "  make test-unit      - Run only unit tests"
	@echo "  make test-integration - Run only integration tests"
	@echo "  make test-slow      - Run tests and show slowest cases"
	@echo "  make test-config    - Run config package tests"
	@echo "  make test-data      - Run data package tests"
	@echo "  make test-models    - Run models package tests"
	@echo "  make test-training  - Run training package tests"
	@echo "  make test-utils     - Run utils package tests"
	@echo "  make test-cli       - Run CLI package tests"
	@echo "  make test-api       - Run API app tests"
	@echo ""
	@echo "Package-specific commands:"
	@echo "  make build-config   - Build config package"
	@echo "  make build-utils    - Build utils package"
	@echo "  make build-data     - Build data package"
	@echo "  make build-models   - Build models package"
	@echo "  make build-training - Build training package"
	@echo "  make build-cli      - Build CLI package"
	@echo "  make build-api      - Build API application"
	@echo ""
	@echo "Application commands:"
	@echo "  make serve-api      - Start API server"
	@echo ""
	@echo "Git hooks:"
	@echo "  Pre-commit checks run automatically on each commit"
	@echo "  Run 'make install-hooks' to install/reinstall hooks"

# Install UV and dependencies
install:
	@echo "Checking UV installation..."
	@if [ "$(DETECTED_OS)" = "Windows" ]; then \
		powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install_uv.ps1; \
	else \
		bash scripts/install_uv.sh; \
	fi
	@echo "Syncing UV workspace (including dev deps)..."
	@uv sync --all-groups
	@echo "Installing Git hooks..."
	@if [ "$(DETECTED_OS)" = "Windows" ]; then \
		powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install_hooks.ps1; \
	else \
		bash scripts/install_hooks.sh; \
	fi

# Sync UV dependencies across workspace
sync:
	@echo "Syncing workspace dependencies..."
	uv sync --all-groups

# Build all packages/apps
build: build-config build-utils build-data build-models build-training build-cli build-api
	@echo "All packages and apps built successfully!"

# Test all packages
test:
	@echo "Running tests"
	uv run python -m pytest -c pyproject.toml --rootdir . -v -n auto --maxfail=3

# Test with coverage reports/combination helper
test-coverage:
	@if [ "$(DETECTED_OS)" = "Windows" ]; then \
		powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_coverage.ps1; \
	else \
		bash scripts/run_coverage.sh; \
	fi

# Test with specific number of workers
test-parallel:
	@echo "Running tests in parallel with custom workers..."
	@uv run python -m pytest -c pyproject.toml --rootdir . -v -n 4 --maxfail=3 --cov=packages --cov=apps --cov-report=term-missing

# Test unit tests only (fast)
test-unit:
	@echo "Running unit tests only..."
	@uv run python -m pytest -c pyproject.toml --rootdir . tests/unit -v -n auto --maxfail=3

# Test integration tests only (slower)
test-integration:
	@echo "Running integration tests only..."
	@uv run python -m pytest -c pyproject.toml --rootdir . tests/integration -v -n auto --maxfail=3

# Show slowest tests
test-slow:
	@echo "Running tests and showing slowest..."
	@uv run python -m pytest -c pyproject.toml --rootdir . -v -n auto --maxfail=3 --durations=10

# Lint all packages
lint:
	@echo "Linting all packages..."
	uv run ruff check apps packages scripts tests

# Format all packages
format:
	@echo "Formatting all packages..."
	uv run ruff format apps packages scripts tests

# Type checking
typecheck:
	@echo "Running pyright..."
	uv run pyright

# Clean build artifacts
clean:
	@if [ "$(DETECTED_OS)" = "Windows" ]; then \
		powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/clean.ps1; \
	else \
		bash scripts/clean.sh; \
	fi

# Package-specific builds
build-config:
	uv build --directory packages/config

build-utils:
	uv build --directory packages/utils

build-data:
	uv build --directory packages/data

build-models:
	uv build --directory packages/models

build-training:
	uv build --directory packages/training

build-cli:
	uv build --directory packages/cli

build-api:
	uv build --directory apps/api

# Application commands
serve-api:
	uv run python apps/api/run_api.py

# Git hooks
install-hooks:
	@echo "Installing Git hooks..."
	@if [ "$(DETECTED_OS)" = "Windows" ]; then \
		powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install_hooks.ps1; \
	else \
		bash scripts/install_hooks.sh; \
	fi

# Pre-commit checks (same as git hooks)
pre-commit:
	@if [ "$(DETECTED_OS)" = "Windows" ]; then \
		powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run-pre-commit.ps1; \
	else \
		bash scripts/run-pre-commit.sh; \
	fi

# Pre-commit auto-fix
pre-commit-fix:
	@if [ "$(DETECTED_OS)" = "Windows" ]; then \
		powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run-pre-commit-fix.ps1; \
	else \
		bash scripts/run-pre-commit-fix.sh; \
	fi

# Package-specific tests
test-config:
	@echo "Running config package tests..."
	@uv run python -m pytest -c pyproject.toml --rootdir . packages/config/img_classifier_config/tests -v

test-utils:
	@echo "Running utils package tests..."
	@uv run python -m pytest -c pyproject.toml --rootdir . packages/utils/img_classifier_utils/tests -v

test-data:
	@echo "Running data package tests..."
	@uv run python -m pytest -c pyproject.toml --rootdir . packages/data/img_classifier_data/tests -v

test-models:
	@echo "Running models package tests..."
	@uv run python -m pytest -c pyproject.toml --rootdir . packages/models/img_classifier_models/tests -v

test-training:
	@echo "Running training package tests..."
	@uv run python -m pytest -c pyproject.toml --rootdir . packages/training/img_classifier_training/tests -v

test-cli:
	@echo "Running CLI package tests..."
	@uv run python -m pytest -c pyproject.toml --rootdir . packages/cli/img_classifier_cli/tests -v

test-api:
	@echo "Running API app tests..."
	@uv run python -m pytest -c pyproject.toml --rootdir . apps/api/img_classifier_api/tests -v

# CI workflow
ci: lint typecheck test build
	@echo "CI checks passed!"
