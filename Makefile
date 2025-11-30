.PHONY: help install sync clean test test-coverage test-parallel test-unit test-integration test-slow test-config test-data test-models test-training test-utils test-cli test-api lint format typecheck build build-config build-utils build-data build-models build-training build-cli build-api serve-api install-hooks pre-commit pre-commit-fix ci

# Default shell for Windows
SHELL := pwsh.exe
.SHELLFLAGS := -NoProfile -Command

PY_PACKAGES = "packages/config","packages/utils","packages/data","packages/models","packages/training","packages/cli","apps/api"

help:
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
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_uv.ps1
	@echo "Syncing UV workspace (including dev deps)..."
	@uv sync --all-groups
	@echo "Installing Git hooks..."
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_hooks.ps1

# Sync UV dependencies across workspace
sync:
	@echo "Syncing workspace dependencies..."
	uv sync --all-groups

# Build all packages/apps
build:
	@echo "Building all packages and apps with uv..."
	@pwsh -NoProfile -Command "foreach ($$pkg in @($(PY_PACKAGES))) { Write-Host \"[build] $$pkg\" -ForegroundColor Cyan; Push-Location $$pkg; uv build; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }; Pop-Location }"

# Test all packages with coverage
test:
	@echo "Running tests with coverage..."
	uv run python -m pytest -c pyproject.toml --rootdir . -v -n auto --maxfail=3 --cov=packages --cov=apps --cov-report=term-missing

# Test with coverage reports/combination helper
test-coverage:
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_coverage.ps1

# Test with specific number of workers
test-parallel:
	@echo "Running tests in parallel with custom workers..."
	.\scripts\run_tests.ps1 -Workers 4

# Test unit tests only (fast)
test-unit:
	@echo "Running unit tests only..."
	.\scripts\run_tests.ps1 -UnitOnly

# Test integration tests only (slower)
test-integration:
	@echo "Running integration tests only..."
	.\scripts\run_tests.ps1 -IntegrationOnly

# Show slowest tests
test-slow:
	@echo "Running tests and showing slowest..."
	.\scripts\run_tests.ps1 -ShowSlowest

# Package-specific test commands
test-config:
	@echo "Running config package tests..."
	.\scripts\run_tests.ps1 -Target config -Coverage

test-data:
	@echo "Running data package tests..."
	.\scripts\run_tests.ps1 -Target data -Coverage

test-models:
	@echo "Running models package tests..."
	.\scripts\run_tests.ps1 -Target models -Coverage

test-training:
	@echo "Running training package tests..."
	.\scripts\run_tests.ps1 -Target training -Coverage

test-utils:
	@echo "Running utils package tests..."
	.\scripts\run_tests.ps1 -Target utils -Coverage

test-cli:
	@echo "Running CLI package tests..."
	.\scripts\run_tests.ps1 -Target cli -Coverage

test-api:
	@echo "Running API app tests..."
	.\scripts\run_tests.ps1 -Target api -Coverage

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
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/clean.ps1

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
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_hooks.ps1

# Pre-commit checks (same as git hooks)
pre-commit:
	@.\scripts\run-pre-commit.ps1

# Pre-commit auto-fix
pre-commit-fix:
	@.\scripts\run-pre-commit-fix.ps1

# CI workflow
ci: lint typecheck test build
	@echo "CI checks passed!"
