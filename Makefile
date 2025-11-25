.PHONY: help install sync clean test lint format build graph install-hooks pre-commit pre-commit-fix ci

# Default shell for Windows
SHELL := pwsh.exe
.SHELLFLAGS := -NoProfile -Command

help:
	@echo "Alzheimer's MRI CNN - NX/UV Monorepo"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install NX and setup workspace"
	@echo "  make install-hooks  - Install Git pre-commit hooks"
	@echo "  make sync           - Sync all UV dependencies"
	@echo "  make build          - Build all packages"
	@echo "  make test           - Run tests for all packages"
	@echo "  make lint           - Lint all packages"
	@echo "  make format         - Format all packages"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make graph          - Show dependency graph"
	@echo ""
	@echo "Pre-commit commands:"
	@echo "  make pre-commit     - Run all pre-commit checks (format, lint, typecheck, test)"
	@echo "  make pre-commit-fix - Auto-fix formatting and linting issues"
	@echo ""
	@echo "Package-specific commands:"
	@echo "  make build-config   - Build config package"
	@echo "  make build-utils    - Build utils package"
	@echo "  make build-data     - Build data package"
	@echo "  make build-models   - Build models package"
	@echo "  make build-training - Build training package"
	@echo "  make build-api      - Build API application"
	@echo ""
	@echo "Application commands:"
	@echo "  make serve-api      - Start API server"
	@echo ""
	@echo "Git hooks:"
	@echo "  Pre-commit checks run automatically on each commit"
	@echo "  Run 'make install-hooks' to install/reinstall hooks"

# Install NX and dependencies
install:
	@echo "Installing NX..."
	npm install
	@echo "Installing UV dependencies..."
	uv sync

# Sync UV dependencies across workspace
sync:
	@echo "Syncing workspace dependencies..."
	uv sync

# Build all packages
build:
	@echo "Building all packages..."
	npx nx run-many --target=build --all

# Test all packages
test:
	@echo "Running tests for all packages..."
	npx nx run-many --target=test --all

# Lint all packages
lint:
	@echo "Linting all packages..."
	npx nx run-many --target=lint --all

# Format all packages
format:
	@echo "Formatting all packages..."
	npx nx run-many --target=format --all

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	Get-ChildItem -Path . -Include __pycache__,*.pyc,.pytest_cache,.ruff_cache,dist,build,*.egg-info,node_modules -Recurse -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
	@echo "Clean complete!"

# Show dependency graph
graph:
	@echo "Generating dependency graph..."
	npx nx graph

# Package-specific builds
build-config:
	npx nx run config:build

build-utils:
	npx nx run utils:build

build-data:
	npx nx run data:build

build-models:
	npx nx run models:build

build-training:
	npx nx run training:build

build-api:
	npx nx run api:build

# Application commands
serve-api:
	npx nx run api:serve

# Git hooks
install-hooks:
	@echo "Installing Git hooks..."
	.\scripts\install_hooks.ps1

# Pre-commit checks (same as git hooks)
pre-commit:
	@.\scripts\run-pre-commit.ps1

# Pre-commit auto-fix
pre-commit-fix:
	@.\scripts\run-pre-commit-fix.ps1

# CI workflow
ci: lint test build
	@echo "CI checks passed!"
