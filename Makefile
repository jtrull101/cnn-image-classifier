.PHONY: help install sync clean test test-coverage test-parallel test-unit test-integration test-slow test-config test-data test-models test-training test-utils test-cli test-api test-e2e install-playwright lint format typecheck spell audit deadcode security check-suppressions build build-config build-utils build-data build-models build-training build-cli build-api serve-api install-hooks pre-commit check ci download-datasets list-datasets db-upgrade db-migrate db-revision db-history db-current db-downgrade db-reset

# Platform detection
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
else
    DETECTED_OS := $(shell uname -s)
endif


# Per-OS script commands (avoid shell conditionals in recipes)
ifeq ($(DETECTED_OS),Windows)
    INSTALL_UV_CMD := powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install_uv.ps1
    INSTALL_HOOKS_CMD := powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install_hooks.ps1
    COVERAGE_CMD := powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run_coverage.ps1
    PRE_COMMIT_CMD := powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/run-pre-commit.ps1
    CLEAN_CMD := powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/clean.ps1
else
    INSTALL_UV_CMD := bash scripts/install_uv.sh
    INSTALL_HOOKS_CMD := bash scripts/install_hooks.sh
    COVERAGE_CMD := bash scripts/run_coverage.sh
    PRE_COMMIT_CMD := bash scripts/run-pre-commit.sh
    CLEAN_CMD := bash scripts/clean.sh
endif


PY_PACKAGES = "packages/config","packages/utils","packages/data","packages/models","packages/training","packages/cli","apps/api"

help:
	@echo "Detected OS: $(DETECTED_OS)"
	@echo "Image Classifier - Python/UV Monorepo"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install UV and sync dependencies"
	@echo "  make install-hooks  - Install Git pre-commit hooks"
	@echo "  make sync           - Sync all UV dependencies"
	@echo "  make build          - Build all packages/apps (uv build)"
	@echo "  make test           - Run tests with coverage (parallel)"
	@echo "  make lint           - Lint all code with ruff"
	@echo "  make format         - Format all code with ruff"
	@echo "  make typecheck      - Run ty across the workspace"
	@echo "  make spell          - Spell check code and docs (codespell)"
	@echo "  make audit          - Audit dependencies for CVEs (pip-audit)"
	@echo "  make deadcode       - Detect unused code (vulture, informational)"
	@echo "  make security       - Run audit + spell + deadcode together"
	@echo "  make check-suppressions - Fail if inline noqa/type-ignore comments exist"
	@echo "  make check          - Run all pre-commit checks (format, lint, typecheck, tests)"
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
	@echo "  make test-e2e       - Run browser E2E tests (Playwright)"
	@echo "  make install-playwright - Download Playwright Chromium binary"
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
	@echo "  make run      	     - Start API server"
	@echo ""
	@echo "Database commands:"
	@echo "  make db-upgrade     - Run database migrations to latest version"
	@echo "  make db-migrate     - Alias for db-upgrade"
	@echo "  make db-revision    - Create new migration (use msg='description')"
	@echo "  make db-history     - Show migration history"
	@echo "  make db-current     - Show current migration version"
	@echo "  make db-downgrade   - Downgrade database by one version"
	@echo "  make db-reset       - Reset database (delete and recreate)"
	@echo ""
	@echo "Dataset commands:"
	@echo "  make download-datasets - Download all datasets from GitHub Releases"
	@echo "  make list-datasets     - List available datasets and their status"
	@echo ""
	@echo "Git hooks:"
	@echo "  Pre-commit checks run automatically on each commit"
	@echo "  Run 'make install-hooks' to install/reinstall hooks"

# Install UV and dependencies
install:
	@echo "Checking UV installation..."
	@$(INSTALL_UV_CMD)
	@echo "Syncing UV workspace (including dev deps)..."
	@uv sync --all-groups
	@echo "Installing Git hooks..."
	@$(INSTALL_HOOKS_CMD)

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
	uv run python -m pytest -c pyproject.toml --rootdir . -v --maxfail=3

# Test with coverage reports/combination helper
test-coverage:
	@$(COVERAGE_CMD)

# Test with specific number of workers
test-parallel:
	@echo "Running tests in parallel with custom workers..."
	@uv run python -m pytest -c pyproject.toml --rootdir . -v --maxfail=3 --cov=packages --cov=apps --cov-report=term-missing

# Test unit tests only (fast)
test-unit:
	@echo "Running unit tests only..."
	@uv run python -m pytest -c pyproject.toml --rootdir . tests/unit -v --maxfail=3

# Test integration tests only (slower)
test-integration:
	@echo "Running integration tests only..."
	@uv run python -m pytest -c pyproject.toml --rootdir . tests/integration -v --maxfail=3

# Show slowest tests
test-slow:
	@echo "Running tests and showing slowest..."
	@uv run python -m pytest -c pyproject.toml --rootdir . -v --maxfail=3 --durations=10

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
	@echo "Running ty..."
	uv run ty check

# Spell checking
spell:
	@echo "Spell checking with codespell..."
	uv run codespell

# Fail if inline linter/type-checker suppressions were added
check-suppressions:
	@echo "Checking for banned inline suppressions..."
	bash scripts/check-no-suppressions.sh

# Dependency vulnerability audit (exits non-zero when CVEs are found)
audit:
	@echo "Auditing dependencies for known vulnerabilities..."
	uv run pip-audit --desc

# Dead-code detection (informational; uses vulture_allowlist.py)
deadcode:
	@echo "Detecting unused code with vulture..."
	uv run vulture

# Aggregate security/quality scans
security: audit spell deadcode
	@echo "Security and quality scans complete!"

# Clean build artifacts
clean:
	@$(CLEAN_CMD)

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
run: docker-up

# Docker Compose stack
certs:
	@bash scripts/generate-certs.sh

docker-up: certs
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# Git hooks
install-hooks:
	@echo "Installing Git hooks..."
	@$(INSTALL_HOOKS_CMD)

# Run all pre-commit checks (format check, lint, typecheck, tests)
check:
	@$(PRE_COMMIT_CMD)

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

test-e2e:
	@echo "Running browser E2E tests (Playwright)..."
	@LD_LIBRARY_PATH=$(PLAYWRIGHT_LIBS):$$LD_LIBRARY_PATH \
		uv run pytest -c pyproject.toml --rootdir . apps/api/tests/e2e -v \
		--override-ini="addopts=" \
		-p no:xdist \
		-m "e2e" \
		--timeout=120 \
		--tb=short \
		-ra

install-playwright:
	@echo "Installing Playwright Chromium browser..."
	uv run playwright install chromium
	@echo "Downloading Chromium system library dependencies (for minimal envs)..."
	@mkdir -p $(PLAYWRIGHT_LIBS)
	@cd /tmp && apt-get download libnspr4 libnss3 libasound2t64 2>/dev/null || true
	@for deb in /tmp/libnspr4*.deb /tmp/libnss3*.deb /tmp/libasound2t64*.deb; do \
		[ -f "$$deb" ] && dpkg-deb -x "$$deb" /tmp/pw-libs 2>/dev/null || true; \
	done
	@cp -rn /tmp/pw-libs/usr/lib/x86_64-linux-gnu/. $(PLAYWRIGHT_LIBS)/ 2>/dev/null || true
	@echo "Playwright install complete. Library path: $(PLAYWRIGHT_LIBS)"

PLAYWRIGHT_LIBS ?= /tmp/nspr-libs/usr/lib/x86_64-linux-gnu

# CI workflow
ci: lint typecheck spell check-suppressions test build
	@echo "CI checks passed!"

# Dataset management
download-datasets:
	@echo "Downloading all datasets from GitHub Releases..."
	uv run python scripts/download_datasets.py

list-datasets:
	@echo "Listing available datasets..."
	uv run python scripts/download_datasets.py

# Database management (Alembic migrations)
db-upgrade:
	@echo "Upgrading database to latest version..."
	cd apps/api && uv run alembic upgrade head

db-migrate: db-upgrade

db-revision:
	@echo "Creating new database migration..."
	@if [ -z "$(msg)" ]; then \
		echo "Error: Please provide a message with msg='description'"; \
		echo "Example: make db-revision msg='add user table'"; \
		exit 1; \
	fi
	cd apps/api && uv run alembic revision --autogenerate -m "$(msg)"

db-history:
	@echo "Migration history:"
	cd apps/api && uv run alembic history --verbose

db-current:
	@echo "Current database version:"
	cd apps/api && uv run alembic current

db-downgrade:
	@echo "Downgrading database by one version..."
	cd apps/api && uv run alembic downgrade -1

db-reset:
	@echo "Resetting database..."
	rm -f apps/api/data/predictions.db
	@echo "Database deleted. Running migrations..."
	cd apps/api && uv run alembic upgrade head
	@echo "Database reset complete!"
