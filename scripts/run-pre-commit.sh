#!/usr/bin/env bash
# Run pre-commit checks
# This script replicates the behavior of the Git pre-commit hook

echo ""
echo "========================================="
echo "Running pre-commit checks..."
echo "========================================="
echo ""

ERROR_COUNT=0

# Format code
echo "[1/4] Formatting code with ruff..."
if uv run ruff format apps packages scripts tests; then
    echo "[OK] Formatting complete"
else
    echo "[ERROR] Formatting failed"
    ((ERROR_COUNT++))
fi
echo ""

# Lint code
echo "[2/4] Linting code with ruff..."
if uv run ruff check apps packages scripts tests; then
    echo "[OK] Linting passed"
else
    echo "[ERROR] Linting failed"
    ((ERROR_COUNT++))
fi
echo ""

# Type check
echo "[3/4] Running type checks with pyright..."
if uv run pyright; then
    echo "[OK] Type checking passed"
else
    echo "[ERROR] Type checking failed"
    ((ERROR_COUNT++))
fi
echo ""

# Run tests
echo "[4/4] Running tests..."
if uv run python -m pytest -c pyproject.toml --rootdir . -v --maxfail=3 --cov=packages --cov=apps --cov-report=term-missing; then
    echo "[OK] Tests passed"
else
    echo "[ERROR] Tests failed"
    ((ERROR_COUNT++))
fi
echo ""

# Summary
echo "========================================="
if [ $ERROR_COUNT -eq 0 ]; then
    echo "All pre-commit checks passed!"
    echo "========================================="
    exit 0
else
    echo "Pre-commit checks failed: $ERROR_COUNT error(s)"
    echo "========================================="
    exit 1
fi

