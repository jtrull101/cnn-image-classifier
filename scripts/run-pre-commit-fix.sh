#!/usr/bin/env bash
# Auto-fix pre-commit issues
# This script attempts to automatically fix formatting and linting issues

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "========================================="
echo "Attempting to auto-fix issues..."
echo "========================================="
echo ""

# Auto-fix format issues
echo "[1/3] Auto-fixing format issues..."
if uv run ruff format apps packages scripts tests; then
    echo "[OK] Formatting applied"
else
    echo "[WARNING] Some formatting issues remain"
fi
echo ""

# Auto-fix linting issues
echo "[2/3] Auto-fixing linting issues with --fix..."
if uv run ruff check --fix .; then
    echo "[OK] Linting issues fixed"
else
    echo "[WARNING] Some linting issues require manual fixing"
fi
echo ""

# Re-run all checks
echo "[3/3] Re-running all checks..."
echo ""
bash "$SCRIPT_DIR/run-pre-commit.sh"

