#!/usr/bin/env bash
# Install Git pre-commit hooks

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo ""
echo "Installing Git pre-commit hooks..."
echo ""

# Check if .git directory exists
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "[ERROR] .git directory not found"
    echo "This doesn't appear to be a Git repository"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Copy pre-commit hook (bash version)
PRE_COMMIT_SRC="$SCRIPT_DIR/pre-commit"
PRE_COMMIT_DST="$HOOKS_DIR/pre-commit"

if [ -f "$PRE_COMMIT_SRC" ]; then
    cp "$PRE_COMMIT_SRC" "$PRE_COMMIT_DST"
    chmod +x "$PRE_COMMIT_DST"
    echo "[OK] Installed pre-commit hook (bash)"
fi

# Copy pre-commit hook (PowerShell version) - for Windows Git Bash users
PRE_COMMIT_PS_SRC="$SCRIPT_DIR/pre-commit.ps1"
PRE_COMMIT_PS_DST="$HOOKS_DIR/pre-commit.ps1"

if [ -f "$PRE_COMMIT_PS_SRC" ]; then
    cp "$PRE_COMMIT_PS_SRC" "$PRE_COMMIT_PS_DST"
    echo "[OK] Installed pre-commit.ps1 hook (PowerShell)"
fi

echo ""
echo "========================================"
echo "Git hooks installed successfully!"
echo "========================================"
echo ""
echo "The pre-commit hook will now run automatically before each commit."
echo ""
echo "It will check:"
echo "  1. Code formatting (ruff)"
echo "  2. Linting (ruff)"
echo "  3. Type checking (pyright)"
echo ""

