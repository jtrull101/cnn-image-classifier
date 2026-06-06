#!/usr/bin/env bash
# Guard: fail if any tracked Python file contains an inline linter/type-checker
# suppression. The project policy is zero inline escape hatches — rule exceptions
# belong in pyproject.toml (e.g. [tool.ruff.lint.per-file-ignores]), not in code.
#
# Banned: # noqa, # type: ignore, # ty: ignore, # pyright:, # mypy:, # pylint:, # flake8:
#
# Used by the pre-commit hook, `make check-suppressions`, and CI.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Pattern of banned inline-suppression comments (case-insensitive).
PATTERN='#[[:space:]]*(noqa|type:[[:space:]]*ignore|ty:[[:space:]]*ignore|pyright:[[:space:]]*(ignore|basic|strict)|mypy:|pylint:[[:space:]]*disable|flake8:[[:space:]]*noqa)'

# Scan tracked Python files only.
files=$(git ls-files '*.py')
if [ -z "$files" ]; then
    echo "[OK] No Python files to scan."
    exit 0
fi

# shellcheck disable=SC2086
matches=$(grep -nEi "$PATTERN" $files || true)

if [ -n "$matches" ]; then
    echo "[ERROR] Inline linter/type-checker suppressions are not allowed:"
    echo ""
    echo "$matches"
    echo ""
    echo "Move the exception into pyproject.toml (e.g. [tool.ruff.lint.per-file-ignores]"
    echo "or [tool.ty.rules]) with a justifying comment, or fix the underlying issue."
    exit 1
fi

echo "[OK] No inline suppressions found."
exit 0
