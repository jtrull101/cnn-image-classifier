Run the full quality check suite using:

```bash
make check
```

This runs four steps in sequence:
1. Format check (`ruff format --check`)
2. Lint (`ruff check`)
3. Type check (`pyright`)
4. Tests (pytest parallel)

If format check fails: run `make format` then re-run `make check`.
If lint fails: fix flagged issues then re-run.
If typecheck fails: fix type errors — do NOT add `# type: ignore` without an explanatory comment.
If tests fail: investigate failures — do NOT delete tests or add `# pragma: no cover` to make coverage pass.

Report the result of all four steps.
