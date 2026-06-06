# Repository Guidelines

## Project Structure & Module Organization
- UV-managed Python monorepo. Libraries live under `packages/*`, REST API in `apps/api` (FastAPI), shared scripts in `scripts/`, docs in `docs/`, and project-wide config in `pyproject.toml` plus the `Makefile`.
- Tests sit in `tests/` plus per-package `tests/` when present; assets (sample MRI images) live in `images/`.
- Core package layout mirrors the architecture docs: `config`, `data`, `models`, `training`, `utils`, and `cli` modules with abstract bases and concrete implementations.

## Build, Test, and Development Commands
- `make help` — list available tasks; start here.
- `make sync` / `uv sync --dev` — install workspace deps.
- `make build` — build all Python packages/apps (`uv build` in each).
- `make test` — run pytest suite with coverage (TensorFlow-dependent tests are skipped unless TF is installed).
- `make lint` / `make format` / `make typecheck` — Ruff lint/format and ty; `make pre-commit` runs format, lint, typecheck, spell, and tests together.
- `make spell` / `make audit` / `make deadcode` — codespell, pip-audit (dependency CVEs), and vulture; `make security` runs all three.
- `make serve-api` — launch FastAPI locally; `python apps/api/run_api.py` also works.

## Coding Style & Naming Conventions
- Python: 4-space indent, line length <=100, type hints required. Prefer explicit imports and absolute module paths.
- Run `make format` (Ruff formatter) before committing; `make lint` enforces PEP 8-ish style, import order, security (`S`), docstrings (`D`, Google convention), and misc rules.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `ALL_CAPS`, branches `feat/<summary>` or `fix/<issue>`.
- Type checking: `ty` (via UV) for static analysis; keep public APIs typed.
- Imports must be absolute (ruff `TID` bans relative imports); public functions need Google-style docstrings.

## Testing Guidelines
- Primary framework: `pytest`; test files named `test_*.py` or `*_test.py` under `tests/`.
- Coverage: aim for meaningful paths (config, utils, data) with fast tests; use `--cov` flag to measure.
- Use temporary directories/fixtures for I/O. Validate split logic, shapes, and error cases.
- Only use mocking in unit tests.
- Ensure tests have a separation between unit and integration tests, with a different pytest mark and directory.
- Integration tests should use testcontainers to pull in any required external dependencies.
- Commands: `make test` for full run; `uv run pytest --cov` for coverage reports; use `scripts/run_tests.ps1` for parallel/filtered runs.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood, ~50 chars summary (e.g., `Add training checkpointing`). Group related changes per commit. A `commit-msg` hook enforces this (subject length, capitalization, no trailing period, blank line before body).
- Before opening a PR: run `make pre-commit`, ensure docs updated (e.g., `docs/ARCHITECTURE.md`, `docs/TESTING.md` if affected), and add/adjust tests.
- PRs should include a concise description of changes, rationale, verification steps (commands run), and linked issues. Add screenshots or sample outputs when altering API responses or visual assets.

## Security & Configuration Tips
- Keep secrets out of the repo; use `.env` based on `.env.example`. Do not commit model weights unless intended; place in `apps/api/static` if needed.
- Pin dependencies via `uv.lock`; prefer `uv add` for changes.
