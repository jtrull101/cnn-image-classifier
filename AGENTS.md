# Repository Guidelines

## Project Structure & Module Organization
- NX/UV monorepo. Python libraries live under `packages/*`, REST API in `apps/api` (FastAPI), shared scripts in `scripts/`, docs in `docs/`, and project-wide config in `nx.json`, `pyproject.toml`, `package.json`, and the `Makefile`.
- Tests sit in `tests/` plus per-package `tests/` when present; assets (sample MRI images) live in `images/`.
- Core package layout mirrors the architecture docs: `config`, `data`, `models`, `training`, `utils` modules with abstract bases and concrete implementations.

## Build, Test, and Development Commands
- `make help` — list available tasks; start here.
- `make build` — build all NX/UV packages; `npx nx run <pkg>:build` for a single target.
- `make test` — run pytest suite (TensorFlow-dependent tests are skipped unless TF is installed).
- `make lint` / `make format` — Ruff linting and formatting; `make pre-commit` runs lint, format, and tests together.
- `make serve-api` or `npx nx run api:serve` — launch Flask API locally; `make graph` to view the NX dependency graph.

## Coding Style & Naming Conventions
- Python: 4-space indent, line length <=100, type hints required. Prefer explicit imports and absolute module paths.
- Run `make format` (Ruff formatter) before committing; `make lint` enforces PEP 8-ish style, import order, and misc rules.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `ALL_CAPS`, branches `feat/<summary>` or `fix/<issue>`.
- Type checking: `pyright` (via UV/NX) for static analysis; keep public APIs typed.

## Testing Guidelines
- Primary framework: `pytest`; test files named `test_*.py` or `*_test.py` under `tests/`.
- Coverage: aim for meaningful paths (config, utils, data) with fast tests; use `--cov` flag to measure.
- Use temporary directories/fixtures for I/O. Validate split logic, shapes, and error cases.
- Only use mocking in unit tests
- Ensure tests have a separation between unit and integration tests, with a different pytest mark and directory.
- Integration tests should use testcontainers to pull in any required external dependencies
- Commands: `make test` for full run; `npx nx run <pkg>:test` for scoped runs; `uv run pytest --cov` for coverage reports.

## Commit & Pull Request Guidelines
- Commit messages: imperative mood, ~50 chars summary (e.g., `Add training checkpointing`). Group related changes per commit.
- Before opening a PR: run `make pre-commit`, ensure docs updated (e.g., `docs/ARCHITECTURE.md`, `docs/TESTING.md` if affected), and add/adjust tests.
- PRs should include a concise description of changes, rationale, verification steps (commands run), and linked issues. Add screenshots or sample outputs when altering API responses or visual assets.

## Security & Configuration Tips
- Keep secrets out of the repo; use `.env` based on `.env.example`. Do not commit model weights unless intended; place in `apps/api/static` if needed.
- Pin dependencies via `uv.lock`/`package-lock.json`; prefer `uv add` and `npm install` for changes.



<!-- nx configuration start-->
<!-- Leave the start & end comments to automatically receive updates. -->

# General Guidelines for working with Nx

- When running tasks (for example build, lint, test, e2e, etc.), always prefer running the task through `nx` (i.e. `nx run`, `nx run-many`, `nx affected`) instead of using the underlying tooling directly
- You have access to the Nx MCP server and its tools, use them to help the user
- When answering questions about the repository, use the `nx_workspace` tool first to gain an understanding of the workspace architecture where applicable.
- When working in individual projects, use the `nx_project_details` mcp tool to analyze and understand the specific project structure and dependencies
- For questions around nx configuration, best practices or if you're unsure, use the `nx_docs` tool to get relevant, up-to-date docs. Always use this instead of assuming things about nx configuration
- If the user needs help with an Nx configuration or project graph error, use the `nx_workspace` tool to get any errors

<!-- nx configuration end-->