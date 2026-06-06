# CLAUDE.md — Image Classifier

Single source of contributor and agent guidelines for this repo. (`AGENTS.md` points here.)

## Project at a Glance

UV-managed Python monorepo. Generalized CNN image classifier (originally Alzheimer's MRI staging). Python 3.11+.

- Libraries live under `packages/*`; the FastAPI REST API is in `apps/api` (an application, not a library). Shared scripts in `scripts/`, docs in `docs/`, sample assets in `images/`. Project-wide config in the root `pyproject.toml` and `Makefile`.
- **Package dependency order:** `config` → `utils` → `data` → `models` → `training` → `api`. `cli` pulls from all packages.
- Each package mirrors the architecture docs: abstract bases plus concrete implementations.

## Build & Dev Commands

- `make help` — list all tasks; start here.
- `make sync` (or `uv sync --all-groups`) — install workspace deps. Prefer `uv add` for dependency changes; everything is pinned via `uv.lock`.
- `make build` — build all packages/apps (`uv build` in each).
- `make lint` / `make format` / `make typecheck` — Ruff lint/format and ty.
- `make spell` / `make audit` / `make deadcode` / `make security` — codespell, pip-audit (dependency CVEs), vulture, and all three together.
- `make check` (alias `make pre-commit`) — format, lint, typecheck, spell, suppression guard, and tests.
- `make serve-api` — launch FastAPI locally (`python apps/api/run_api.py` also works).

Scoped tests:
```bash
make test-api       # apps/api/img_classifier_api/tests
make test-config / test-data / test-models / test-training / test-utils / test-cli
uv run pytest path/to/test_file.py -v   # Single file
```

## Testing Conventions

**Markers:** `unit` (default), `integration`, `smoke`, `slow`, `serial`, `requires_gpu`, `requires_data`

- Test files: `test_*.py` or `*_test.py`, under the root `tests/` or a per-package `tests/`.
- Unit tests: mocks allowed, no real I/O, fast. Mark with `pytestmark = pytest.mark.unit` at top of file.
- Integration tests: real dependencies, testcontainers for external services. Separate directory; no mocking.
- Tests auto-marked `unit` if no marker present (see `conftest.py:pytest_collection_modifyitems`).
- Tests run in parallel by default. Use `@pytest.mark.serial` for tests that cannot parallelize.
- Use temp dirs/fixtures for I/O; validate split logic, shapes, and error cases. Coverage sources: all `img_classifier_*` packages.

## Code Style

- Line length: **100** characters; 4-space indent. Type hints required on public functions/methods.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `ALL_CAPS`. Branches `feat/<summary>` or `fix/<issue>`.
- Imports: **absolute only** (ruff `TID` bans relative imports). Quote style: double (ruff enforced).
- Docstrings: **Google convention** (`Args:`/`Returns:`/`Raises:`), enforced by ruff `D` rules.
- Run `make format` before committing; `make lint` to check.
- Ruff selects security (`S`), docstrings (`D`), pytest-style (`PT`), return/perf/logging/async (`RET`/`PERF`/`LOG`/`G`/`ASYNC`), tidy-imports (`TID`), future-annotations (`FA`) on top of the base set.
- **Single ruff config:** all rules live in the root `pyproject.toml`. Do NOT add `[tool.ruff]` to per-package `pyproject.toml` (it shadows the root and silently disables rules).
- Ruff ignores (do not "fix"): `ARG001`/`ARG002`/`ARG004` (ML callbacks/uniform signatures), `E501` (line length), `RUF001-003` (em-dash typography), `DTZ001`/`DTZ005` (intentional naive timestamps).
- ty: `unresolved-attribute = "ignore"` (and related attribute rules) for TensorFlow/Keras dynamic attrs.
- Tests additionally suppress `ARG`, `S` (assert/pickle/tmp), `D`, `N806`/`N817`, `B017`, `SIM117`, `PTH`, `E402`, `F401`.
- **No inline suppressions.** `# noqa`, `# type: ignore`, `# ty: ignore`, `# pyright:`, etc. are banned and enforced by `make check-suppressions` (in CI + pre-commit). Put rule exceptions in `[tool.ruff.lint.per-file-ignores]` / `[tool.ty.rules]` with a justifying comment, or fix the underlying issue.

## Quality & Security Tooling

- `make spell` — codespell (config + word allowlist in `[tool.codespell]`).
- `make audit` — pip-audit for dependency CVEs (informational; Dependabot does the bumps).
- `make deadcode` — vulture dead-code scan; false positives go in `vulture_allowlist.py`.
- `make security` — runs audit + spell + deadcode together.
- Secrets scanning: gitleaks runs in CI (`.github/workflows/security.yml`) and locally in the pre-commit hook **if** the `gitleaks` binary is installed; config/allowlist in `.gitleaks.toml`.
- A `commit-msg` hook (`scripts/commit-msg`) enforces the commit convention (installed by `make install-hooks`).

## API Conventions

- Auth: `X-API-Key` header required when `IMG_CLASSIFIER_API_KEY` env var is set; disabled otherwise.
- `/health` endpoint requires no auth.
- WebSocket at `/ws` for real-time training progress.
- Model files discovered from `IMG_CLASSIFIER_MODEL_DIR` or `~/.local/share/img_classifier/models/`.

## Environment & Configuration

- Copy `.env.example` → `.env`. Key vars: `IMG_CLASSIFIER_API_KEY`, `IMG_CLASSIFIER_MODEL_DIR`, `IMG_CLASSIFIER_WORKING_DIR`, `IMG_CLASSIFIER_MAX_UPLOAD_BYTES`. Never commit `.env`.
- Keep secrets out of the repo. Don't commit model weights unless intended; place them under `apps/api/static` if needed.

## Before Committing / Opening a PR

1. `make check` must pass completely.
2. Update `docs/ARCHITECTURE.md` or `docs/TESTING.md` if structure changed.
3. New code needs tests; don't delete tests to meet coverage.
4. Commit messages: imperative, ~50 chars (e.g., `Add path traversal guard to model loader`). Group related changes per commit. The `commit-msg` hook enforces subject length, capitalization, no trailing period, and a blank line before the body.
5. PRs: include a concise description, rationale, verification steps (commands run), and linked issues. Add screenshots or sample outputs when changing API responses or visual assets.
