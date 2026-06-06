# CLAUDE.md — Image Classifier

Agentic instructions for Claude Code. See `AGENTS.md` for full contributor guidelines.

## Project at a Glance

UV-managed Python monorepo. Generalized CNN image classifier (originally Alzheimer's MRI staging). Python 3.11+.

**Package dependency order:** `config` → `utils` → `data` → `models` → `training` → `api`
`cli` pulls from all packages. `api` (FastAPI) is an application, not a library.

Scoped tests:
```bash
make test-api       # apps/api/img_classifier_api/tests
make test-config / test-data / test-models / test-training / test-utils / test-cli
uv run pytest path/to/test_file.py -v   # Single file
```

## Testing Conventions

**Markers:** `unit` (default), `integration`, `smoke`, `slow`, `serial`, `requires_gpu`, `requires_data`

- Unit tests: mocks allowed, no real I/O, fast. Mark with `pytestmark = pytest.mark.unit` at top of file.
- Integration tests: real dependencies, testcontainers for external services. Separate directory.
- Tests auto-marked `unit` if no marker present (see `conftest.py:pytest_collection_modifyitems`).
- Tests run parallel by default (`-n auto`). Use `@pytest.mark.serial` for tests that cannot parallelize.
- Coverage sources: all `img_classifier_*` packages. Only mock in unit tests; no mocking in integration tests.

## Code Style

- Line length: **100** characters. Type hints required on public functions/methods.
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

## Environment

Copy `.env.example` → `.env`. Key vars: `IMG_CLASSIFIER_API_KEY`, `IMG_CLASSIFIER_MODEL_DIR`, `IMG_CLASSIFIER_WORKING_DIR`, `IMG_CLASSIFIER_MAX_UPLOAD_BYTES`. Never commit `.env`.

## Before Committing / Opening a PR

1. `make check` must pass completely
2. Update `docs/ARCHITECTURE.md` or `docs/TESTING.md` if structure changed
3. New code needs tests; don't delete tests to meet coverage
4. Commit messages: imperative, ~50 chars (e.g., `Add path traversal guard to model loader`)
