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
- Imports: absolute only. Quote style: double (ruff enforced).
- Run `make format` before committing; `make lint` to check.
- Ruff ignores (do not "fix"): `ARG001`/`ARG002` (ML callbacks), `E501` (line length).
- Pyright: `reportAttributeAccessIssue = false` for TensorFlow/Keras dynamic attrs.
- Tests additionally suppress `ARG`, `S101`, `N806` (ML variable names like `X_train`).

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
