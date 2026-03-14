"""E2E test configuration: live uvicorn server + Playwright browser fixtures."""

import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path

import pytest

# Add the e2e package directory to sys.path so tests can import
#   from fixtures.api_responses import ...
# without needing __init__.py files in all ancestor directories.
_E2E_DIR = str(Path(__file__).parent)
if _E2E_DIR not in sys.path:
    sys.path.insert(0, _E2E_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to port 0 and let the OS pick a free ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, timeout: float = 30.0) -> None:
    """Poll GET /health until HTTP 200 or *timeout* seconds have elapsed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout}s")


# ---------------------------------------------------------------------------
# Session-scoped live server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def live_server_url(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Start a real uvicorn server once for the entire e2e test session.

    * Points IMG_CLASSIFIER_MODEL_DIR at a fresh temp dir → "no model loaded" state.
    * Unsets IMG_CLASSIFIER_API_KEY → auth disabled (dev mode).
    * IMG_CLASSIFIER_WORKING_DIR is also pointed at the temp dir to avoid
      writing to the real user home directory.
    """
    tmp_root = tmp_path_factory.mktemp("e2e_server")
    model_dir = tmp_root / "models"
    model_dir.mkdir()

    # Must set env vars BEFORE importing img_classifier_api.app because
    # model_manager = ModelManager() runs at module level and calls
    # _default_model_dirs() which reads IMG_CLASSIFIER_MODEL_DIR.
    os.environ["IMG_CLASSIFIER_MODEL_DIR"] = str(model_dir)
    os.environ["IMG_CLASSIFIER_WORKING_DIR"] = str(tmp_root)
    os.environ.pop("IMG_CLASSIFIER_API_KEY", None)

    import uvicorn  # noqa: PLC0415 (deferred to respect env var ordering)

    from img_classifier_api.app import app  # noqa: PLC0415

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    _wait_for_server(base_url)
    yield base_url

    server.should_exit = True
    thread.join(timeout=10.0)


# ---------------------------------------------------------------------------
# Function-scoped page override
# ---------------------------------------------------------------------------


@pytest.fixture()
def page(page, live_server_url: str):  # noqa: F811  (intentional shadow of playwright fixture)
    """Override pytest-playwright's page fixture to set defaults."""
    page.set_default_timeout(15_000)
    return page
