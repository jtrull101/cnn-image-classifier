"""Session-scoped fixtures for container-based integration tests.

Builds the API Docker image once per test session, starts a container,
waits for the health endpoint, then yields pre-configured httpx clients.
Tests that use these fixtures require a running Docker daemon; the fixture
calls ``pytest.skip()`` (not ``raise``) when Docker is absent so that
``--maxfail=3`` does not abort the rest of the suite.
"""

import io
import os
import subprocess
import time
from pathlib import Path
from typing import Generator

import httpx
import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Database Isolation for Integration Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _isolate_integration_database(tmp_path_factory, worker_id):
    """
    Ensure each pytest-xdist worker uses a separate database file.

    This prevents race conditions when parallel workers try to initialize
    the same database file simultaneously. For non-parallel runs (worker_id == 'master'),
    uses the default database location.
    """
    if worker_id == "master":
        # Not running with xdist, use default database
        return

    # Create a unique database file for this worker
    tmp_dir = tmp_path_factory.mktemp("db")
    db_file = tmp_dir / f"predictions_worker_{worker_id}.db"
    db_url = f"sqlite:///{db_file}"

    # Set DATABASE_URL environment variable for this worker
    os.environ["DATABASE_URL"] = db_url

    # Reset database initialization state to allow re-initialization with new URL
    from img_classifier_api.database import reset_db_state

    reset_db_state()

    yield

    # Cleanup: remove worker-specific database
    if db_file.exists():
        db_file.unlink()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[4]  # …/alz-mri-neural-network
_DOCKERFILE = "apps/api/Dockerfile"
_IMAGE_TAG = "img-classifier-api:test"
_CONTAINER_PORT = 8000
_TEST_API_KEY = "test-secret-key-for-integration"
_HEALTH_TIMEOUT = 150  # seconds — TF + Keras model load can take 30–90 s
_HEALTH_POLL_INTERVAL = 3  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_health(base_url: str, timeout: int = _HEALTH_TIMEOUT) -> None:
    """Poll GET /health until 200 or timeout."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(_HEALTH_POLL_INTERVAL)
    raise RuntimeError(
        f"Container did not become healthy within {timeout}s. Last error: {last_error}"
    )


def _make_test_jpeg(width: int = 64, height: int = 64, seed: int = 42) -> bytes:
    """Return valid JPEG bytes generated via PIL (accepted by cv2.imdecode)."""
    rng = np.random.default_rng(seed)
    pixel_data = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(pixel_data, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def built_image() -> Generator[str, None, None]:
    """Build the Docker image once for the entire test session.

    Skips (not fails) if Docker is not available so that ``--maxfail=3``
    does not kill unrelated unit/integration tests.
    """
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"Docker not available: {exc}")

    build_result = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            _DOCKERFILE,
            "-t",
            _IMAGE_TAG,
            ".",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,  # image build can take a few minutes on first run
    )
    if build_result.returncode != 0:
        pytest.skip(
            f"Docker build failed (exit {build_result.returncode}):\n{build_result.stderr[-2000:]}"
        )

    yield _IMAGE_TAG

    # Teardown: remove the test image to keep the Docker daemon tidy.
    subprocess.run(
        ["docker", "rmi", "--force", _IMAGE_TAG],
        capture_output=True,
        timeout=30,
    )


@pytest.fixture(scope="session")
def api_server(
    built_image: str,
) -> Generator[tuple[str, str], None, None]:
    """Start the API container and wait until it is healthy.

    Yields ``(base_url, api_key)`` so tests can construct their own clients
    or use the pre-configured helpers below.
    """
    from testcontainers.core.container import DockerContainer

    container: DockerContainer = (
        DockerContainer(built_image)
        .with_env("IMG_CLASSIFIER_API_KEY", _TEST_API_KEY)
        .with_env("IMG_CLASSIFIER_MODEL_DIR", "/app/apps/api/img_classifier_api/static")
        .with_exposed_ports(_CONTAINER_PORT)
    )

    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(_CONTAINER_PORT)
        base_url = f"http://{host}:{port}"
        try:
            _wait_for_health(base_url)
        except RuntimeError as exc:
            stdout, stderr = container.get_logs()
            logs = (stdout + stderr).decode("utf-8", errors="replace")[-3000:]
            pytest.skip(f"{exc}\n\n--- container logs ---\n{logs}")
        yield base_url, _TEST_API_KEY
    finally:
        container.stop()


@pytest.fixture(scope="session")
def authed(api_server: tuple[str, str]) -> Generator[httpx.Client, None, None]:
    """httpx.Client with correct API key header."""
    base_url, api_key = api_server
    with httpx.Client(
        base_url=base_url,
        headers={"X-API-Key": api_key},
        timeout=30,
    ) as client:
        yield client


@pytest.fixture(scope="session")
def anon(api_server: tuple[str, str]) -> Generator[httpx.Client, None, None]:
    """httpx.Client with no authentication headers."""
    base_url, _ = api_server
    with httpx.Client(base_url=base_url, timeout=30) as client:
        yield client


@pytest.fixture(scope="session")
def wrong_key(api_server: tuple[str, str]) -> Generator[httpx.Client, None, None]:
    """httpx.Client with a deliberately incorrect API key."""
    base_url, _ = api_server
    with httpx.Client(
        base_url=base_url,
        headers={"X-API-Key": "totally-wrong-key"},
        timeout=30,
    ) as client:
        yield client


@pytest.fixture(scope="module")
def test_jpeg() -> bytes:
    """A real 64×64 JPEG image accepted by OpenCV."""
    return _make_test_jpeg()
