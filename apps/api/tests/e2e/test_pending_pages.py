"""Skeleton stubs for UI pages that are not yet implemented.

Tests skip gracefully when a route returns non-200, and pass automatically
once the pages are built.  Direct API-level tests verify that the backing
JSON endpoints exist even before the UI pages are live.
"""

import pytest
from playwright.sync_api import Page


pytestmark = [pytest.mark.e2e, pytest.mark.serial]


# ---------------------------------------------------------------------------
# Pending UI pages — skip until implemented
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/history", "/analytics", "/datasets", "/compare", "/batch"],
)
def test_pending_page_loads_or_skips(page: Page, live_server_url: str, path: str) -> None:
    """Navigate to *path*; skip gracefully if the page is not yet built (non-200)."""
    response = page.goto(f"{live_server_url}{path}")
    assert response is not None

    if response.status != 200:
        pytest.skip(f"{path} not yet implemented (HTTP {response.status})")

    # If the page exists, at minimum verify it renders an HTML document
    assert "html" in page.content().lower(), f"{path} returned non-HTML response"


# ---------------------------------------------------------------------------
# API-level checks — these endpoints must exist even without UI pages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "api_path",
    [
        "/api/history",
        "/api/analytics/summary",
        "/api/analytics/performance",
    ],
)
def test_api_endpoint_returns_200(page: Page, live_server_url: str, api_path: str) -> None:
    """The backing JSON API endpoint returns HTTP 200."""
    response = page.request.get(f"{live_server_url}{api_path}")
    assert response.status == 200, (
        f"Expected 200 from {api_path}, got {response.status}: {response.text()[:200]}"
    )
