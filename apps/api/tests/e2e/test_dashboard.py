"""E2E tests for the main dashboard (/ route, index.html)."""

import json

import pytest
from fixtures.api_responses import (
    ANALYTICS_SUMMARY_EMPTY,
    MODELS_EMPTY,
    MODELS_ONE_LOADED,
)
from playwright.sync_api import Page, expect


pytestmark = [pytest.mark.e2e, pytest.mark.serial]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_empty(page: Page) -> None:
    """Route API calls to return empty / zero data."""
    page.route(
        "**/api/models",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(MODELS_EMPTY),
        ),
    )
    page.route(
        "**/api/analytics/summary",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(ANALYTICS_SUMMARY_EMPTY),
        ),
    )
    # HTMX history widget — return minimal empty HTML
    page.route(
        "**/api/history*",
        lambda route: route.fulfill(
            status=200,
            content_type="text/html",
            body="<p class='text-gray-400 text-sm'>No predictions yet.</p>",
        ),
    )


# ---------------------------------------------------------------------------
# Page structure
# ---------------------------------------------------------------------------


def test_dashboard_heading_present(page: Page, live_server_url: str) -> None:
    """The h1 'Dashboard' heading is visible."""
    _stub_empty(page)
    page.goto(f"{live_server_url}/")
    expect(page.locator("h1")).to_contain_text("Dashboard")


def test_four_stat_cards_present(page: Page, live_server_url: str) -> None:
    """All four stat-card headings are rendered."""
    _stub_empty(page)
    page.goto(f"{live_server_url}/")
    page.wait_for_load_state("networkidle")

    expect(page.get_by_text("Total Predictions")).to_be_visible()
    expect(page.get_by_text("Active Model")).to_be_visible()
    expect(page.get_by_text("Avg Confidence")).to_be_visible()
    expect(page.get_by_text("Models Loaded")).to_be_visible()


def test_stat_cards_show_zero_for_empty_data(page: Page, live_server_url: str) -> None:
    """Stats cards show 0/No model when API returns empty data."""
    _stub_empty(page)
    page.goto(f"{live_server_url}/")
    page.wait_for_load_state("networkidle")

    # Total Predictions should be 0
    total = page.locator("p.text-sm.font-medium:has-text('Total Predictions') + p.text-3xl")
    expect(total).to_have_text("0", timeout=8_000)

    # Models Loaded should be 0
    models_loaded = page.locator("p.text-sm.font-medium:has-text('Models Loaded') + p.text-3xl")
    expect(models_loaded).to_have_text("0", timeout=8_000)


def test_active_model_card_shows_none(page: Page, live_server_url: str) -> None:
    """The Active Model stat card shows 'None' when no model is loaded."""
    _stub_empty(page)
    page.goto(f"{live_server_url}/")
    page.wait_for_load_state("networkidle")

    active_card = page.locator("p.text-sm.font-medium:has-text('Active Model') + p.text-xl")
    expect(active_card).to_have_text("None", timeout=8_000)


# ---------------------------------------------------------------------------
# Model selector sidebar
# ---------------------------------------------------------------------------


def test_model_selector_shows_loaded_model(page: Page, live_server_url: str) -> None:
    """The Active Model sidebar card shows the model name when one is loaded."""
    page.route(
        "**/api/models",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(MODELS_ONE_LOADED),
        ),
    )
    page.route(
        "**/api/analytics/summary",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(ANALYTICS_SUMMARY_EMPTY),
        ),
    )
    page.route(
        "**/api/history*",
        lambda route: route.fulfill(
            status=200,
            content_type="text/html",
            body="<p></p>",
        ),
    )
    page.goto(f"{live_server_url}/")
    page.wait_for_load_state("networkidle")

    # The model card in the sidebar shows the model name
    expect(page.get_by_text("alzheimer_cnn_v1")).to_be_visible(timeout=8_000)


# ---------------------------------------------------------------------------
# Upload zone
# ---------------------------------------------------------------------------


def test_quick_prediction_upload_zone_present(page: Page, live_server_url: str) -> None:
    """The Quick Prediction card has the 'Click to upload' upload zone."""
    _stub_empty(page)
    page.goto(f"{live_server_url}/")
    expect(page.get_by_text("Click to upload or drag and drop")).to_be_visible()


# ---------------------------------------------------------------------------
# Quick Actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "href"),
    [
        ("Batch Upload", "/batch"),
        ("Compare Models", "/compare"),
        ("View Analytics", "/analytics"),
        ("View History", "/history"),
    ],
)
def test_quick_action_links(page: Page, live_server_url: str, label: str, href: str) -> None:
    """Each Quick Actions link is present and points to the correct route."""
    _stub_empty(page)
    page.goto(f"{live_server_url}/")
    link = page.locator(f"a[href='{href}']").filter(has_text=label)
    expect(link).to_be_visible()
