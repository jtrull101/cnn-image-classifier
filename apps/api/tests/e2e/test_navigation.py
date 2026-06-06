"""E2E tests for shared navigation component (navbar.html)."""

import json
import re

import pytest
from fixtures.api_responses import MODELS_EMPTY
from playwright.sync_api import Page, expect


pytestmark = [pytest.mark.e2e, pytest.mark.serial]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_models_empty(page: Page) -> None:
    """Route /api/models to return an empty model list."""
    page.route(
        "**/api/models",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(MODELS_EMPTY),
        ),
    )


# ---------------------------------------------------------------------------
# Navbar presence on each main page
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/", "/predict", "/models-ui", "/train"],
)
def test_navbar_renders_on_page(page: Page, live_server_url: str, path: str) -> None:
    """Navbar <nav> element is present on every UI page."""
    _stub_models_empty(page)
    page.goto(f"{live_server_url}{path}")
    expect(page.locator("nav")).to_be_visible()


# ---------------------------------------------------------------------------
# Active link highlighting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "link_href"),
    [
        ("/", "/"),
        ("/predict", "/predict"),
        ("/models-ui", "/models-ui"),
        ("/train", "/train"),
    ],
)
def test_active_nav_link_has_highlight_class(
    page: Page, live_server_url: str, path: str, link_href: str
) -> None:
    """The nav link matching the current page has bg-primary-100 in its class."""
    _stub_models_empty(page)
    page.goto(f"{live_server_url}{path}")
    # The active link is the desktop-nav anchor with the exact href
    active_link = page.locator(f"nav .hidden.md\\:flex a[href='{link_href}']")
    expect(active_link).to_have_class(re.compile(r"bg-primary-100"))


# ---------------------------------------------------------------------------
# Active-model display
# ---------------------------------------------------------------------------


def test_active_model_shows_no_model_when_none_loaded(page: Page, live_server_url: str) -> None:
    """The active-model status span shows 'No model' when no model is loaded."""
    _stub_models_empty(page)
    page.goto(f"{live_server_url}/")
    page.wait_for_load_state("networkidle")
    # The x-text span reads $store.app.activeModel || 'No model'
    model_span = page.locator("nav .hidden.lg\\:flex span[x-text]").first
    expect(model_span).to_be_visible()
    # After Alpine initialises, text should be 'No model'
    expect(model_span).to_have_text("No model", timeout=8_000)


# ---------------------------------------------------------------------------
# WebSocket status indicator
# ---------------------------------------------------------------------------


def test_ws_indicator_shows_connected(page: Page, live_server_url: str) -> None:
    """The WS dot acquires bg-green-500 after the WebSocket connects."""
    _stub_models_empty(page)
    page.goto(f"{live_server_url}/")
    # Wait for the indicator dot to turn green (WS connected)
    ws_dot = page.locator("nav .w-3.h-3.rounded-full").first
    expect(ws_dot).to_have_class(re.compile(r"bg-green-500"), timeout=12_000)


def test_ws_status_text_shows_connected(page: Page, live_server_url: str) -> None:
    """The WS status text span reads 'Connected' after the WebSocket connects."""
    _stub_models_empty(page)
    page.goto(f"{live_server_url}/")
    ws_text = page.locator("nav span.text-xs").filter(has_text="Connected")
    expect(ws_text).to_be_visible(timeout=12_000)


# ---------------------------------------------------------------------------
# Mobile hamburger menu
# ---------------------------------------------------------------------------


def test_mobile_hamburger_button_is_present(page: Page, live_server_url: str) -> None:
    """The hamburger button exists in the DOM (visible at mobile widths)."""
    _stub_models_empty(page)
    # Set a mobile viewport so md:hidden is respected
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(f"{live_server_url}/")
    hamburger = page.locator('button[aria-label="Toggle navigation menu"]')
    expect(hamburger).to_be_visible()


def test_mobile_hamburger_toggles_menu(page: Page, live_server_url: str) -> None:
    """Clicking the hamburger button shows the mobile menu; clicking again hides it."""
    _stub_models_empty(page)
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(f"{live_server_url}/")
    page.wait_for_load_state("domcontentloaded")

    hamburger = page.locator('button[aria-label="Toggle navigation menu"]')
    mobile_menu = page.locator("#mobile-menu")

    # Menu starts hidden (x-show="mobileMenuOpen" → false)
    expect(mobile_menu).to_be_hidden()

    # Open menu
    hamburger.click()
    expect(mobile_menu).to_be_visible()

    # Close menu
    hamburger.click()
    expect(mobile_menu).to_be_hidden()
