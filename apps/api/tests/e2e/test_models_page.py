"""E2E tests for the Model Management page (/models-ui)."""

import json
import re

import pytest
from playwright.sync_api import Page, expect

from fixtures.api_responses import (
    DISCOVER_MODELS_RESPONSE,
    MODELS_EMPTY,
    MODELS_ONE_LOADED,
)


pytestmark = [pytest.mark.e2e, pytest.mark.serial]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _goto_models(page: Page, live_server_url: str, models_stub: dict) -> None:
    """Navigate to /models-ui with a given /api/models stub pre-routed."""
    page.route(
        "**/api/models",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(models_stub),
        ),
    )
    page.goto(f"{live_server_url}/models-ui")
    page.wait_for_load_state("networkidle")
    # Wait for Alpine.js to initialize and render x-if / x-for templates
    page.wait_for_function("window.Alpine !== undefined", timeout=10_000)


# ---------------------------------------------------------------------------
# Page structure
# ---------------------------------------------------------------------------


def test_models_page_heading_present(page: Page, live_server_url: str) -> None:
    """The 'Model Management' h1 heading is visible."""
    _goto_models(page, live_server_url, MODELS_EMPTY)
    expect(page.locator("h1")).to_contain_text("Model Management")


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


def test_empty_state_shows_no_models_text(page: Page, live_server_url: str) -> None:
    """When no models are loaded, the empty-state message is shown."""
    _goto_models(page, live_server_url, MODELS_EMPTY)
    expect(page.get_by_text("No models loaded")).to_be_visible(timeout=8_000)


# ---------------------------------------------------------------------------
# One loaded model
# ---------------------------------------------------------------------------


def test_loaded_model_card_shows_name(page: Page, live_server_url: str) -> None:
    """The model card displays the model name."""
    _goto_models(page, live_server_url, MODELS_ONE_LOADED)
    expect(page.get_by_text("alzheimer_cnn_v1")).to_be_visible(timeout=8_000)


def test_loaded_model_card_shows_num_classes(page: Page, live_server_url: str) -> None:
    """The model card displays class count."""
    _goto_models(page, live_server_url, MODELS_ONE_LOADED)
    expect(page.get_by_text("4")).to_be_visible(timeout=8_000)


def test_loaded_model_card_shows_accuracy(page: Page, live_server_url: str) -> None:
    """The model card displays the accuracy percentage."""
    _goto_models(page, live_server_url, MODELS_ONE_LOADED)
    # Accuracy 0.947 → "94.7%"
    expect(page.get_by_text("94.7%")).to_be_visible(timeout=8_000)


def test_loaded_model_card_shows_input_shape(page: Page, live_server_url: str) -> None:
    """The model card displays the input shape."""
    _goto_models(page, live_server_url, MODELS_ONE_LOADED)
    # Input shape [176, 208, 3] → "176 × 208 × 3"
    expect(page.get_by_text("176 × 208 × 3")).to_be_visible(timeout=8_000)


def test_active_model_has_active_badge(page: Page, live_server_url: str) -> None:
    """The current active model's card displays the '✓ Active' badge."""
    _goto_models(page, live_server_url, MODELS_ONE_LOADED)
    expect(page.get_by_text("✓ Active")).to_be_visible(timeout=8_000)


def test_activate_button_disabled_for_current_model(page: Page, live_server_url: str) -> None:
    """The Activate button is disabled when the model is already the active model."""
    _goto_models(page, live_server_url, MODELS_ONE_LOADED)
    # The activate button for the current model should be disabled
    activate_btn = page.locator("button:has-text('Activate')").first
    expect(activate_btn).to_be_disabled(timeout=8_000)


# ---------------------------------------------------------------------------
# Load model panel
# ---------------------------------------------------------------------------


def test_load_model_path_input_present(page: Page, live_server_url: str) -> None:
    """The Load Model path input is present."""
    _goto_models(page, live_server_url, MODELS_EMPTY)
    expect(page.locator("#model-path")).to_be_visible()


def test_load_model_button_disabled_when_input_empty(page: Page, live_server_url: str) -> None:
    """The Load Model button is disabled when the path input is empty."""
    _goto_models(page, live_server_url, MODELS_EMPTY)
    load_btn = page.locator("button:has-text('Load Model')")
    expect(load_btn).to_be_disabled(timeout=8_000)


def test_load_model_button_enables_after_filling_path(page: Page, live_server_url: str) -> None:
    """The Load Model button becomes enabled after a path is entered."""
    _goto_models(page, live_server_url, MODELS_EMPTY)
    page.fill("#model-path", "/path/to/model.keras")
    load_btn = page.locator("button:has-text('Load Model')")
    expect(load_btn).to_be_enabled(timeout=4_000)


# ---------------------------------------------------------------------------
# Discover models
# ---------------------------------------------------------------------------


def test_discover_button_present(page: Page, live_server_url: str) -> None:
    """The 'Discover Models' button is visible."""
    _goto_models(page, live_server_url, MODELS_EMPTY)
    # Use role=button to avoid strict mode violation (tips section also has "Discover Models" text)
    expect(page.get_by_role("button", name=re.compile(r"Discover Models"))).to_be_visible()


def test_discover_button_triggers_api_call(page: Page, live_server_url: str) -> None:
    """Clicking Discover Models fires a request to /api/models/discover."""
    _goto_models(page, live_server_url, MODELS_EMPTY)

    discover_called: list[bool] = []

    def handle_discover(route) -> None:  # noqa: ANN001
        discover_called.append(True)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(DISCOVER_MODELS_RESPONSE),
        )

    page.route("**/api/models/discover", handle_discover)
    page.get_by_role("button", name=re.compile(r"Discover Models")).click()
    page.wait_for_load_state("networkidle")
    assert discover_called, "Expected /api/models/discover to have been requested"


# ---------------------------------------------------------------------------
# Unload confirmation dialog
# ---------------------------------------------------------------------------


def test_unload_button_opens_confirmation_dialog(page: Page, live_server_url: str) -> None:
    """Clicking the Unload button shows the confirmation dialog with the model name."""
    _goto_models(page, live_server_url, MODELS_ONE_LOADED)
    unload_btn = page.locator("button:has-text('Unload')").first
    unload_btn.click()
    # Dialog should show the model name in bold
    dialog_text = page.locator("text=Unload Model?")
    expect(dialog_text).to_be_visible(timeout=4_000)
    expect(page.get_by_text("alzheimer_cnn_v1", exact=False)).to_be_visible()


def test_cancel_button_dismisses_unload_dialog(page: Page, live_server_url: str) -> None:
    """Clicking Cancel in the unload dialog hides it."""
    _goto_models(page, live_server_url, MODELS_ONE_LOADED)
    page.locator("button:has-text('Unload')").first.click()
    expect(page.locator("text=Unload Model?")).to_be_visible()

    page.locator("button:has-text('Cancel')").click()
    expect(page.locator("text=Unload Model?")).to_be_hidden(timeout=4_000)
