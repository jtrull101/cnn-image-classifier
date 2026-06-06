"""E2E tests for the Single Image Prediction page (/predict)."""

import json

import pytest
from fixtures.api_responses import (
    MODELS_EMPTY,
    MODELS_ONE_LOADED,
    PREDICTION_RESULT,
)
from fixtures.test_image import create_png_1x1
from playwright.sync_api import Page, expect


pytestmark = [pytest.mark.e2e, pytest.mark.serial]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _goto_predict(page: Page, live_server_url: str, models_stub: dict) -> None:
    page.route(
        "**/api/models",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(models_stub),
        ),
    )
    page.goto(f"{live_server_url}/predict")
    page.wait_for_load_state("networkidle")
    # Wait for Alpine.js to initialize and render x-if / x-for templates
    page.wait_for_function("window.Alpine !== undefined", timeout=10_000)


def _upload_test_image(page: Page, tmp_path) -> None:
    """Write a minimal PNG to disk and set it on the hidden file input."""
    img_path = tmp_path / "test.png"
    img_path.write_bytes(create_png_1x1())
    page.locator("input[type='file'][accept='image/*']").set_input_files(str(img_path))


# ---------------------------------------------------------------------------
# Page structure
# ---------------------------------------------------------------------------


def test_predict_heading_present(page: Page, live_server_url: str) -> None:
    """The 'Single Image Prediction' h1 is visible."""
    _goto_predict(page, live_server_url, MODELS_EMPTY)
    expect(page.locator("h1")).to_contain_text("Single Image Prediction")


def test_upload_zone_visible(page: Page, live_server_url: str) -> None:
    """The upload drop zone is visible on the page."""
    _goto_predict(page, live_server_url, MODELS_EMPTY)
    expect(page.get_by_text("Click to upload or drag & drop")).to_be_visible(timeout=10_000)


def test_no_models_warning_shows_link_to_models_ui(page: Page, live_server_url: str) -> None:
    """When no models are loaded, a link to /models-ui is shown."""
    _goto_predict(page, live_server_url, MODELS_EMPTY)
    page.wait_for_load_state("networkidle")
    link = page.locator("a[href='/models-ui']").filter(has_text="Load one")
    expect(link).to_be_visible(timeout=8_000)


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------


def test_model_select_has_options_when_model_loaded(page: Page, live_server_url: str) -> None:
    """The model <select> has an option for the loaded model."""
    _goto_predict(page, live_server_url, MODELS_ONE_LOADED)
    page.wait_for_load_state("networkidle")
    option = page.locator("select option[value='alzheimer_cnn_v1']")
    expect(option).to_be_attached(timeout=8_000)


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


def test_file_upload_shows_preview_image(page: Page, live_server_url: str, tmp_path) -> None:
    """Uploading a file renders a preview <img> tag in the upload zone."""
    _goto_predict(page, live_server_url, MODELS_EMPTY)
    _upload_test_image(page, tmp_path)
    # The preview img should appear inside the upload zone
    expect(page.locator(".upload-zone img")).to_be_visible(timeout=6_000)


def test_clear_button_resets_preview(page: Page, live_server_url: str, tmp_path) -> None:
    """Clicking 'Clear' removes the preview image and restores the placeholder."""
    _goto_predict(page, live_server_url, MODELS_EMPTY)
    _upload_test_image(page, tmp_path)
    expect(page.locator(".upload-zone img")).to_be_visible(timeout=6_000)

    page.locator("button:has-text('Clear')").click()
    expect(page.locator(".upload-zone img")).to_be_hidden(timeout=4_000)
    expect(page.get_by_text("Click to upload or drag & drop")).to_be_visible()


# ---------------------------------------------------------------------------
# Classify button state
# ---------------------------------------------------------------------------


def test_classify_button_disabled_without_model_selection(
    page: Page, live_server_url: str, tmp_path
) -> None:
    """Classify Image button stays disabled when no model is selected."""
    _goto_predict(page, live_server_url, MODELS_ONE_LOADED)
    _upload_test_image(page, tmp_path)
    # Deselect model (select the blank option)
    page.select_option("select", value="")
    classify_btn = page.locator("button:has-text('Classify Image')")
    expect(classify_btn).to_be_disabled(timeout=4_000)


def test_classify_button_enabled_with_file_and_model(
    page: Page, live_server_url: str, tmp_path
) -> None:
    """Classify Image button is enabled when both file and model are selected."""
    _goto_predict(page, live_server_url, MODELS_ONE_LOADED)
    page.wait_for_load_state("networkidle")
    _upload_test_image(page, tmp_path)
    # Select the model
    page.select_option("select", value="alzheimer_cnn_v1")
    classify_btn = page.locator("button:has-text('Classify Image')")
    expect(classify_btn).to_be_enabled(timeout=4_000)


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------


def test_predict_result_card_visible_after_successful_classify(
    page: Page, live_server_url: str, tmp_path
) -> None:
    """After a successful prediction, the Result card is visible."""
    _goto_predict(page, live_server_url, MODELS_ONE_LOADED)
    page.route(
        "**/api/predict*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(PREDICTION_RESULT),
        ),
    )
    page.wait_for_load_state("networkidle")
    _upload_test_image(page, tmp_path)
    page.select_option("select", value="alzheimer_cnn_v1")
    page.locator("button:has-text('Classify Image')").click()

    result_card = page.locator(".card").filter(has_text="Result").first
    expect(result_card).to_be_visible(timeout=10_000)


def test_predict_result_shows_class_name(page: Page, live_server_url: str, tmp_path) -> None:
    """The result card displays the predicted class name."""
    _goto_predict(page, live_server_url, MODELS_ONE_LOADED)
    page.route(
        "**/api/predict*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(PREDICTION_RESULT),
        ),
    )
    page.wait_for_load_state("networkidle")
    _upload_test_image(page, tmp_path)
    page.select_option("select", value="alzheimer_cnn_v1")
    page.locator("button:has-text('Classify Image')").click()

    expect(page.get_by_text("NonDemented")).to_be_visible(timeout=10_000)


def test_predict_result_shows_confidence_bar(page: Page, live_server_url: str, tmp_path) -> None:
    """The result card renders the confidence bar element."""
    _goto_predict(page, live_server_url, MODELS_ONE_LOADED)
    page.route(
        "**/api/predict*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(PREDICTION_RESULT),
        ),
    )
    page.wait_for_load_state("networkidle")
    _upload_test_image(page, tmp_path)
    page.select_option("select", value="alzheimer_cnn_v1")
    page.locator("button:has-text('Classify Image')").click()

    expect(page.locator(".confidence-bar")).to_be_visible(timeout=10_000)


def test_predict_result_shows_probability_rows(page: Page, live_server_url: str, tmp_path) -> None:
    """All four class probability rows are rendered in the result card."""
    _goto_predict(page, live_server_url, MODELS_ONE_LOADED)
    page.route(
        "**/api/predict*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(PREDICTION_RESULT),
        ),
    )
    page.wait_for_load_state("networkidle")
    _upload_test_image(page, tmp_path)
    page.select_option("select", value="alzheimer_cnn_v1")
    page.locator("button:has-text('Classify Image')").click()

    # Wait for result to render
    expect(page.get_by_text("NonDemented")).to_be_visible(timeout=10_000)

    for class_name in PREDICTION_RESULT["probabilities"]:
        expect(page.get_by_text(class_name)).to_be_visible(timeout=4_000)


def test_classify_another_clears_result(page: Page, live_server_url: str, tmp_path) -> None:
    """Clicking 'Classify another image' clears the result card."""
    _goto_predict(page, live_server_url, MODELS_ONE_LOADED)
    page.route(
        "**/api/predict*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(PREDICTION_RESULT),
        ),
    )
    page.wait_for_load_state("networkidle")
    _upload_test_image(page, tmp_path)
    page.select_option("select", value="alzheimer_cnn_v1")
    page.locator("button:has-text('Classify Image')").click()
    expect(page.get_by_text("NonDemented")).to_be_visible(timeout=10_000)

    page.locator("button:has-text('Classify another image')").click()
    expect(page.locator(".card").filter(has_text="Result")).to_be_hidden(timeout=4_000)
    expect(page.get_by_text("Click to upload or drag & drop")).to_be_visible()


def test_predict_error_banner_shown_on_api_failure(
    page: Page, live_server_url: str, tmp_path
) -> None:
    """An error banner appears when the predict API returns a 500."""
    _goto_predict(page, live_server_url, MODELS_ONE_LOADED)
    page.route(
        "**/api/predict*",
        lambda route: route.fulfill(
            status=500,
            content_type="application/json",
            body=json.dumps({"detail": "Internal server error"}),
        ),
    )
    page.wait_for_load_state("networkidle")
    _upload_test_image(page, tmp_path)
    page.select_option("select", value="alzheimer_cnn_v1")
    page.locator("button:has-text('Classify Image')").click()

    # Error div has x-show="error" and x-text="error"
    error_div = page.locator("[x-show='error']").filter(has_text="Internal server error")
    expect(error_div).to_be_visible(timeout=8_000)


# ---------------------------------------------------------------------------
# WebSocket Live badge
# ---------------------------------------------------------------------------


def test_ws_live_badge_visible_when_connected(page: Page, live_server_url: str) -> None:
    """The ⚡ Live badge is shown in the upload card when WS is connected."""
    _goto_predict(page, live_server_url, MODELS_EMPTY)
    # The badge is rendered with x-show="wsConnected" which mirrors WS state
    badge = page.locator("span.badge:has-text('Live')")
    expect(badge).to_be_visible(timeout=12_000)
