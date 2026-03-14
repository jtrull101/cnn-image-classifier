"""E2E tests for the Training page (/train)."""

import pytest
from playwright.sync_api import Page, expect


pytestmark = [pytest.mark.e2e, pytest.mark.serial]


def test_training_page_loads(page: Page, live_server_url: str) -> None:
    """The /train route returns HTTP 200 and loads successfully."""
    response = page.goto(f"{live_server_url}/train")
    assert response is not None
    assert response.status == 200


def test_training_page_title_contains_train(page: Page, live_server_url: str) -> None:
    """The page <title> includes 'Train'."""
    page.goto(f"{live_server_url}/train")
    assert "Train" in page.title()


def test_training_page_navbar_rendered(page: Page, live_server_url: str) -> None:
    """The navbar is rendered on the training page."""
    page.goto(f"{live_server_url}/train")
    expect(page.locator("nav")).to_be_visible()
