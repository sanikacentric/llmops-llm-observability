# conftest.py — shared pytest fixtures and configuration

import os
import sys
import unittest.mock as mock
import pytest


# ── Ensure test environment variables ────────────────────────────────────────
@pytest.fixture(autouse=True)
def env_vars():
    """Set required env vars for all tests."""
    overrides = {
        "DD_API_KEY": "test-dd-api-key",
        "DD_APP_KEY": "test-dd-app-key",
        "DD_SITE": "datadoghq.com",
        "DD_ENV": "test",
        "OPENAI_API_KEY": "test-openai-key",
        "SLACK_WEBHOOK_URL": "https://hooks.slack.com/test",
        "DD_WEBHOOK_SECRET": "",
    }
    with mock.patch.dict(os.environ, overrides):
        yield


# ── Silence noisy loggers during tests ───────────────────────────────────────
@pytest.fixture(autouse=True)
def silence_loggers():
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    yield
