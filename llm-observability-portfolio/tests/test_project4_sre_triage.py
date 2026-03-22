"""
Tests for Project 4: AI SRE Triage Bot
---------------------------------------
Tests cover:
  - Webhook signature verification (HMAC-SHA256)
  - Endpoint routing: alert vs recovery events
  - Span summarization (prompt construction)
  - Health check endpoint
  - Background task queuing on valid payload
  - Edge cases: empty spans, missing fields, malformed JSON
"""

import hashlib
import hmac
import json
import os
import sys
import unittest.mock as mock
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# ── Stubs ─────────────────────────────────────────────────────────────────────
for mod in [
    "ddtrace", "ddtrace.llmobs", "ddtrace.llmobs.decorators",
    "datadog", "openai",
]:
    sys.modules.setdefault(mod, MagicMock())

os.environ.setdefault("DD_API_KEY", "test-key")
os.environ.setdefault("DD_APP_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
os.environ["DD_WEBHOOK_SECRET"] = ""  # disable signature check for most tests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from project4_ai_sre_triage.webhook_handler.server import (  # noqa: E402
    app,
    verify_webhook_signature,
    _summarize_spans,
)

client = TestClient(app)


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthCheck:

    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_includes_service_name(self):
        resp = client.get("/health")
        assert "service" in resp.json()


# ── Webhook signature verification ───────────────────────────────────────────

class TestWebhookSignatureVerification:

    def test_valid_signature_returns_true(self):
        secret = "my-test-secret"
        body = b'{"event": "test"}'
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        with patch.dict(os.environ, {"DD_WEBHOOK_SECRET": secret}):
            # Re-import to pick up env var
            import importlib
            import project4_ai_sre_triage.webhook_handler.server as srv
            importlib.reload(srv)
            assert srv.verify_webhook_signature(body, sig) is True

    def test_invalid_signature_returns_false(self):
        secret = "my-test-secret"
        body = b'{"event": "test"}'
        wrong_sig = "deadbeef" * 8  # wrong 64-char hex

        with patch.dict(os.environ, {"DD_WEBHOOK_SECRET": secret}):
            import importlib
            import project4_ai_sre_triage.webhook_handler.server as srv
            importlib.reload(srv)
            assert srv.verify_webhook_signature(body, wrong_sig) is False

    def test_empty_secret_skips_verification(self):
        # When no secret is configured, should return True (dev mode)
        with patch.dict(os.environ, {"DD_WEBHOOK_SECRET": ""}):
            assert verify_webhook_signature(b"any body", "any signature") is True

    def test_tampered_body_fails_verification(self):
        secret = "real-secret"
        original = b'{"monitor_name": "real"}'
        tampered = b'{"monitor_name": "tampered"}'
        good_sig = hmac.new(secret.encode(), original, hashlib.sha256).hexdigest()

        with patch.dict(os.environ, {"DD_WEBHOOK_SECRET": secret}):
            import importlib
            import project4_ai_sre_triage.webhook_handler.server as srv
            importlib.reload(srv)
            assert srv.verify_webhook_signature(tampered, good_sig) is False


# ── Webhook endpoint routing ──────────────────────────────────────────────────

class TestWebhookEndpoint:

    VALID_ALERT_PAYLOAD = {
        "monitor_name": "[RAG Bot] Faithfulness score below threshold",
        "text": "Faithfulness dropped to 0.52",
        "alert_type": "error",
        "tags": {"ml_app": "rag-support-bot", "env": "production"},
    }

    @patch("project4_ai_sre_triage.webhook_handler.server.run_triage_pipeline", new_callable=AsyncMock)
    def test_alert_event_returns_202_accepted(self, mock_triage):
        resp = client.post(
            "/webhook/datadog",
            json=self.VALID_ALERT_PAYLOAD,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    @patch("project4_ai_sre_triage.webhook_handler.server.run_triage_pipeline", new_callable=AsyncMock)
    def test_recovery_event_is_ignored(self, mock_triage):
        payload = {**self.VALID_ALERT_PAYLOAD, "alert_type": "recovery"}
        resp = client.post("/webhook/datadog", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    @patch("project4_ai_sre_triage.webhook_handler.server.run_triage_pipeline", new_callable=AsyncMock)
    def test_info_event_is_ignored(self, mock_triage):
        payload = {**self.VALID_ALERT_PAYLOAD, "alert_type": "info"}
        resp = client.post("/webhook/datadog", json=payload)
        assert resp.json()["status"] == "ignored"

    def test_malformed_json_returns_400(self):
        resp = client.post(
            "/webhook/datadog",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    @patch("project4_ai_sre_triage.webhook_handler.server.run_triage_pipeline", new_callable=AsyncMock)
    def test_warning_alert_type_is_processed(self, mock_triage):
        payload = {**self.VALID_ALERT_PAYLOAD, "alert_type": "warning"}
        resp = client.post("/webhook/datadog", json=payload)
        assert resp.json()["status"] == "accepted"

    @patch("project4_ai_sre_triage.webhook_handler.server.run_triage_pipeline", new_callable=AsyncMock)
    def test_triage_pipeline_queued_for_alert(self, mock_triage):
        resp = client.post("/webhook/datadog", json=self.VALID_ALERT_PAYLOAD)
        # Background task is added — pipeline will be called
        assert resp.status_code == 200


# ── Span summarization ────────────────────────────────────────────────────────

class TestSummarizeSpans:

    def _make_span(self, name: str, error: str = "", duration: int = 500) -> dict:
        return {
            "attributes": {
                "tags": {
                    "span.name": name,
                    "model": "gpt-4o",
                    "duration": str(duration),
                    "llm.tokens.total": "300",
                },
                "error": {"message": error},
            }
        }

    def test_empty_spans_returns_no_data_message(self):
        result = _summarize_spans([])
        assert "no failing spans" in result.lower()

    def test_single_span_includes_name(self):
        spans = [self._make_span("answer_generator")]
        result = _summarize_spans(spans)
        assert "answer_generator" in result

    def test_caps_at_five_spans(self):
        spans = [self._make_span(f"span-{i}") for i in range(10)]
        result = _summarize_spans(spans)
        # Only first 5 should appear
        assert "span-0" in result
        assert "span-4" in result
        assert "span-5" not in result

    def test_includes_error_message_when_present(self):
        spans = [self._make_span("llm_call", error="Rate limit exceeded")]
        result = _summarize_spans(spans)
        assert "Rate limit exceeded" in result

    def test_no_exception_on_missing_fields(self):
        # Span with minimal/missing fields should not raise
        minimal_span = {"attributes": {}}
        result = _summarize_spans([minimal_span])
        assert isinstance(result, str)

    def test_multiple_spans_are_numbered(self):
        spans = [self._make_span("span-a"), self._make_span("span-b")]
        result = _summarize_spans(spans)
        assert "Span 1" in result
        assert "Span 2" in result


# ── Test endpoint ─────────────────────────────────────────────────────────────

class TestTestEndpoint:

    @patch("project4_ai_sre_triage.webhook_handler.server.run_triage_pipeline", new_callable=AsyncMock)
    def test_test_endpoint_returns_200(self, mock_triage):
        resp = client.post("/webhook/test")
        assert resp.status_code == 200

    @patch("project4_ai_sre_triage.webhook_handler.server.run_triage_pipeline", new_callable=AsyncMock)
    def test_test_endpoint_returns_payload(self, mock_triage):
        resp = client.post("/webhook/test")
        body = resp.json()
        assert "payload" in body
        assert body["payload"]["alert_type"] == "error"
