"""
Project 2: Eval Score Monitor — watches for quality regressions in production.

This script creates three Datadog monitors via the API:
  1. Faithfulness drop alert (hallucination spike)
  2. Overall pass rate SLO
  3. Anomaly detection on eval score distribution

Run once during deployment setup:
  python metrics/create_monitors.py
"""

import os
import json
import requests

DD_API_KEY = os.environ["DD_API_KEY"]
DD_APP_KEY = os.environ["DD_APP_KEY"]
DD_SITE = os.environ.get("DD_SITE", "datadoghq.com")

HEADERS = {
    "DD-API-KEY": DD_API_KEY,
    "DD-APPLICATION-KEY": DD_APP_KEY,
    "Content-Type": "application/json",
}
BASE_URL = f"https://api.{DD_SITE}/api/v1/monitor"


MONITORS = [
    {
        "name": "[RAG Eval] Faithfulness score below threshold (hallucination risk)",
        "type": "metric alert",
        "query": (
            "avg(last_15m):avg:llm.eval.faithfulness.score"
            "{ml_app:rag-support-bot,env:production} < 0.7"
        ),
        "message": (
            "## Faithfulness Alert — Possible Hallucination Spike\n\n"
            "The average faithfulness score has dropped below 0.7 over the last 15 minutes.\n\n"
            "**Current value:** {{value}}\n\n"
            "**What this means:** The model may be generating answers not grounded in "
            "retrieved context. Common causes:\n"
            "- Prompt template regression (check recent deploys)\n"
            "- Retrieval quality drop (context docs may be irrelevant)\n"
            "- Model behavior change after a provider update\n\n"
            "**Runbook:** https://wiki.internal/runbooks/faithfulness-alert\n\n"
            "@pagerduty-ml-platform @slack-ml-quality-alerts"
        ),
        "thresholds": {"critical": 0.7, "warning": 0.8},
        "options": {
            "notify_no_data": True,
            "no_data_timeframe": 30,
            "renotify_interval": 60,
            "evaluation_delay": 60,
        },
        "tags": ["ml_app:rag-support-bot", "team:ml-platform", "severity:high"],
    },
    {
        "name": "[RAG Eval] Overall pass rate SLO breach",
        "type": "metric alert",
        "query": (
            "sum(last_1h):sum:llm.eval.overall.pass{ml_app:rag-support-bot,env:production}.as_rate() "
            "/ sum:dd.llm.request.count{ml_app:rag-support-bot,env:production}.as_rate() * 100 < 80"
        ),
        "message": (
            "## Quality SLO Alert\n\n"
            "The overall eval pass rate has dropped below 80% (SLO target).\n\n"
            "**Current pass rate:** {{value}}%\n\n"
            "Review the Eval Quality dashboard for breakdown by metric.\n"
            "@slack-ml-quality-alerts"
        ),
        "thresholds": {"critical": 80, "warning": 85},
        "tags": ["ml_app:rag-support-bot", "slo:quality"],
    },
    {
        "name": "[RAG Eval] Relevancy score anomaly detected",
        "type": "metric alert",
        "query": (
            "avg(last_30m):anomalies("
            "avg:llm.eval.relevancy.score{ml_app:rag-support-bot,env:production}, "
            "'adaptive', 2, direction='below', alert_window='last_30m', "
            "interval=300, count_default_zero='true') >= 1"
        ),
        "message": (
            "## Relevancy Anomaly Detected\n\n"
            "Relevancy scores are significantly below their historical baseline.\n\n"
            "This may indicate that incoming questions have shifted in topic "
            "and the knowledge base needs updating.\n"
            "@slack-ml-quality-alerts"
        ),
        "thresholds": {"critical": 1},
        "tags": ["ml_app:rag-support-bot", "team:ml-platform"],
    },
]


def create_monitors() -> None:
    print(f"Creating {len(MONITORS)} monitors on {DD_SITE}...")
    for monitor in MONITORS:
        resp = requests.post(BASE_URL, headers=HEADERS, data=json.dumps(monitor))
        if resp.status_code == 200:
            monitor_id = resp.json()["id"]
            print(f"  ✓ Created: '{monitor['name']}' (id={monitor_id})")
        else:
            print(f"  ✗ Failed:  '{monitor['name']}' — {resp.status_code}: {resp.text}")


if __name__ == "__main__":
    create_monitors()
