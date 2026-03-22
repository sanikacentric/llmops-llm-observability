"""
Project 4: AI SRE Triage Bot — Automated Alert Analysis
--------------------------------------------------------
When a Datadog LLM Observability monitor fires, this system:
  1. Receives the webhook payload from Datadog
  2. Fetches related LLM spans from the Datadog API
  3. Sends span data to Claude/GPT for root-cause analysis
  4. Posts a structured hypothesis to Slack with actionable next steps

This is a lightweight proof-of-concept of the same pattern behind
Datadog Bits AI SRE (GA'd late 2024), built to understand the mechanism.

Architecture:
  Datadog Monitor → Webhook → FastAPI handler → Span fetcher
                                                      ↓
                                              LLM root-cause analyzer
                                                      ↓
                                              Slack notification

Run locally:
  uvicorn webhook_handler.server:app --port 8000
  # Then configure Datadog webhook to hit http://your-host:8000/webhook/datadog
"""

import os
import json
import time
import hmac
import hashlib
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import httpx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI SRE Triage Bot", version="1.0.0")

DD_API_KEY = os.environ["DD_API_KEY"]
DD_APP_KEY = os.environ["DD_APP_KEY"]
DD_SITE = os.environ.get("DD_SITE", "datadoghq.com")
WEBHOOK_SECRET = os.environ.get("DD_WEBHOOK_SECRET", "")  # set in Datadog webhook config
SLACK_WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# ── Webhook payload model ─────────────────────────────────────────────────────
def verify_webhook_signature(body: bytes, signature: str) -> bool:
    """
    Verify the Datadog webhook HMAC signature to prevent spoofing.
    Datadog signs with SHA-256 using the webhook secret.
    """
    if not WEBHOOK_SECRET:
        logger.warning("No WEBHOOK_SECRET set — skipping signature verification (dev mode)")
        return True
    expected = hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ── Datadog API: fetch recent failing spans ───────────────────────────────────
async def fetch_failing_spans(
    ml_app: str,
    monitor_name: str,
    lookback_minutes: int = 15,
) -> list[dict]:
    """
    Query Datadog LLM Observability for recent spans from the alerting app.
    Fetches the 10 most recent failing traces to give the LLM analyst context.
    """
    now_ns = int(time.time() * 1e9)
    from_ns = now_ns - (lookback_minutes * 60 * 1e9)

    # Datadog Spans API v2
    url = f"https://api.{DD_SITE}/api/v2/spans/events/search"
    headers = {
        "DD-API-KEY": DD_API_KEY,
        "DD-APPLICATION-KEY": DD_APP_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "data": {
            "type": "search_request",
            "attributes": {
                "filter": {
                    "query": f"@ml_app:{ml_app} status:error",
                    "from": str(int(from_ns)),
                    "to": str(int(now_ns)),
                },
                "sort": "-timestamp",
                "page": {"limit": 10},
            },
        }
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        logger.error("Span API error: %s %s", resp.status_code, resp.text[:200])
        return []

    spans = resp.json().get("data", [])
    logger.info("Fetched %d failing spans for %s", len(spans), ml_app)
    return spans


# ── LLM analyst ───────────────────────────────────────────────────────────────
TRIAGE_PROMPT = """You are an expert SRE analyzing a Datadog LLM Observability alert.
Your job is to identify the most likely root cause and provide actionable remediation steps.

## Alert Details
Monitor: {monitor_name}
Alert message: {alert_message}
Triggered at: {triggered_at}
ML App: {ml_app}

## Recent Failing Spans (last 15 minutes)
{spans_summary}

## Instructions
1. Identify the most likely root cause (be specific, not generic)
2. List 3 concrete remediation steps in priority order
3. Identify what additional data would help confirm the diagnosis
4. Estimate blast radius (how many users/requests are affected)

Format your response as JSON:
{{
  "root_cause": "<specific hypothesis>",
  "confidence": "high|medium|low",
  "remediation_steps": ["step 1", "step 2", "step 3"],
  "additional_data_needed": "<what else to check>",
  "blast_radius": "<estimate>",
  "summary": "<one sentence for the Slack notification headline>"
}}"""


async def analyze_with_llm(
    monitor_name: str,
    alert_message: str,
    ml_app: str,
    spans: list[dict],
) -> dict:
    """
    Send span data to OpenAI for root-cause analysis.
    Uses gpt-4o for reasoning quality — this is a low-volume, high-stakes call.
    """
    # Summarize spans for the prompt (avoid blowing the context window)
    spans_summary = _summarize_spans(spans)

    prompt = TRIAGE_PROMPT.format(
        monitor_name=monitor_name,
        alert_message=alert_message,
        triggered_at=datetime.now(timezone.utc).isoformat(),
        ml_app=ml_app,
        spans_summary=spans_summary,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 600,
                "response_format": {"type": "json_object"},
            },
        )

    if resp.status_code != 200:
        logger.error("OpenAI error: %s", resp.text[:300])
        return {"summary": "Analysis failed — check logs", "root_cause": "Unknown"}

    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def _summarize_spans(spans: list[dict]) -> str:
    """Convert raw span data into a readable summary for the LLM prompt."""
    if not spans:
        return "No failing spans found in the lookback window."

    lines = []
    for i, span in enumerate(spans[:5]):  # cap at 5 to control prompt length
        attrs = span.get("attributes", {})
        tags = attrs.get("tags", {})
        lines.append(
            f"Span {i+1}: "
            f"name={tags.get('span.name', 'unknown')} "
            f"model={tags.get('model', 'unknown')} "
            f"error_msg={attrs.get('error', {}).get('message', 'none')[:100]} "
            f"duration_ms={tags.get('duration', 'N/A')} "
            f"tokens={tags.get('llm.tokens.total', 'N/A')}"
        )
    return "\n".join(lines)


# ── Slack notifier ────────────────────────────────────────────────────────────
async def post_to_slack(
    monitor_name: str,
    analysis: dict,
    ml_app: str,
    span_count: int,
) -> None:
    """
    Post a structured triage summary to Slack.
    Uses Block Kit for readable formatting.
    """
    confidence_emoji = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢",
    }.get(analysis.get("confidence", "low"), "⚪")

    remediation = analysis.get("remediation_steps", [])
    remediation_text = "\n".join(
        f"{i+1}. {step}" for i, step in enumerate(remediation)
    )

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🚨 AI SRE Triage — {monitor_name}"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{analysis.get('summary', 'Alert triggered')}*\n"
                        f"App: `{ml_app}` | Spans analyzed: {span_count}",
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Root Cause {confidence_emoji}*\n{analysis.get('root_cause', 'Unknown')}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Blast Radius*\n{analysis.get('blast_radius', 'Unknown')}",
                },
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Remediation Steps*\n{remediation_text}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*What else to check*\n{analysis.get('additional_data_needed', 'N/A')}",
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Generated by AI SRE Triage Bot • {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} • Confidence: {analysis.get('confidence', 'unknown')}",
                }
            ],
        },
    ]

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            SLACK_WEBHOOK_URL,
            json={"blocks": blocks, "text": f"Alert: {monitor_name}"},
        )

    if resp.status_code == 200:
        logger.info("Slack notification sent")
    else:
        logger.error("Slack failed: %s %s", resp.status_code, resp.text)


# ── Triage pipeline ───────────────────────────────────────────────────────────
async def run_triage_pipeline(webhook_data: dict) -> None:
    """
    Full triage pipeline — called as a background task so the webhook
    returns 200 immediately (Datadog requires fast acknowledgment).
    """
    monitor_name = webhook_data.get("monitor_name", "Unknown Monitor")
    alert_message = webhook_data.get("text", "")
    ml_app = webhook_data.get("tags", {}).get("ml_app", "rag-support-bot")

    logger.info("Triage pipeline started | monitor='%s' app='%s'", monitor_name, ml_app)

    # 1. Fetch recent failing spans
    spans = await fetch_failing_spans(ml_app=ml_app, monitor_name=monitor_name)

    # 2. Analyze with LLM
    analysis = await analyze_with_llm(
        monitor_name=monitor_name,
        alert_message=alert_message,
        ml_app=ml_app,
        spans=spans,
    )

    logger.info("Analysis complete | root_cause='%s'", analysis.get("root_cause", "?"))

    # 3. Post to Slack
    await post_to_slack(
        monitor_name=monitor_name,
        analysis=analysis,
        ml_app=ml_app,
        span_count=len(spans),
    )


# ── FastAPI endpoints ─────────────────────────────────────────────────────────
@app.post("/webhook/datadog")
async def receive_datadog_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Receives Datadog monitor webhook payloads.
    Acknowledges immediately, processes in background.
    """
    body = await request.body()

    # Verify signature
    signature = request.headers.get("X-Datadog-Signature", "")
    if not verify_webhook_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Only triage alert (not recovery) events
    alert_type = data.get("alert_type", "")
    if alert_type not in ("error", "warning"):
        logger.info("Ignoring non-alert event: %s", alert_type)
        return JSONResponse({"status": "ignored", "reason": "non-alert event"})

    logger.info(
        "Webhook received | monitor='%s' type='%s'",
        data.get("monitor_name", "?"),
        alert_type,
    )

    # Queue triage in background
    background_tasks.add_task(run_triage_pipeline, data)

    return JSONResponse({"status": "accepted", "message": "Triage queued"})


@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "ai-sre-triage-bot"})


@app.post("/webhook/test")
async def test_triage(background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Test endpoint — triggers a simulated alert for local development.
    Call: curl -X POST http://localhost:8000/webhook/test
    """
    test_payload = {
        "monitor_name": "[RAG Bot] Faithfulness score below threshold",
        "text": "Faithfulness score dropped to 0.52 (threshold: 0.70)",
        "alert_type": "error",
        "tags": {"ml_app": "rag-support-bot", "env": "production"},
    }
    background_tasks.add_task(run_triage_pipeline, test_payload)
    return JSONResponse({"status": "test triggered", "payload": test_payload})
