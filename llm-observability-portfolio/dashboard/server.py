"""
dashboard/server.py
────────────────────────────────────────────────────
LLM Observability Dashboard — FastAPI backend
Runs on http://localhost:8080

Endpoints:
  GET  /            → serves the HTML dashboard
  GET  /api/metrics → returns all stored project metrics
  POST /api/run     → triggers all 4 projects in background
  GET  /api/status  → returns current run status
"""

import os
import json
import asyncio
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="LLM Observability Dashboard", version="1.0.0")

# ── Global in-memory metrics store ────────────────────────────────────────────
store: dict[str, Any] = {
    "status": "idle",       # idle | running | done | error
    "last_run": None,
    "error": None,
    "project1": None,
    "project2": None,
    "project3": None,
    "project4": None,
}


# ── Project runners (lazy imports to avoid module-level side effects) ─────────

def _run_project1() -> dict:
    from project1_llm_trace_pipeline.app.rag_support_bot import (
        InMemoryVectorStore, run_rag_pipeline, seed_vector_store,
    )
    questions = [
        "How do I set up Datadog LLM Observability for my OpenAI app?",
        "How can I track token costs in Datadog?",
        "What is prompt template versioning?",
    ]
    vs = InMemoryVectorStore()
    seed_vector_store(vs)

    runs = []
    for q in questions:
        r = run_rag_pipeline(q, vs)
        runs.append({
            "question": q,
            "answer": r.answer[:250],
            "sources": r.sources,
            "tokens": r.tokens_used,
            "latency_ms": round(r.total_latency_ms),
            "rewritten_query": r.rewritten_query,
        })

    avg_lat = sum(r["latency_ms"] for r in runs) / len(runs)
    total_tok = sum(r["tokens"] for r in runs)
    prompt_tok = round(total_tok * 0.85)
    completion_tok = total_tok - prompt_tok

    return {
        "runs": runs,
        "avg_latency_ms": round(avg_lat),
        "total_tokens": total_tok,
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "span_breakdown": {
            "answer_generator":  {"avg_ms": round(avg_lat * 0.70), "model": "gpt-4o"},
            "query_rewriter":    {"avg_ms": round(avg_lat * 0.17), "model": "gpt-4o-mini"},
            "embed_text":        {"avg_ms": round(avg_lat * 0.11), "model": "text-embedding-3-small"},
            "vector_retrieval":  {"avg_ms": round(avg_lat * 0.001), "model": "cosine similarity"},
        },
    }


def _run_project2(p1: dict) -> dict:
    from project2_eval_framework.evaluators.eval_suite import run_eval_suite

    results = []
    for run in p1["runs"][:2]:
        b = run_eval_suite(
            question=run["question"],
            answer=run["answer"],
            context=" ".join(run["sources"]),
            env="development",
        )
        results.append({
            "question": run["question"][:60] + "...",
            "scores": {
                s.metric_name: {
                    "score": round(s.score, 2),
                    "label": s.label.value,
                    "reasoning": s.reasoning,
                }
                for s in b.scores
            },
            "overall_pass": b.overall_pass,
        })

    metrics = ["faithfulness", "relevancy", "completeness"]
    avg = {
        m: round(sum(r["scores"][m]["score"] for r in results) / len(results), 3)
        for m in metrics
    }
    pass_rate = sum(1 for r in results if r["overall_pass"]) / len(results)

    return {
        "results": results,
        "avg_scores": avg,
        "pass_rate": round(pass_rate, 2),
        "total_evaluated": len(results),
    }


def _run_project3() -> dict:
    from project3_cost_latency_optimizer.router.model_router import (
        route_and_respond, simulate_routing_savings,
    )

    queries = [
        "What is an LLM span?",
        "How do I configure LLM Observability for a multi-step LangChain pipeline?",
        "Design a multi-region AI observability architecture with compliance requirements for 50M daily requests.",
    ]

    routing = []
    for q in queries:
        r = route_and_respond(q, env="development")
        routing.append({
            "query": q[:65] + ("..." if len(q) > 65 else ""),
            "complexity": r.complexity.value,
            "model": r.model_used,
            "cost_usd": round(r.cost_usd, 6),
            "latency_ms": round(r.latency_ms),
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
        })

    savings = simulate_routing_savings(
        query_distribution={"simple": 0.60, "moderate": 0.30, "complex": 0.10},
        daily_request_count=10_000,
    )

    dist = {"simple": 0, "moderate": 0, "complex": 0}
    for r in routing:
        dist[r["complexity"]] = dist.get(r["complexity"], 0) + 1

    return {"routing": routing, "savings": savings, "distribution": dist}


async def _run_project4() -> dict:
    from project4_ai_sre_triage.webhook_handler.server import (
        fetch_failing_spans, analyze_with_llm, post_to_slack,
    )

    monitor = "[RAG Bot] Faithfulness score below threshold"
    spans = await fetch_failing_spans(ml_app="rag-support-bot", monitor_name=monitor)
    analysis = await analyze_with_llm(
        monitor, "Faithfulness dropped to 0.52 (threshold: 0.70)", "rag-support-bot", spans,
    )

    slack_sent = False
    slack_error = None
    try:
        await post_to_slack(monitor, analysis, "rag-support-bot", len(spans))
        slack_sent = True
    except Exception as e:
        slack_error = str(e)

    return {
        "analysis": analysis,
        "spans_analyzed": len(spans),
        "slack_sent": slack_sent,
        "slack_error": slack_error,
        "monitor_name": monitor,
        "triggered_at": datetime.now().isoformat(),
    }


def _run_all_background() -> None:
    global store
    store.update({"status": "running", "last_run": datetime.now().isoformat(), "error": None})

    try:
        logger.info("▶  Project 1 — RAG Pipeline")
        store["project1"] = _run_project1()

        logger.info("▶  Project 2 — Eval Framework")
        store["project2"] = _run_project2(store["project1"])

        logger.info("▶  Project 3 — Cost Router")
        store["project3"] = _run_project3()

        logger.info("▶  Project 4 — AI SRE Triage")
        loop = asyncio.new_event_loop()
        store["project4"] = loop.run_until_complete(_run_project4())
        loop.close()

        store["status"] = "done"
        logger.info("✓  All 4 projects completed successfully")

    except Exception as exc:
        store["status"] = "error"
        store["error"] = str(exc)
        logger.error("Run failed: %s", exc, exc_info=True)


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/run")
async def trigger_run() -> JSONResponse:
    if store["status"] == "running":
        return JSONResponse({"status": "already_running"})
    threading.Thread(target=_run_all_background, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.get("/api/metrics")
async def get_metrics() -> JSONResponse:
    return JSONResponse(store)


@app.get("/api/status")
async def get_status() -> JSONResponse:
    return JSONResponse({"status": store["status"], "last_run": store["last_run"]})


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
