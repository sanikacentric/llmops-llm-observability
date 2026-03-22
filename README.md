# LLM Observability Portfolio
### Datadog LLM Observability — End-to-End Demo Portfolio

A portfolio of four production-grade projects demonstrating end-to-end expertise across
LLM application engineering, observability instrumentation, cost management, and AI-powered
operations — all wired into Datadog LLM Observability.

---

## Table of Contents

- [Quickstart](#quickstart)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Project 1 — LLM Trace Pipeline (RAG Support Bot)](#project-1--llm-trace-pipeline-rag-support-bot)
- [Project 2 — Evaluation Framework (Quality Loop)](#project-2--evaluation-framework-quality-loop)
- [Project 3 — Cost + Latency Optimizer (Model Router)](#project-3--cost--latency-optimizer-model-router)
- [Project 4 — AI SRE Triage Bot (Webhook + Slack)](#project-4--ai-sre-triage-bot-webhook--slack)
- [Run All Projects Together](#run-all-projects-together)
- [Where to See Everything in Datadog](#where-to-see-everything-in-datadog)
- [How to Get Your Slack Webhook URL](#how-to-get-your-slack-webhook-url)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

> The fastest way to run everything — one command installs deps and launches the dashboard.

### Step 1 — Fill in your `.env`

Copy `.env.example` to `.env` and set your credentials:

```bash
DD_API_KEY=your-datadog-api-key
DD_APP_KEY=your-datadog-app-key
DD_SITE=datadoghq.com          # or us5.datadoghq.com — match your Datadog URL
DD_ENV=development

OPENAI_API_KEY=sk-proj-...

SLACK_WEBHOOK_URL=  # see guide below
DD_WEBHOOK_SECRET=your-secret  # optional, only needed for real Datadog webhooks
```

### Step 2 — Run everything

```powershell
python start.py
```

This installs all Python dependencies and starts the dashboard server.

### Step 3 — Open the dashboard

```
http://localhost:8080
```

Click **"Run All Projects"** and watch all 4 projects execute live (~60 seconds).

### Step 4 — See your data in Datadog

After the run completes, open Datadog and navigate to:

```
https://us5.datadoghq.com/llm/traces
```

Filter by `ml_app:rag-support-bot` to see all traces, spans, and eval scores.
*(Replace `us5.datadoghq.com` with `us5.datadoghq.com` if your `DD_SITE` is `us5.datadoghq.com`)*

---

## Architecture Overview

All four projects share the same `ml_app="rag-support-bot"` namespace in Datadog so they
appear unified in the LLM Observability Explorer.

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Question (input)                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  PROJECT 1 — RAG Pipeline                                       │
│  query_rewriter (gpt-4o-mini)                                   │
│    → embed_text (text-embedding-3-small)                        │
│    → vector_retrieval (cosine similarity)                       │
│    → answer_generator (gpt-4o)                                  │
│  Emits: @workflow @llm @task @embedding spans → Datadog         │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  PROJECT 2 — Eval Framework                                     │
│  faithfulness score  (LLM-as-judge via gpt-4o)                  │
│  relevancy score     (LLM-as-judge via gpt-4o)                  │
│  completeness score  (LLM-as-judge via gpt-4o)                  │
│  Emits: llm.eval.* metrics + LLMObs.submit_evaluation()         │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼ (on quality alert)
┌─────────────────────────────────────────────────────────────────┐
│  PROJECT 3 — Cost Router                                        │
│  classify_complexity (gpt-4o-mini)                              │
│    → simple/moderate → gpt-4o-mini                              │
│    → complex         → gpt-4o                                   │
│  Emits: llm.cost.* + llm.routing.* + llm.request.duration_ms   │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼ (Datadog monitor fires)
┌─────────────────────────────────────────────────────────────────┐
│  PROJECT 4 — AI SRE Triage Bot                                  │
│  Datadog monitor → POST /webhook/datadog                        │
│    → fetch failing spans (Datadog Spans API v2)                 │
│    → root-cause analysis (gpt-4o)                               │
│    → Slack Block Kit notification                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Runtime |
| Datadog account | Free tier works | LLM Observability Explorer, Metrics, Monitors |
| OpenAI API key | Any tier | Powers gpt-4o, gpt-4o-mini, text-embedding-3-small |
| Slack workspace | Free tier works | Project 4 alert notifications |

---

## Installation & Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd llm-observability-portfolio

# Create virtual environment (recommended)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure environment variables

Your `.env` file should look like this:

```bash
# Datadog credentials
DD_API_KEY=your-datadog-api-key
DD_APP_KEY=your-datadog-app-key
DD_SITE=us5.datadoghq.com          # change to datadoghq.com / datadoghq.eu if needed
DD_ENV=development

# Disable APM tracer (no local Datadog Agent needed — LLM Observability is agentless)
DD_TRACE_ENABLED=false

# OpenAI
OPENAI_API_KEY=sk-proj-...

# Slack (Project 4)
SLACK_WEBHOOK_URL=


# Datadog webhook HMAC secret (Project 4)
DD_WEBHOOK_SECRET=your-secret
```

> **Note:** `DD_TRACE_ENABLED=false` stops the APM tracer from trying to reach a local
> Datadog Agent on port 8126. LLM Observability uses `agentless_enabled=True` and sends
> data directly to the Datadog API — it is unaffected by this setting.

### 3. How to get each credential

| Credential | Where to get it |
|-----------|----------------|
| `DD_API_KEY` | Datadog → Organization Settings → API Keys → New Key |
| `DD_APP_KEY` | Datadog → Organization Settings → Application Keys → New Key |
| `DD_SITE` | Check your Datadog URL: `us5.datadoghq.com` → `datadoghq.com`, `us5.datadoghq.com` → `us5.datadoghq.com` |
| `OPENAI_API_KEY` | platform.openai.com → API Keys → Create new secret key |
| `SLACK_WEBHOOK_URL` | api.slack.com/apps → Create App → Incoming Webhooks → Add Webhook to Workspace |

---

## Project 1 — LLM Trace Pipeline (RAG Support Bot)

### What it does

A production-style RAG (Retrieval-Augmented Generation) support bot that answers questions
from a knowledge base. Every stage of the pipeline is instrumented with Datadog LLM
Observability decorators, creating a full nested span tree in the Datadog Explorer.

**Pipeline flow:**
```
User Question
  → Query Rewriter     [@llm]       gpt-4o-mini rewrites question for better retrieval
  → Embed Text         [@embedding] text-embedding-3-small converts query to vector
  → Vector Retrieval   [@task]      cosine similarity finds top-3 matching KB docs
  → Answer Generator   [@llm]       gpt-4o generates grounded answer from context
```

**5 knowledge base documents (in-memory vector store):**
- `kb-001` — Datadog LLM Observability overview
- `kb-002` — Instrumentation guide (LangChain, OpenAI, Anthropic)
- `kb-003` — Token cost tracking with DogStatsD
- `kb-004` — LLM evaluations and quality metrics
- `kb-005` — Prompt template versioning

### How to run

```bash
cd llm-observability-portfolio

python -X utf8 -c "
from dotenv import load_dotenv; load_dotenv()
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
exec(open('project1_llm_trace_pipeline/app/rag_support_bot.py').read())
"
```

Or run the individual script directly (after activating venv and exporting env vars):

```bash
python project1_llm_trace_pipeline/app/rag_support_bot.py
```

### What to expect

**Terminal output:**
```
Q: How do I set up Datadog LLM Observability for my OpenAI app?
A: To set up Datadog LLM Observability for your OpenAI app, install ddtrace and call LLMObs.enable()...
   Sources: ['kb-001', 'kb-002', 'kb-003']
   Tokens: 311 | Latency: 2518ms
   Rewritten query: 'Set up Datadog LLM Observability OpenAI app'
```

**Live metrics per run:**
| Metric | Typical value |
|--------|--------------|
| Total tokens per question | 270–320 tokens |
| End-to-end latency | 1800–3000ms |
| Docs retrieved per query | 3 (top-k=3 cosine similarity) |
| Models used | gpt-4o-mini (rewrite) + gpt-4o (answer) |
| Embeddings | text-embedding-3-small (1536 dims) |

### Where to see it in Datadog

1. **LLM Observability Explorer**
   ```
   https://us5.datadoghq.com/llm/traces
   ```
   - Filter by `ml_app:rag-support-bot`
   - Click any trace to see the nested span tree:
     ```
     rag_support_pipeline  (workflow)
       ├── query_rewriter      (llm)       gpt-4o-mini
       ├── embed_text          (embedding) text-embedding-3-small
       ├── vector_retrieval    (task)
       └── answer_generator    (llm)       gpt-4o
     ```
   - Each span shows: prompt, completion, token counts, latency, model name

2. **Import the pre-built dashboard**
   - Datadog → Dashboards → New Dashboard → Import JSON
   - Paste contents of `project1_llm_trace_pipeline/datadog_config/dashboard.json`
   - Shows: token usage, latency percentiles, error rate, cost over time

3. **Import monitors**
   - The file `project1_llm_trace_pipeline/datadog_config/monitors.yaml` contains
     monitor definitions for error rate, latency SLO, and cost threshold alerts

---

## Project 2 — Evaluation Framework (Quality Loop)

### What it does

An automated LLM-as-judge evaluation system that scores every answer the RAG bot produces
on three quality dimensions. Scores are emitted to Datadog as custom metrics and linked
to traces via `LLMObs.submit_evaluation()`, enabling quality SLOs and anomaly detection.

**Three evaluators (all use gpt-4o as judge):**

| Evaluator | Question it answers | Alert threshold |
|-----------|-------------------|----------------|
| `faithfulness` | Is the answer grounded in the retrieved docs? (no hallucination) | < 0.70 |
| `relevancy` | Does the answer directly address the user's question? | < 0.70 |
| `completeness` | Does the answer cover all important aspects? | < 0.70 |

### How to run

```bash
cd llm-observability-portfolio

python -X utf8 -c "
from dotenv import load_dotenv; load_dotenv()
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from project2_eval_framework.evaluators.eval_suite import run_eval_suite

bundle = run_eval_suite(
    question='How do I set up Datadog LLM Observability for my OpenAI app?',
    answer='Install ddtrace and call LLMObs.enable() before any OpenAI calls.',
    context='Datadog LLM Observability provides tracing. Install ddtrace and call LLMObs.enable().',
    env='development',
)
for score in bundle.scores:
    print(f'{score.metric_name}: {score.score:.2f} ({score.label.value})')
print(f'Overall: {\"PASS\" if bundle.overall_pass else \"FAIL\"}')
"
```

To create Datadog monitors for eval quality alerts:

```bash
python project2_eval_framework/metrics/create_monitors.py
```

### What to expect

**Terminal output:**
```
--- Eval 1 ---
faithfulness    ████████████████████  1.00  [PASS]
relevancy       ████████████████████  1.00  [PASS]
completeness    ████████████████████  1.00  [PASS]
Overall: PASS
```

**Score interpretation:**

| Score range | Label | Meaning |
|------------|-------|---------|
| 0.80 – 1.00 | `pass` | High quality answer |
| 0.50 – 0.79 | `partial` | Acceptable but could be improved |
| 0.00 – 0.49 | `fail` | Quality regression — monitor will alert |

**What triggers an alert:** If `faithfulness < 0.70`, a Datadog monitor fires, which
in turn triggers Project 4 (the AI SRE Triage Bot) via webhook.

### Where to see it in Datadog

1. **Evaluations tab on any trace**
   ```
   https://us5.datadoghq.com/llm/traces → click a trace → Evaluations tab
   ```
   Shows faithfulness, relevancy, completeness scores linked to that specific trace.

2. **Metrics Explorer**
   ```
   https://us5.datadoghq.com/metric/explorer
   ```
   Search for:
   - `llm.eval.faithfulness.score`
   - `llm.eval.relevancy.score`
   - `llm.eval.completeness.score`
   - `llm.eval.overall.pass`

3. **Monitors** (after running `create_monitors.py`)
   ```
   https://us5.datadoghq.com/monitors/manage
   ```
   Look for `[RAG Bot] Faithfulness score below threshold`

---

## Project 3 — Cost + Latency Optimizer (Model Router)

### What it does

An intelligent model router that classifies query complexity first (using a fast, cheap
gpt-4o-mini call), then routes to the most cost-effective model capable of answering it.
Tracks per-request cost as Datadog custom metrics and maintains a Datadog SLO on p95 latency.

**Routing logic:**
```
User Query
  → complexity_classifier (gpt-4o-mini, ~$0.000001)
      ├── simple   → gpt-4o-mini  (fast, cheap: ~$0.15/1M tokens input)
      ├── moderate → gpt-4o-mini  (with larger token budget)
      └── complex  → gpt-4o       (premium: ~$2.50/1M tokens input)
  → emit cost + latency metrics to Datadog
```

**Cost pricing table used:**
| Model | Input per 1M tokens | Output per 1M tokens |
|-------|--------------------|--------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |

### How to run

```bash
cd llm-observability-portfolio

python -X utf8 -c "
from dotenv import load_dotenv; load_dotenv()
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
exec(open('project3_cost_latency_optimizer/router/model_router.py').read())
"
```

To set up the Datadog SLO and cost dashboard:

```bash
python project3_cost_latency_optimizer/dashboard/setup_slo_and_dashboard.py
```

### What to expect

**Terminal output:**
```
Query: What is an LLM span?
  Model:      gpt-4o-mini
  Complexity: simple
  Cost:       $0.000094
  Latency:    3335ms

Query: How do I configure LLM Observability for a multi-step LangChain pipeline?
  Model:      gpt-4o
  Complexity: complex
  Cost:       $0.006755
  Latency:    5678ms

Query: Design a multi-region AI observability architecture...
  Model:      gpt-4o
  Complexity: complex
  Cost:       $0.007890
  Latency:    6202ms

Cost simulation (10,000 req/day):
  Baseline (all gpt-4o):  $32.50/day
  With routing:            $5.00/day
  Savings:                 $27.50/day  (84.6%)
  Annual savings:          $10,036
```

**Routing decisions observed in real runs:**

| Query type | Classified as | Routed to | Why |
|-----------|--------------|-----------|-----|
| Definitions, simple how-to | simple | gpt-4o-mini | Factual, no reasoning needed |
| Multi-step setup, comparisons | complex | gpt-4o | Multi-step explanation required |
| Architecture design, open-ended | complex | gpt-4o | Deep reasoning needed |

**Key insight:** At 10,000 req/day with a typical distribution (60% simple, 30% moderate,
10% complex), routing saves **84.6%** on model costs — from $32.50/day down to $5.00/day.

### Where to see it in Datadog

1. **LLM Observability Explorer**
   ```
   https://us5.datadoghq.com/llm/traces
   ```
   Filter by `ml_app:rag-support-bot` → look for `cost_optimized_llm_router` workflow traces.
   Each trace shows the full routing decision: classifier → routed model call.

2. **Metrics Explorer**
   ```
   https://us5.datadoghq.com/metric/explorer
   ```
   Search for:
   - `llm.cost.usd_per_request` — tagged by `model`, `complexity`, `model_tier`
   - `llm.cost.savings_percent` — overall savings from routing
   - `llm.request.duration_ms` — latency histogram by model
   - `llm.routing.simple` / `llm.routing.complex` — routing distribution counters
   - `llm.tokens.prompt` / `llm.tokens.completion` — token usage by tier

3. **SLO dashboard** (after running `setup_slo_and_dashboard.py`)
   ```
   https://us5.datadoghq.com/slo
   ```
   SLO: "99% of requests complete in < 2000ms"

---

## Project 4 — AI SRE Triage Bot (Webhook + Slack)

### What it does

A FastAPI webhook server that receives Datadog monitor alerts, automatically fetches
the relevant failing spans from Datadog, runs GPT-4o root-cause analysis, and posts a
structured triage report to Slack — all within 30 seconds of the alert firing.

**Pipeline:**
```
Datadog monitor alert fires
  → POST /webhook/datadog  (FastAPI acknowledges in < 1s)
  → BackgroundTask starts:
      1. fetch_failing_spans()  — Datadog Spans API v2, last 15 min
      2. analyze_with_llm()     — GPT-4o root-cause analysis (JSON output)
      3. post_to_slack()        — Slack Block Kit formatted message
```

**FastAPI endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| POST | `/webhook/datadog` | Real Datadog webhook receiver (requires HMAC signature) |
| POST | `/webhook/test` | Test endpoint — triggers a simulated alert, no signature needed |

### How to run

**Start the server:**

```bash
cd llm-observability-portfolio

python -X utf8 -c "
from dotenv import load_dotenv; load_dotenv()
import subprocess, sys, os
env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
subprocess.run([sys.executable, '-X', 'utf8', '-m', 'uvicorn',
    'project4_ai_sre_triage.webhook_handler.server:app',
    '--port', '8000', '--host', '0.0.0.0', '--reload'], env=env)
"
```

**Trigger a test triage (in a new terminal):**

```bash
curl -X POST http://localhost:8000/webhook/test
```

**Or run the full pipeline directly in Python:**

```bash
python -X utf8 -c "
from dotenv import load_dotenv; load_dotenv()
import asyncio
from project4_ai_sre_triage.webhook_handler.server import fetch_failing_spans, analyze_with_llm, post_to_slack

async def main():
    spans = await fetch_failing_spans(ml_app='rag-support-bot', monitor_name='[RAG Bot] Faithfulness below threshold')
    analysis = await analyze_with_llm('[RAG Bot] Faithfulness score below threshold', 'Score dropped to 0.52', 'rag-support-bot', spans)
    import json; print(json.dumps(analysis, indent=2))
    await post_to_slack('[RAG Bot] Faithfulness score below threshold', analysis, 'rag-support-bot', len(spans))
    print('Check your Slack channel!')

asyncio.run(main())
"
```

### What to expect

**Terminal (server logs):**
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     POST /webhook/test  200 OK
INFO:     Triage pipeline started | monitor='[RAG Bot] Faithfulness score below threshold'
INFO:     Fetched 0 failing spans for rag-support-bot
INFO:     Analysis complete | root_cause='Recent changes to knowledge base...'
INFO:     Slack notification sent
```

**GPT-4o analysis JSON output:**
```json
{
  "root_cause": "Recent update to knowledge base introduced inaccuracies in retrieved information",
  "confidence": "medium",
  "remediation_steps": [
    "Roll back the knowledge base to the last known stable version",
    "Review recent changes to identify specific inaccuracies",
    "Add stricter validation checks for future KB updates"
  ],
  "additional_data_needed": "Change logs for recent KB updates, error logs from retrieval",
  "blast_radius": "Potentially all users of rag-support-bot",
  "summary": "Faithfulness drop likely due to recent KB update; rollback and review needed"
}
```

**Slack message (Block Kit format):**
```
🚨 AI SRE Triage — [RAG Bot] Faithfulness score below threshold
Faithfulness drop likely due to knowledge base update; rollback needed
App: rag-support-bot | Spans analyzed: 0

Root Cause 🔴                        Blast Radius
Recent KB update caused inaccuracies  All users of rag-support-bot

Remediation Steps
1. Roll back knowledge base to last stable version
2. Review recent changes for inaccuracies
3. Add stricter validation for future KB updates

What else to check
Change logs for KB updates, error logs from retrieval

Generated by AI SRE Triage Bot • 2026-03-14 21:48 UTC • Confidence: medium
```

### Where to see it

| UI | URL | What to look for |
|----|-----|-----------------|
| **Swagger UI** | `http://localhost:8000/docs` | Interactive API docs — try `POST /webhook/test` |
| **Health check** | `http://localhost:8000/health` | `{"status": "ok", "service": "ai-sre-triage-bot"}` |
| **Slack channel** | Your configured channel | Formatted Block Kit triage message | https://llm-sanika.slack.com/ssb/redirect?entry_point=workspace_signin

### Setting up the real Datadog webhook (optional)

To wire Project 4 to a real Datadog monitor:

1. Expose the server publicly: `ngrok http 8000`
2. In Datadog: Integrations → Webhooks → New Webhook
   - URL: `https://your-ngrok-url/webhook/datadog`
   - Secret: set to `DD_WEBHOOK_SECRET` value from `.env`
3. Add the webhook to any Datadog monitor's notification channel
4. When that monitor fires, Project 4 will automatically analyze and post to Slack

---

## Run All Projects Together

The `demo.py` script runs all 4 projects in sequence with rich terminal output:

```bash
# Run all 4 projects
python -X utf8 -c "
from dotenv import load_dotenv; load_dotenv()
import subprocess, sys, os
subprocess.run([sys.executable, '-X', 'utf8', 'demo.py'],
    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
"

# Run a specific project only
python -X utf8 -c "
from dotenv import load_dotenv; load_dotenv()
import subprocess, sys, os
subprocess.run([sys.executable, '-X', 'utf8', 'demo.py', '--project', '3'],
    env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
"

# Dry run (no real API calls — for demos without credentials)
python demo.py --dry-run
```

### Recommended run order

```
1. Project 1  →  generates traces in Datadog to explore
2. Project 2  →  evaluates Project 1 output, links scores to traces
3. Project 3  →  shows cost savings vs always using gpt-4o
4. Project 4  →  start server, then trigger test to see Slack alert
```

---

## Where to See Everything in Datadog

> Replace `us5.datadoghq.com` with your actual `DD_SITE` value in `.env`.
> Common sites: `datadoghq.com` (US1), `us5.datadoghq.com` (US5), `datadoghq.eu` (EU).
> Your site matches the URL you use when you log in to Datadog.

### LLM Traces and Spans

**URL:** `https://us5.datadoghq.com/llm/traces`

This is the main place to see everything. After running the projects:

1. Open the URL above
2. In the search bar type: `ml_app:rag-support-bot`
3. You will see all traces from Project 1 and Project 3
4. Click any trace to expand the full nested span tree:
   ```
   rag_support_pipeline      (workflow)
     ├── query_rewriter       (llm)       gpt-4o-mini
     ├── embed_text           (embedding) text-embedding-3-small
     ├── vector_retrieval     (task)
     └── answer_generator     (llm)       gpt-4o
   ```
5. Click the **Evaluations** tab on any trace to see faithfulness, relevancy, completeness scores from Project 2

### Metrics (costs, latency, eval scores)

**URL:** `https://us5.datadoghq.com/metric/explorer`

Search for these metric names:

| Metric | Source | What it shows |
|--------|--------|---------------|
| `llm.eval.faithfulness.score` | Project 2 | Hallucination score per answer |
| `llm.eval.relevancy.score` | Project 2 | Relevancy score per answer |
| `llm.eval.completeness.score` | Project 2 | Completeness score per answer |
| `llm.cost.usd_per_request` | Project 3 | Per-request cost tagged by model |
| `llm.cost.savings_percent` | Project 3 | Savings from routing vs always gpt-4o |
| `llm.request.duration_ms` | Project 3 | Latency histogram by model |
| `llm.routing.simple` | Project 3 | Count of requests routed to cheap model |
| `llm.routing.complex` | Project 3 | Count of requests routed to gpt-4o |
| `llm.tokens.prompt` | Project 1/3 | Prompt token usage |
| `llm.tokens.completion` | Project 1/3 | Completion token usage |

### Monitors

**URL:** `https://us5.datadoghq.com/monitors/manage`

Search for `RAG Bot` — the monitors are created when you run:
```bash
python project2_eval_framework/metrics/create_monitors.py
python project3_cost_latency_optimizer/dashboard/setup_slo_and_dashboard.py
```

### SLOs

**URL:** `https://us5.datadoghq.com/slo`

After running the setup script above, look for `99% of requests complete in < 2000ms`.

### Pre-built Dashboard

1. Go to `https://us5.datadoghq.com/dashboard/lists`
2. Click **New Dashboard** → **Import dashboard JSON**
3. Paste the contents of `project1_llm_trace_pipeline/datadog_config/dashboard.json`

### Local URLs (Project 4)

| URL | What it shows |
|-----|--------------|
| `http://localhost:8000/docs` | Swagger UI — try the endpoints interactively |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8080` | Main observability dashboard |

---

## How to Get Your Slack Webhook URL

Project 4 posts AI triage alerts to Slack. Follow these steps to get a webhook URL:

### Step 1 — Create a Slack App

1. Go to `https://api.slack.com/apps` or https://llm-sanika.slack.com/ssb/redirect?entry_point=workspace_signin
2. Click **Create New App**
3. Choose **From scratch**
4. Give it a name (e.g. `AI SRE Bot`) and select your workspace
5. Click **Create App**

### Step 2 — Enable Incoming Webhooks

1. In the left sidebar click **Incoming Webhooks**
2. Toggle **Activate Incoming Webhooks** to **On**
3. Scroll down and click **Add New Webhook to Workspace**
4. Choose the channel where you want alerts to appear (e.g. `#alerts` or `#general`)
5. Click **Allow**

### Step 3 — Copy the Webhook URL

After authorizing, Slack shows you a URL like:

```

```

Copy this URL and paste it into your `.env`:

```
SLACK_WEBHOOK_URL=
```

### Step 4 — Test it

After setting the URL, trigger a test from the triage bot:

```bash
curl -X POST http://localhost:8000/webhook/test
```

Check your Slack channel — you should see a formatted alert message within a few seconds.

> **Note:** The free Slack plan supports incoming webhooks. No paid plan needed.

---

| What | URL | Filter |
|------|-----|--------|
| LLM traces | `https://us5.datadoghq.com/llm/traces` | `ml_app:rag-support-bot` |
| Metrics | `https://us5.datadoghq.com/metric/explorer` | `llm.eval.*`, `llm.cost.*` |
| Monitors | `https://us5.datadoghq.com/monitors/manage` | search `RAG Bot` |
| SLOs | `https://us5.datadoghq.com/slo` | — |
| Local dashboard | `http://localhost:8080` | — |
| Local API docs | `http://localhost:8000/docs` | — |

---

## Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=project1_llm_trace_pipeline \
                  --cov=project2_eval_framework \
                  --cov=project3_cost_latency_optimizer \
                  --cov=project4_ai_sre_triage \
                  --cov-report=term-missing

# Run a single project's tests
pytest tests/test_project1_rag_pipeline.py -v
pytest tests/test_project2_eval_framework.py -v
pytest tests/test_project3_cost_router.py -v
pytest tests/test_project4_sre_triage.py -v

# Lint
pip install ruff
ruff check .
```

---

## Troubleshooting

### `failed to send, dropping traces to localhost:8126`
The APM tracer is trying to reach a local Datadog Agent. This is a harmless warning — LLM Observability uses `agentless_enabled=True` and sends data directly to Datadog's API, so traces will still appear in Datadog. You can ignore these log lines.

### `LLMObsAnnotateSpanError: Failed to parse input documents`
The `@embedding` span's `LLMObs.annotate()` requires `input_data` as a list of strings
or `[{"text": "..."}]` dicts — not `[{"content": "..."}]` (that's the LLM chat format).
This bug is already fixed in `project1_llm_trace_pipeline/app/rag_support_bot.py`.

### `Invalid webhook signature` on `/webhook/datadog`
The real webhook endpoint verifies an HMAC-SHA256 signature. Use `/webhook/test` for
local testing — it skips signature verification.

### Span API returns 0 results
Datadog's spans search index has a ~5-10 minute delay. Run Projects 1-3 first to
generate some traces, wait a few minutes, then trigger Project 4's triage.

### `No OPENAI_API_KEY found — running in dry-run mode`
The `.env` file is not being loaded. Use:
```python
from dotenv import load_dotenv; load_dotenv()
```
before running, or export the variable directly:
```bash
export OPENAI_API_KEY=sk-proj-...
```

### Slack returns `404 no_team`
The `SLACK_WEBHOOK_URL` in `.env` is still the placeholder value. Replace it with a real
incoming webhook URL from api.slack.com/apps.

### Unicode errors on Windows
Run Python with the `-X utf8` flag and set `PYTHONIOENCODING=utf-8`:
```bash
python -X utf8 your_script.py
```

---

## Project Structure

```
llm-observability-portfolio/
├── project1_llm_trace_pipeline/
│   ├── app/
│   │   └── rag_support_bot.py          # RAG pipeline with full span instrumentation
│   └── datadog_config/
│       ├── dashboard.json              # Importable Datadog dashboard
│       └── monitors.yaml               # Monitor definitions (error rate, latency, cost)
│
├── project2_eval_framework/
│   ├── evaluators/
│   │   └── eval_suite.py               # LLM-as-judge: faithfulness, relevancy, completeness
│   └── metrics/
│       └── create_monitors.py          # Creates quality monitors in Datadog via API
│
├── project3_cost_latency_optimizer/
│   ├── router/
│   │   └── model_router.py             # Complexity classifier + model router + cost tracker
│   └── dashboard/
│       └── setup_slo_and_dashboard.py  # Creates SLO + cost dashboard in Datadog
│
├── project4_ai_sre_triage/
│   └── webhook_handler/
│       └── server.py                   # FastAPI: webhook receiver + span fetcher + Slack poster
│
├── tests/
│   ├── conftest.py
│   ├── test_project1_rag_pipeline.py
│   ├── test_project2_eval_framework.py
│   ├── test_project3_cost_router.py
│   └── test_project4_sre_triage.py
│
├── demo.py                             # Full portfolio demo — runs all 4 projects
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── .env                                # Your credentials (never commit this)
```
