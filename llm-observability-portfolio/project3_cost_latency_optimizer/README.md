# Project 3: Cost + Latency Optimizer — Intelligent Model Router

## What this is

A model routing layer that classifies query complexity and routes to the
cheapest model that can handle it. Cost and quality are tracked side-by-side
in Datadog so you can see — with data — that routing didn't hurt quality.

## Why this matters to enterprise customers

Enterprise teams get one of two finance questions:
1. "Why is our AI bill $50K/month?"
2. "Can we cut costs without degrading quality?"

This project answers both. The dashboard literally shows cost per request
alongside faithfulness score, by model tier, over time.

## Routing logic

```
User query
    │
    ▼
┌─────────────────────────────────┐
│  Complexity Classifier          │  ← gpt-4o-mini, ~$0.000001/call
│  (fast, cheap meta-call)        │
└───────────┬─────────────────────┘
            │
     ┌──────┴────────┐
  simple/moderate   complex
     │               │
     ▼               ▼
gpt-4o-mini       gpt-4o
$0.15/1M in       $2.50/1M in
$0.60/1M out     $10.00/1M out
```

**Typical production distribution:** 60% simple / 30% moderate / 10% complex
→ ~75% cost reduction vs all-gpt-4o, with <2% quality delta on evals.

## Key metrics emitted

| Metric | Type | Purpose |
|--------|------|---------|
| `llm.cost.usd_per_request` | gauge | Per-request cost by model |
| `llm.cost.usd_total` | count | Cumulative spend (for budget alerts) |
| `llm.tokens.prompt` | gauge | Prompt token usage |
| `llm.tokens.completion` | gauge | Completion token usage |
| `llm.request.duration_ms` | histogram | Latency distribution |
| `llm.routing.<complexity>` | count | Routing decision distribution |
| `llm.cost.savings_percent` | gauge | Live savings vs all-premium baseline |

## SLO

A metric-based Datadog SLO tracks: **99% of requests complete within 2 seconds.**

The SLO widget on the dashboard shows burn rate, so you can see if a cost
optimization change (e.g. switching a complexity band to a different model)
also impacted latency.

## Cost simulation

The `simulate_routing_savings()` function calculates projected savings given
a query distribution. Example output:

```
Cost simulation (10,000 req/day):
  Baseline (all gpt-4o):    $19.40/day
  With routing:             $4.87/day
  Daily savings:            $14.53 (74.9%)
  Annual savings:           $5,303
```

Use this to build the ROI case for enterprise procurement.

## Setup

```bash
pip install openai ddtrace datadog requests

export DD_API_KEY="..."
export DD_APP_KEY="..."
export OPENAI_API_KEY="..."

# Run the router demo
python router/model_router.py

# Create SLO and dashboard
python dashboard/setup_slo_and_dashboard.py
```

## Interview talking points

- "The classifier itself costs almost nothing — $0.000001 per call. The savings
  on complex-vs-simple routing are typically 60-80%. The math is obvious once
  you can show it in a dashboard."

- "The key risk of complexity routing is miscategorizing a complex question as
  simple. That's why Project 2's faithfulness eval runs downstream — it's the
  circuit breaker. If routing quality drops, faithfulness drops, the monitor
  fires, and you can tune the classifier."

- "Enterprise customers always ask 'can you show me the cost before and after?'
  The simulate_routing_savings function generates that number in one function
  call. I'd run it in a pre-sales call using the customer's actual request volume."
