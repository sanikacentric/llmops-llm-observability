# LLM Evaluation Framework — Closing the Quality Loop with Datadog

## Overview

Most teams instrument LLM latency and token cost on day one. Quality degrades
quietly. This cookbook shows how to wire automated LLM evaluations into Datadog
so that a faithfulness regression triggers a page the same way a latency spike does.

**What you'll build:**

```
RAG Pipeline Output
        │
        ▼
┌───────────────────────────────────────────────┐
│  Eval Suite (runs after every pipeline call)  │
│                                               │
│  ┌──────────────┐  ┌───────────┐  ┌────────┐ │
│  │ Faithfulness │  │ Relevancy │  │ Compl. │ │
│  │  evaluator   │  │ evaluator │  │ eval   │ │
│  └──────┬───────┘  └─────┬─────┘  └───┬────┘ │
└─────────┼────────────────┼────────────┼───────┘
          │                │            │
          ▼                ▼            ▼
    DogStatsD gauges  LLMObs.submit_evaluation()  eval_log.jsonl
          │
          ▼
  Datadog Monitor → PagerDuty / Slack
```

## Why LLM-as-judge?

Using a small, fast LLM (gpt-4o-mini) to evaluate a larger LLM's output is
cost-effective and scales to production traffic. The key insight: evaluation
prompts are simpler than generation prompts, so a cheaper model scores reliably.

Cost estimate: evaluating 10,000 RAG responses/day with gpt-4o-mini ≈ $0.50/day.

## The three evaluators

### Faithfulness (most critical for RAG)
Detects hallucination — answers that make claims not supported by retrieved docs.
Score range: 0.0 (contradicts context) → 1.0 (fully grounded).
Alert threshold: < 0.7 triggers PagerDuty.

### Relevancy
Detects off-topic answers — the model answered a different question than asked.
Often caused by query rewriting failures or retrieval returning wrong docs.
Score range: 0.0 (unrelated) → 1.0 (directly addresses question).

### Completeness
Detects answer truncation — relevant context was retrieved but not used.
Softer threshold (0.7) since some truncation is intentional for conciseness.

## Connecting evals to traces

Every eval score is linked to the originating RAG trace via `trace_id` and
`span_id`. In Datadog LLM Observability Explorer, you can:
- Filter traces by `eval_metric:faithfulness label:fail`
- Click through to see the exact prompt, completion, and retrieved context
- Understand *why* the model hallucinated, not just *that* it did

This linkage is what transforms evaluation from a batch job into a live
observability signal.

## Metric naming convention

```
llm.eval.<metric_name>.score   # gauge, 0.0–1.0, for trend dashboards
llm.eval.<metric_name>.pass    # count, 0 or 1, for SLO math
llm.eval.overall.pass          # count, 0 or 1, aggregate quality signal
```

All metrics tagged with: `ml_app`, `env`, `version`, `eval_metric`.

## Setup

```bash
pip install openai ddtrace datadog

export DD_API_KEY="..."
export DD_APP_KEY="..."
export DD_SITE="datadoghq.com"
export OPENAI_API_KEY="..."

# Run the evaluator
python evaluators/eval_suite.py

# Create Datadog monitors
python metrics/create_monitors.py
```

## Running evals in production

**Option A: Synchronous (simple)** — run evals inline after each RAG call.
Adds ~500ms latency. Good for low-volume or latency-insensitive workloads.

**Option B: Async (recommended for production)** — publish RAG outputs to
a queue (SQS, Kafka, or Celery). Eval workers consume the queue separately.
No latency impact on the user path.

```python
# Option B sketch
from celery import Celery
app = Celery("evals", broker=os.environ["REDIS_URL"])

@app.task
def async_eval(question, answer, context, trace_id, span_id):
    run_eval_suite(question, answer, context, trace_id=trace_id, span_id=span_id)

# After RAG pipeline:
async_eval.delay(question, response.answer, context, trace_id, span_id)
```

**Option C: Sampled (cost optimization)** — eval a percentage of requests.

```python
import random
if random.random() < 0.1:   # eval 10% of traffic
    run_eval_suite(...)
```

## Interview talking points

- "Most customers treat evals as a CI/CD gate — they run them before deploy
  and never again. The gap is that model behavior drifts in production due to
  real user queries. Wiring evals into Datadog metrics closes that gap."

- "The LLM-as-judge pattern needs calibration. I'd work with a customer to
  build a golden dataset of 100 labeled Q&A pairs, run the evaluators against
  it, and tune thresholds to match human judgment. Without calibration, the
  monitors are noise."

- "Linking eval scores to trace IDs is the key UX insight. If faithfulness
  drops at 2am, the on-call engineer can click into the failing traces and see
  the exact context that caused the hallucination — without needing to
  reproduce it."

## Files

| File | Purpose |
|------|---------|
| `evaluators/eval_suite.py` | Three evaluators + metric emission + JSONL logging |
| `metrics/create_monitors.py` | Creates Datadog monitors via API |
