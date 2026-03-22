# Project 1: LLM Trace Pipeline — Datadog LLM Observability

## What this is

A production-instrumented RAG (Retrieval-Augmented Generation) support bot.
Every stage of the pipeline emits structured Datadog spans so you can see
exactly where latency, cost, and errors originate.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  @workflow  rag_support_pipeline                    │
│                                                     │
│  ┌──────────────────┐    ┌───────────────────────┐  │
│  │ @llm             │    │ @embedding            │  │
│  │ query_rewriter   │───▶│ embed_text            │  │
│  │ (gpt-4o-mini)    │    │ (text-embedding-3-sm) │  │
│  └──────────────────┘    └──────────┬────────────┘  │
│                                     │               │
│                          ┌──────────▼────────────┐  │
│                          │ @task                 │  │
│                          │ vector_retrieval      │  │
│                          │ (cosine similarity)   │  │
│                          └──────────┬────────────┘  │
│                                     │               │
│                          ┌──────────▼────────────┐  │
│                          │ @llm                  │  │
│                          │ answer_generator      │  │
│                          │ (gpt-4o)              │  │
│                          └──────────┬────────────┘  │
└─────────────────────────────────────┼───────────────┘
                                      │
                                 BotResponse
```

**What Datadog captures automatically:**
- Full trace tree (workflow → task → llm spans)
- Prompt and completion text for every LLM call
- Token counts (prompt, completion, total) per span
- Latency at each step
- Model name and provider
- Custom metadata (doc count, retrieval scores, template version)

## Key design decisions

### Two-model strategy
- `gpt-4o-mini` for query rewriting (cheap, fast, simple task)
- `gpt-4o` for answer generation (premium, handles complex reasoning)

This is intentional: Project 3 builds a full cost router on top of this pattern.

### Prompt template versioning
Every LLM span includes `template_version` in metadata. When you update a
prompt, bump the version string. In Datadog you can then filter traces by
template version and compare quality/cost before rolling out.

### @workflow decorator
The `@workflow` span wraps the entire pipeline. This gives you a single
root span in Datadog — you can filter the LLM Observability Explorer by
`span.name:rag_support_pipeline` to see all end-to-end traces.

## Setup

```bash
pip install openai ddtrace numpy

export DD_API_KEY="your-datadog-api-key"
export DD_SITE="datadoghq.com"          # or datadoghq.eu
export OPENAI_API_KEY="your-openai-key"
export DD_ENV="development"

python app/rag_support_bot.py
```

## What to look at in Datadog

1. **LLM Observability Explorer** → filter by `ml_app:rag-support-bot`
2. Click any trace → expand the span tree → see the full workflow
3. Click `answer_generator` span → view prompt, completion, and token count
4. **Dashboard** → import `datadog_config/dashboard.json`
5. **Monitors** → apply `datadog_config/monitors.yaml` via Terraform

## Interview talking points

- "I hit the reality that customers often onboard with no span naming strategy —
  their traces show up as a flat list. The `@workflow` decorator solves this by
  giving the pipeline a root span, making the trace tree readable."

- "The two-model pattern isn't just about cost — it's observable. In Datadog you
  can immediately see that query_rewriter (gpt-4o-mini) costs 10x less than
  answer_generator, which is the data you need to justify the routing strategy."

- "Template versioning in metadata is something I'd push every enterprise customer
  to implement from day one. Prompt changes are deployments — you need the same
  observability you'd give a code deploy."

## Files

| File | Purpose |
|------|---------|
| `app/rag_support_bot.py` | Full RAG pipeline with DD instrumentation |
| `datadog_config/dashboard.json` | Importable Datadog dashboard |
| `datadog_config/monitors.yaml` | SLO + cost + error rate monitors |
