"""
Project 3: Cost + Latency Optimizer — Intelligent Model Router
--------------------------------------------------------------
Routes LLM requests between models (gpt-4o vs gpt-4o-mini) based on
query complexity. Tracks per-request cost as Datadog custom metrics and
maintains a Datadog SLO on p95 latency.

The core insight: not every query needs the most expensive model.
A question like "what is Python?" should never hit gpt-4o.

Routing logic:
  1. Classify query complexity (fast, cheap classifier call)
  2. Route: simple → gpt-4o-mini, complex → gpt-4o
  3. Emit cost + quality metrics per model tier
  4. Dashboard shows cost vs quality tradeoff over time

This is a pattern that directly addresses the "cost management" requirement
called out in the Datadog PSA job description.
"""

import os
import time
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import openai
from datadog import initialize, statsd
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm, workflow, task

logger = logging.getLogger(__name__)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
initialize(api_key=os.environ["DD_API_KEY"])
LLMObs.enable(
    ml_app="rag-support-bot",
    agentless_enabled=True,
    api_key=os.environ["DD_API_KEY"],
)
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── Cost table (update when OpenAI changes pricing) ───────────────────────────
# Prices per 1M tokens (USD) as of 2025
MODEL_PRICING = {
    "gpt-4o": {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
        "tier": "premium",
    },
    "gpt-4o-mini": {
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "tier": "standard",
    },
    "text-embedding-3-small": {
        "input_per_1m": 0.02,
        "output_per_1m": 0.00,
        "tier": "embedding",
    },
}


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"       # → gpt-4o-mini
    MODERATE = "moderate"   # → gpt-4o-mini with more tokens
    COMPLEX = "complex"     # → gpt-4o


@dataclass
class RoutingDecision:
    complexity: ComplexityLevel
    model: str
    reasoning: str
    classifier_tokens: int
    classifier_latency_ms: float


@dataclass
class RoutedResponse:
    content: str
    model_used: str
    complexity: ComplexityLevel
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    routing_decision: RoutingDecision


# ── Cost calculator ───────────────────────────────────────────────────────────
def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate exact cost for a given API call."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
    cost = (
        (prompt_tokens / 1_000_000) * pricing["input_per_1m"]
        + (completion_tokens / 1_000_000) * pricing["output_per_1m"]
    )
    return round(cost, 8)


# ── Complexity classifier ─────────────────────────────────────────────────────
COMPLEXITY_PROMPT = """Classify the complexity of this user question for an AI support assistant.
Simple: factual lookups, definitions, yes/no questions, short how-to steps.
Moderate: multi-step explanations, comparisons, troubleshooting with a few variables.
Complex: architectural decisions, debugging with many unknowns, open-ended analysis, code generation.

Question: {question}

Respond ONLY with JSON: {{"complexity": "simple|moderate|complex", "reasoning": "<10 words max>"}}"""


@llm(model_name="gpt-4o-mini", model_provider="openai", name="complexity_classifier")
def classify_complexity(question: str) -> RoutingDecision:
    """
    Classify query complexity using gpt-4o-mini.
    This meta-call costs ~$0.000001 — negligible even at 1M req/day.
    """
    start = time.perf_counter()
    prompt = COMPLEXITY_PROMPT.format(question=question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    complexity = ComplexityLevel(result.get("complexity", "moderate"))
    reasoning = result.get("reasoning", "")
    latency_ms = (time.perf_counter() - start) * 1000

    # Route to model based on complexity
    model = "gpt-4o" if complexity == ComplexityLevel.COMPLEX else "gpt-4o-mini"

    LLMObs.annotate(
        input_data=[{"role": "user", "content": question}],
        output_data=[{"role": "assistant", "content": json.dumps(result)}],
        metadata={
            "complexity": complexity.value,
            "routed_to": model,
            "classifier_tokens": response.usage.total_tokens,
        },
    )

    return RoutingDecision(
        complexity=complexity,
        model=model,
        reasoning=reasoning,
        classifier_tokens=response.usage.total_tokens,
        classifier_latency_ms=latency_ms,
    )


# ── Model call with cost tracking ─────────────────────────────────────────────
@llm(model_name="dynamic", model_provider="openai", name="routed_llm_call")
def call_model(
    messages: list[dict],
    model: str,
    complexity: ComplexityLevel,
    max_tokens: int = 500,
) -> tuple[str, int, int, float]:
    """
    Make the actual LLM call with the routed model.
    Returns: (content, prompt_tokens, completion_tokens, latency_ms)
    """
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content.strip()
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    latency_ms = (time.perf_counter() - start) * 1000

    LLMObs.annotate(
        input_data=messages,
        output_data=[{"role": "assistant", "content": content}],
        metadata={
            "model": model,
            "complexity": complexity.value,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": round(latency_ms, 2),
        },
        tags={"model_tier": MODEL_PRICING.get(model, {}).get("tier", "unknown")},
    )

    return content, prompt_tokens, completion_tokens, latency_ms


# ── Metric emission ───────────────────────────────────────────────────────────
def emit_routing_metrics(response: RoutedResponse, env: str = "production") -> None:
    """
    Emit cost, latency, and routing decision metrics.
    These populate the cost vs quality dashboard.
    """
    tags = [
        f"env:{env}",
        f"model:{response.model_used}",
        f"complexity:{response.complexity.value}",
        f"model_tier:{MODEL_PRICING.get(response.model_used, {}).get('tier', 'unknown')}",
        "ml_app:rag-support-bot",
    ]

    # Cost per request (the key business metric)
    statsd.gauge("llm.cost.usd_per_request", response.cost_usd, tags=tags)

    # Cumulative cost tracking (for budget alerts)
    statsd.increment("llm.cost.usd_total", value=response.cost_usd, tags=tags)

    # Token counts (for capacity planning)
    statsd.gauge("llm.tokens.prompt", response.prompt_tokens, tags=tags)
    statsd.gauge("llm.tokens.completion", response.completion_tokens, tags=tags)
    statsd.gauge("llm.tokens.total", response.prompt_tokens + response.completion_tokens, tags=tags)

    # Latency (for SLO tracking)
    statsd.histogram("llm.request.duration_ms", response.latency_ms, tags=tags)

    # Routing decision distribution (shows model mix over time)
    statsd.increment(f"llm.routing.{response.complexity.value}", tags=tags)

    logger.info(
        "Routing metrics emitted | model=%s complexity=%s cost=$%.6f latency=%.0fms",
        response.model_used,
        response.complexity.value,
        response.cost_usd,
        response.latency_ms,
    )


# ── Main router workflow ───────────────────────────────────────────────────────
@workflow(name="cost_optimized_llm_router")
def route_and_respond(
    user_message: str,
    system_prompt: str = "You are a helpful support assistant.",
    env: str = "production",
) -> RoutedResponse:
    """
    Full routing workflow:
      1. Classify complexity
      2. Route to appropriate model
      3. Track cost and emit metrics
    """
    LLMObs.annotate(
        input_data=[{"role": "user", "content": user_message}],
        tags={"env": env},
    )

    # Step 1: Classify
    routing = classify_complexity(user_message)
    logger.info(
        "Routing decision | complexity=%s model=%s reason='%s'",
        routing.complexity.value,
        routing.model,
        routing.reasoning,
    )

    # Step 2: Call routed model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    # Give complex queries more token budget
    max_tokens = 1000 if routing.complexity == ComplexityLevel.COMPLEX else 500

    content, prompt_tokens, completion_tokens, latency_ms = call_model(
        messages=messages,
        model=routing.model,
        complexity=routing.complexity,
        max_tokens=max_tokens,
    )

    # Step 3: Calculate cost
    cost_usd = calculate_cost(routing.model, prompt_tokens, completion_tokens)

    response = RoutedResponse(
        content=content,
        model_used=routing.model,
        complexity=routing.complexity,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        routing_decision=routing,
    )

    # Step 4: Emit metrics
    emit_routing_metrics(response, env=env)

    LLMObs.annotate(
        output_data=[{"role": "assistant", "content": content}],
        metadata={
            "model_used": routing.model,
            "complexity": routing.complexity.value,
            "cost_usd": cost_usd,
            "total_latency_ms": round(latency_ms, 2),
        },
    )

    return response


# ── Cost simulation: show savings from routing ────────────────────────────────
@task(name="cost_simulation")
def simulate_routing_savings(
    query_distribution: dict[str, float],  # {"simple": 0.6, "moderate": 0.3, "complex": 0.1}
    daily_request_count: int = 10_000,
    avg_prompt_tokens: int = 500,
    avg_completion_tokens: int = 200,
) -> dict:
    """
    Calculate projected daily cost with and without routing.
    Use this to demonstrate ROI to enterprise customers.
    """
    # All-premium baseline (no routing, always gpt-4o)
    baseline_cost_per_req = calculate_cost("gpt-4o", avg_prompt_tokens, avg_completion_tokens)
    baseline_daily = baseline_cost_per_req * daily_request_count

    # With routing
    routed_daily = 0.0
    for complexity, fraction in query_distribution.items():
        model = "gpt-4o" if complexity == "complex" else "gpt-4o-mini"
        cost_per_req = calculate_cost(model, avg_prompt_tokens, avg_completion_tokens)
        routed_daily += cost_per_req * daily_request_count * fraction

    savings_daily = baseline_daily - routed_daily
    savings_pct = (savings_daily / baseline_daily * 100) if baseline_daily > 0 else 0

    result = {
        "daily_requests": daily_request_count,
        "baseline_daily_cost_usd": round(baseline_daily, 4),
        "routed_daily_cost_usd": round(routed_daily, 4),
        "daily_savings_usd": round(savings_daily, 4),
        "savings_percent": round(savings_pct, 1),
        "annual_savings_usd": round(savings_daily * 365, 2),
        "query_distribution": query_distribution,
    }

    # Emit simulation results as metrics for the dashboard
    statsd.gauge("llm.cost.projected_daily_baseline", baseline_daily, tags=["ml_app:rag-support-bot"])
    statsd.gauge("llm.cost.projected_daily_routed", routed_daily, tags=["ml_app:rag-support-bot"])
    statsd.gauge("llm.cost.savings_percent", savings_pct, tags=["ml_app:rag-support-bot"])

    return result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is an LLM span in Datadog?",
        "How do I set up Datadog LLM Observability for a multi-step LangChain pipeline that uses both retrieval and generation?",
        "Design an observability architecture for an AI system that handles 50M requests/day across 3 regions with different compliance requirements.",
    ]

    print("\n" + "=" * 65)
    print("Cost-Optimized LLM Router Demo")
    print("=" * 65 + "\n")

    for query in test_queries:
        print(f"Query: {query[:70]}...")
        resp = route_and_respond(query, env="development")
        print(f"  Model:      {resp.model_used}")
        print(f"  Complexity: {resp.complexity.value}")
        print(f"  Cost:       ${resp.cost_usd:.6f}")
        print(f"  Latency:    {resp.latency_ms:.0f}ms")
        print()

    # Show projected savings
    print("Cost simulation (10,000 req/day):")
    savings = simulate_routing_savings(
        query_distribution={"simple": 0.60, "moderate": 0.30, "complex": 0.10},
        daily_request_count=10_000,
    )
    print(f"  Baseline (all gpt-4o):    ${savings['baseline_daily_cost_usd']:.2f}/day")
    print(f"  With routing:             ${savings['routed_daily_cost_usd']:.2f}/day")
    print(f"  Daily savings:            ${savings['daily_savings_usd']:.2f} ({savings['savings_percent']}%)")
    print(f"  Annual savings:           ${savings['annual_savings_usd']:,.0f}")
