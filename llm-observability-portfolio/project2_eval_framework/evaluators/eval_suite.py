"""
Project 2: LLM Evaluation Framework with Datadog Custom Metrics
---------------------------------------------------------------
Automated quality scoring for RAG pipeline outputs.

Evaluators:
  - Faithfulness:  Does the answer stick to the retrieved context?
  - Relevancy:     Does the answer address the user's question?
  - Completeness:  Does the answer cover the key points from context?

Scores are:
  1. Submitted to Datadog via LLMObs.submit_evaluation()
  2. Emitted as custom DogStatsD metrics for alerting and dashboards
  3. Written to a local JSONL log for offline analysis

This mirrors how a PSA team would productionize eval — not just running
evals in a notebook, but wiring them into the observability loop.
"""

import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum

import openai
from datadog import initialize, statsd
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm

logger = logging.getLogger(__name__)

# ── Initialize Datadog ────────────────────────────────────────────────────────
initialize(
    api_key=os.environ["DD_API_KEY"],
    app_key=os.environ.get("DD_APP_KEY", ""),
)

LLMObs.enable(
    ml_app="rag-support-bot",
    agentless_enabled=True,
    api_key=os.environ["DD_API_KEY"],
    site=os.environ.get("DD_SITE", "datadoghq.com"),
)

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── Data models ───────────────────────────────────────────────────────────────
class EvalLabel(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


@dataclass
class EvalScore:
    metric_name: str        # "faithfulness", "relevancy", "completeness"
    score: float            # 0.0 – 1.0
    label: EvalLabel        # pass / fail / partial
    reasoning: str          # LLM's explanation (for debugging)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class EvalBundle:
    """All evaluation scores for a single RAG pipeline run."""
    question: str
    answer: str
    context: str
    rewritten_query: str
    scores: list[EvalScore]
    overall_pass: bool
    timestamp: float


# ── Prompt templates ──────────────────────────────────────────────────────────
FAITHFULNESS_PROMPT = """You are an evaluation judge. Your task is to assess whether 
an answer is faithful to the provided context — meaning every claim in the answer 
can be directly supported by the context.

Context:
{context}

Answer:
{answer}

Score the faithfulness from 0.0 to 1.0:
- 1.0 = Every claim is supported by the context
- 0.5 = Some claims are supported, some are hallucinated
- 0.0 = Answer contradicts or ignores the context entirely

Respond ONLY with a JSON object:
{{"score": <float 0-1>, "reasoning": "<one sentence explanation>"}}"""

RELEVANCY_PROMPT = """You are an evaluation judge. Assess whether the answer 
actually addresses the user's question.

Question: {question}
Answer: {answer}

Score relevancy from 0.0 to 1.0:
- 1.0 = Answer directly and completely addresses the question
- 0.5 = Answer is partially relevant or goes off-topic
- 0.0 = Answer does not address the question at all

Respond ONLY with a JSON object:
{{"score": <float 0-1>, "reasoning": "<one sentence explanation>"}}"""

COMPLETENESS_PROMPT = """You are an evaluation judge. Assess whether the answer 
covers all the key information available in the context that is relevant to the question.

Question: {question}
Context: {context}
Answer: {answer}

Score completeness from 0.0 to 1.0:
- 1.0 = Answer uses all relevant information from context
- 0.5 = Answer misses some important points from context
- 0.0 = Answer ignores most relevant context

Respond ONLY with a JSON object:
{{"score": <float 0-1>, "reasoning": "<one sentence explanation>"}}"""


# ── Individual evaluators ─────────────────────────────────────────────────────
@llm(model_name="gpt-4o-mini", model_provider="openai", name="eval_faithfulness")
def evaluate_faithfulness(
    answer: str,
    context: str,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> EvalScore:
    """
    Check if the answer hallucinates beyond the retrieved context.
    This is the most critical eval for RAG — hallucination detection.
    """
    start = time.perf_counter()
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    score_val = float(result["score"])
    reasoning = result.get("reasoning", "")
    latency_ms = (time.perf_counter() - start) * 1000

    # Determine label
    label = EvalLabel.PASS if score_val >= 0.8 else (
        EvalLabel.PARTIAL if score_val >= 0.5 else EvalLabel.FAIL
    )

    eval_score = EvalScore(
        metric_name="faithfulness",
        score=score_val,
        label=label,
        reasoning=reasoning,
        trace_id=trace_id,
        span_id=span_id,
        latency_ms=latency_ms,
    )

    LLMObs.annotate(
        input_data=[{"role": "user", "content": prompt}],
        output_data=[{"role": "assistant", "content": json.dumps(result)}],
        metadata={"eval_metric": "faithfulness", "score": score_val, "label": label},
    )

    return eval_score


@llm(model_name="gpt-4o-mini", model_provider="openai", name="eval_relevancy")
def evaluate_relevancy(
    question: str,
    answer: str,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> EvalScore:
    """
    Check if the answer actually addresses what the user asked.
    """
    start = time.perf_counter()
    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    score_val = float(result["score"])
    reasoning = result.get("reasoning", "")
    latency_ms = (time.perf_counter() - start) * 1000

    label = EvalLabel.PASS if score_val >= 0.8 else (
        EvalLabel.PARTIAL if score_val >= 0.5 else EvalLabel.FAIL
    )

    eval_score = EvalScore(
        metric_name="relevancy",
        score=score_val,
        label=label,
        reasoning=reasoning,
        trace_id=trace_id,
        span_id=span_id,
        latency_ms=latency_ms,
    )

    LLMObs.annotate(
        input_data=[{"role": "user", "content": prompt}],
        output_data=[{"role": "assistant", "content": json.dumps(result)}],
        metadata={"eval_metric": "relevancy", "score": score_val, "label": label},
    )

    return eval_score


@llm(model_name="gpt-4o-mini", model_provider="openai", name="eval_completeness")
def evaluate_completeness(
    question: str,
    answer: str,
    context: str,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> EvalScore:
    """
    Check if the answer surfaces all relevant information from context.
    """
    start = time.perf_counter()
    prompt = COMPLETENESS_PROMPT.format(
        question=question, context=context, answer=answer
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    score_val = float(result["score"])
    reasoning = result.get("reasoning", "")
    latency_ms = (time.perf_counter() - start) * 1000

    label = EvalLabel.PASS if score_val >= 0.7 else (
        EvalLabel.PARTIAL if score_val >= 0.4 else EvalLabel.FAIL
    )

    eval_score = EvalScore(
        metric_name="completeness",
        score=score_val,
        label=label,
        reasoning=reasoning,
        trace_id=trace_id,
        span_id=span_id,
        latency_ms=latency_ms,
    )

    LLMObs.annotate(
        input_data=[{"role": "user", "content": prompt}],
        output_data=[{"role": "assistant", "content": json.dumps(result)}],
        metadata={"eval_metric": "completeness", "score": score_val, "label": label},
    )

    return eval_score


# ── Metric emission ───────────────────────────────────────────────────────────
def emit_eval_metrics(bundle: EvalBundle, env: str = "production") -> None:
    """
    Emit evaluation scores as Datadog custom metrics via DogStatsD.
    These feed the quality dashboard and threshold monitors.

    Metric naming convention:
      llm.eval.<metric_name>.score   — gauge (0.0-1.0)
      llm.eval.<metric_name>.pass    — count (1 if pass, 0 otherwise)
      llm.eval.overall.pass          — count (1 if all metrics pass)
    """
    base_tags = [
        f"env:{env}",
        "ml_app:rag-support-bot",
        "version:1.0.0",
    ]

    for score in bundle.scores:
        metric_tags = base_tags + [f"eval_metric:{score.metric_name}"]

        # Gauge: the raw score (useful for trends and regression detection)
        statsd.gauge(
            f"llm.eval.{score.metric_name}.score",
            score.score,
            tags=metric_tags,
        )

        # Count: pass/fail for SLO-style tracking
        statsd.increment(
            f"llm.eval.{score.metric_name}.pass",
            value=1 if score.label == EvalLabel.PASS else 0,
            tags=metric_tags,
        )

        # Submit to LLM Observability eval panel (if trace_id is available)
        if score.trace_id and score.span_id:
            LLMObs.submit_evaluation(
                span_context={"trace_id": score.trace_id, "span_id": score.span_id},
                label=score.label.value,
                metric_type="score",
                value=score.score,
                ml_app="rag-support-bot",
                tags={"eval_metric": score.metric_name},
            )

        logger.info(
            "Eval metric emitted | metric=%s score=%.2f label=%s",
            score.metric_name,
            score.score,
            score.label,
        )

    # Overall pipeline quality signal
    statsd.increment(
        "llm.eval.overall.pass",
        value=1 if bundle.overall_pass else 0,
        tags=base_tags,
    )


def log_eval_bundle(bundle: EvalBundle, log_file: str = "eval_log.jsonl") -> None:
    """
    Append eval results to a JSONL file for offline analysis, regression tracking,
    and fine-tuning data collection.
    """
    record = {
        "timestamp": bundle.timestamp,
        "question": bundle.question,
        "answer": bundle.answer[:200] + "..." if len(bundle.answer) > 200 else bundle.answer,
        "overall_pass": bundle.overall_pass,
        "scores": {
            s.metric_name: {
                "score": round(s.score, 3),
                "label": s.label.value,
                "reasoning": s.reasoning,
            }
            for s in bundle.scores
        },
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Main orchestrator ─────────────────────────────────────────────────────────
def run_eval_suite(
    question: str,
    answer: str,
    context: str,
    rewritten_query: str = "",
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    env: str = "production",
) -> EvalBundle:
    """
    Run all three evaluators and emit metrics to Datadog.
    
    This is designed to be called immediately after run_rag_pipeline():
    
        response = run_rag_pipeline(question, store)
        eval_bundle = run_eval_suite(
            question=question,
            answer=response.answer,
            context=response.sources_text,
            trace_id=response.trace_id,
        )
    """
    logger.info("Running eval suite | question='%s'", question[:80])

    faithfulness = evaluate_faithfulness(
        answer=answer, context=context, trace_id=trace_id, span_id=span_id
    )
    relevancy = evaluate_relevancy(
        question=question, answer=answer, trace_id=trace_id, span_id=span_id
    )
    completeness = evaluate_completeness(
        question=question, answer=answer, context=context,
        trace_id=trace_id, span_id=span_id,
    )

    scores = [faithfulness, relevancy, completeness]

    # Overall pass: faithfulness and relevancy must both pass (completeness is softer)
    overall_pass = (
        faithfulness.label == EvalLabel.PASS
        and relevancy.label != EvalLabel.FAIL
    )

    bundle = EvalBundle(
        question=question,
        answer=answer,
        context=context,
        rewritten_query=rewritten_query,
        scores=scores,
        overall_pass=overall_pass,
        timestamp=time.time(),
    )

    emit_eval_metrics(bundle, env=env)
    log_eval_bundle(bundle)

    logger.info(
        "Eval suite complete | overall_pass=%s faithfulness=%.2f relevancy=%.2f completeness=%.2f",
        overall_pass,
        faithfulness.score,
        relevancy.score,
        completeness.score,
    )

    return bundle


# ── Test harness ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a RAG pipeline output and evaluate it
    test_cases = [
        {
            "question": "How do I instrument a LangChain app with Datadog?",
            "context": (
                "To instrument a LangChain application, install ddtrace and call "
                "LLMObs.enable() before initializing your chain. The dd-trace library "
                "automatically patches LangChain, LlamaIndex, OpenAI, and Anthropic clients."
            ),
            "answer": (
                "Install ddtrace with pip, then call LLMObs.enable() at the start of "
                "your application before creating any LangChain chains. Datadog will "
                "automatically instrument LangChain, OpenAI, and Anthropic clients."
            ),
        },
        {
            # Test hallucination detection
            "question": "What database does Datadog use internally?",
            "context": (
                "Datadog LLM Observability captures spans for every LLM call including "
                "model name, prompt, completion, and token usage."
            ),
            "answer": (
                "Datadog uses PostgreSQL internally for storing span data, and Redis "
                "for caching. They also use Cassandra for time-series metrics."
            ),
        },
    ]

    for case in test_cases:
        print(f"\nEvaluating: {case['question'][:60]}...")
        bundle = run_eval_suite(
            question=case["question"],
            answer=case["answer"],
            context=case["context"],
            env="development",
        )
        print(f"  Overall: {'PASS' if bundle.overall_pass else 'FAIL'}")
        for score in bundle.scores:
            print(f"  {score.metric_name:15} {score.score:.2f} ({score.label.value})")
            print(f"    → {score.reasoning}")
