#!/usr/bin/env python3
"""
demo.py — Full Portfolio Demo
==============================
Runs all four projects in sequence with rich terminal output.
Use this in the interview screen share.

Usage:
  python demo.py              # full demo, all 4 projects
  python demo.py --project 1  # run only project 1
  python demo.py --project 3  # run only project 3 (cost simulation, no API needed)
  python demo.py --dry-run    # show what would run, skip all API calls

What it shows:
  Project 1 → RAG pipeline trace (spans emitted to Datadog)
  Project 2 → Eval suite on P1 output (metrics emitted)
  Project 3 → Cost routing comparison + savings simulation
  Project 4 → Triage bot triggered on a simulated alert

The --dry-run flag is useful for demos without live credentials.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# ── Terminal formatting ───────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"
BG_DARK = "\033[48;5;236m"


def header(text: str) -> None:
    width = 65
    print()
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def step(label: str, detail: str = "") -> None:
    print(f"  {GREEN}▶{RESET} {BOLD}{label}{RESET}", end="")
    if detail:
        print(f"  {DIM}{detail}{RESET}", end="")
    print()


def result(key: str, value: str, color: str = WHITE) -> None:
    print(f"    {DIM}{key:<22}{RESET}{color}{value}{RESET}")


def metric(name: str, value: str, tag: str = "") -> None:
    tag_str = f"  {DIM}[{tag}]{RESET}" if tag else ""
    print(f"    {MAGENTA}◆{RESET} {name:<35} {YELLOW}{value}{RESET}{tag_str}")


def divider() -> None:
    print(f"  {DIM}{'·' * 60}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def thinking(msg: str) -> None:
    print(f"  {BLUE}⟳{RESET}  {DIM}{msg}...{RESET}")


def emit(metric_name: str, value: str) -> None:
    print(f"  {MAGENTA}→ DD{RESET} {DIM}statsd.gauge({metric_name!r}, {value}){RESET}")


# ── Dry-run simulation helpers ────────────────────────────────────────────────

SIMULATED_PIPELINE_OUTPUT = {
    "question": "How do I set up Datadog LLM Observability for my OpenAI app?",
    "rewritten_query": "Datadog LLM Observability OpenAI setup instrumentation",
    "answer": (
        "Install ddtrace with pip, then call LLMObs.enable() at the start of "
        "your application before any OpenAI calls. The dd-trace library will "
        "automatically instrument your OpenAI client, capturing spans for "
        "every completion including prompt, response, and token counts."
    ),
    "sources": ["kb-001", "kb-002"],
    "tokens_used": 287,
    "total_latency_ms": 1243.0,
}

SIMULATED_EVAL_OUTPUT = {
    "faithfulness": {"score": 0.93, "label": "pass", "reasoning": "All claims grounded in retrieved docs"},
    "relevancy":    {"score": 0.91, "label": "pass", "reasoning": "Directly addresses the setup question"},
    "completeness": {"score": 0.78, "label": "pass", "reasoning": "Covers main steps; could mention agentless mode"},
    "overall_pass": True,
}

SIMULATED_ROUTING_RESULTS = [
    {
        "query":      "What is an LLM span?",
        "complexity": "simple",
        "model":      "gpt-4o-mini",
        "cost_usd":   0.0000043,
        "latency_ms": 312,
    },
    {
        "query":      "How do I set up LLM Observability for a multi-step LangChain pipeline?",
        "complexity": "moderate",
        "model":      "gpt-4o-mini",
        "cost_usd":   0.0000187,
        "latency_ms": 489,
    },
    {
        "query":      "Design an observability architecture for AI across 3 regions with compliance requirements.",
        "complexity": "complex",
        "model":      "gpt-4o",
        "cost_usd":   0.0008420,
        "latency_ms": 1821,
    },
]

SIMULATED_SAVINGS = {
    "distribution": {"simple": 0.60, "moderate": 0.30, "complex": 0.10},
    "daily_requests": 10_000,
    "baseline_daily_cost_usd": 19.40,
    "routed_daily_cost_usd": 4.87,
    "daily_savings_usd": 14.53,
    "savings_percent": 74.9,
    "annual_savings_usd": 5_303,
}

SIMULATED_TRIAGE_ANALYSIS = {
    "root_cause": "Prompt template v1.2 regression — ANSWER_TEMPLATE changed 47 min ago reduced context grounding instructions",
    "confidence": "high",
    "remediation_steps": [
        "Roll back ANSWER_TEMPLATE to v1.1 (last known good, deployed 3h ago)",
        "Filter traces by template_version:v1.2 in DD Explorer to confirm blast radius",
        "If rollback reduces load, check for correlated deployment in the last 2h",
    ],
    "additional_data_needed": "Compare faithfulness distribution between template_version:v1.1 and v1.2 traces",
    "blast_radius": "~800 req/hour affected (all requests hitting the rag_support_pipeline workflow)",
    "summary": "Prompt template regression in v1.2 causing faithfulness drop from 0.91 → 0.52",
}


# ── Project runners ───────────────────────────────────────────────────────────

def run_project1(dry_run: bool) -> dict:
    header("Project 1 — LLM Trace Pipeline  (RAG Support Bot)")
    print()
    print(f"  {DIM}Demonstrates: @workflow → @llm → @task → @embedding span tree{RESET}")
    print(f"  {DIM}Model strategy: gpt-4o-mini (rewrite) + gpt-4o (answer){RESET}")
    print()

    q = SIMULATED_PIPELINE_OUTPUT["question"]
    step("User question", f'"{q[:55]}..."')

    if dry_run:
        thinking("Rewriting query (gpt-4o-mini)")
        time.sleep(0.3)
        result("→ rewritten query", SIMULATED_PIPELINE_OUTPUT["rewritten_query"])

        thinking("Embedding rewritten query (text-embedding-3-small)")
        time.sleep(0.2)
        result("→ embedding", "vector [1536 dims]", DIM)

        thinking("Retrieving top-3 docs by cosine similarity")
        time.sleep(0.2)
        result("→ docs retrieved", "kb-001 (0.94)  kb-002 (0.87)  kb-005 (0.71)")

        thinking("Generating answer (gpt-4o)")
        time.sleep(0.4)

        divider()
        print()
        print(f"  {BOLD}Answer:{RESET}")
        print(f"  {WHITE}{SIMULATED_PIPELINE_OUTPUT['answer']}{RESET}")
        print()
        result("Sources",    ", ".join(SIMULATED_PIPELINE_OUTPUT["sources"]))
        result("Tokens",     str(SIMULATED_PIPELINE_OUTPUT["tokens_used"]))
        result("Latency",    f"{SIMULATED_PIPELINE_OUTPUT['total_latency_ms']:.0f}ms")

        divider()
        print()
        step("Datadog spans emitted")
        emit("dd.llm.request.count",    "1  [span:rag_support_pipeline]")
        emit("dd.llm.request.duration", "1243ms  [span:rag_support_pipeline]")
        emit("llm.tokens.prompt",       "237  [model:gpt-4o]")
        emit("llm.tokens.completion",   "50   [model:gpt-4o]")
        emit("llm.cost.usd",            "$0.00283  [model:gpt-4o]")

    else:
        try:
            from project1_llm_trace_pipeline.app.rag_support_bot import (
                InMemoryVectorStore, run_rag_pipeline, seed_vector_store
            )
            store = InMemoryVectorStore()
            thinking("Seeding vector store")
            seed_vector_store(store)
            thinking("Running RAG pipeline")
            resp = run_rag_pipeline(q, store)
            SIMULATED_PIPELINE_OUTPUT.update({
                "answer": resp.answer,
                "rewritten_query": resp.rewritten_query,
                "sources": resp.sources,
                "tokens_used": resp.tokens_used,
                "total_latency_ms": resp.total_latency_ms,
            })
            ok("Pipeline complete")
            result("Tokens",  str(resp.tokens_used))
            result("Latency", f"{resp.total_latency_ms:.0f}ms")
            result("Sources", ", ".join(resp.sources))
        except Exception as e:
            warn(f"Live run failed ({e}) — showing simulated output")

    ok("Trace visible in: Datadog → LLM Observability Explorer → ml_app:rag-support-bot")
    return SIMULATED_PIPELINE_OUTPUT


def run_project2(dry_run: bool, p1_output: dict) -> dict:
    header("Project 2 — Evaluation Framework  (Quality Loop)")
    print()
    print(f"  {DIM}Evaluators: faithfulness · relevancy · completeness{RESET}")
    print(f"  {DIM}Scores → DogStatsD + LLMObs.submit_evaluation(){RESET}")
    print()

    step("Running eval suite on Project 1 output")
    print()

    if dry_run:
        time.sleep(0.3)
        for metric_name, scores in SIMULATED_EVAL_OUTPUT.items():
            if metric_name == "overall_pass":
                continue
            label_color = GREEN if scores["label"] == "pass" else (
                YELLOW if scores["label"] == "partial" else RED
            )
            bar_len = int(scores["score"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(
                f"  {BOLD}{metric_name:<15}{RESET}  "
                f"{CYAN}{bar}{RESET}  "
                f"{label_color}{scores['score']:.2f}{RESET}  "
                f"{DIM}{scores['reasoning']}{RESET}"
            )
        print()

        overall = SIMULATED_EVAL_OUTPUT["overall_pass"]
        color = GREEN if overall else RED
        symbol = "✓" if overall else "✗"
        print(f"  {color}{BOLD}{symbol} Overall: {'PASS' if overall else 'FAIL'}{RESET}")
        print()

        divider()
        step("Datadog metrics emitted")
        for m in ["faithfulness", "relevancy", "completeness"]:
            score = SIMULATED_EVAL_OUTPUT[m]["score"]
            emit(f"llm.eval.{m}.score", f"{score:.2f}  [ml_app:rag-support-bot, env:production]")
        emit("llm.eval.overall.pass", "1  [ml_app:rag-support-bot]")

    else:
        try:
            from project2_eval_framework.evaluators.eval_suite import run_eval_suite
            bundle = run_eval_suite(
                question=p1_output["question"],
                answer=p1_output["answer"],
                context="",
                env="development",
            )
            ok(f"Overall: {'PASS' if bundle.overall_pass else 'FAIL'}")
            for s in bundle.scores:
                result(s.metric_name, f"{s.score:.2f} ({s.label.value})")
        except Exception as e:
            warn(f"Live run failed ({e}) — showing simulated output")

    ok("Quality monitor: flames if faithfulness < 0.7 → PagerDuty fires")
    return SIMULATED_EVAL_OUTPUT


def run_project3(dry_run: bool) -> dict:
    header("Project 3 — Cost + Latency Optimizer  (Model Router)")
    print()
    print(f"  {DIM}Routes simple → gpt-4o-mini, complex → gpt-4o{RESET}")
    print(f"  {DIM}Tracks per-request cost + emits Datadog SLO metrics{RESET}")
    print()

    step("Routing decisions")
    print()

    for r in SIMULATED_ROUTING_RESULTS:
        complexity_color = {
            "simple": GREEN, "moderate": YELLOW, "complex": RED
        }.get(r["complexity"], WHITE)
        model_abbr = "mini" if "mini" in r["model"] else "gpt-4o"
        print(
            f"  {DIM}Q:{RESET} {r['query'][:58]:<58}\n"
            f"      {complexity_color}▸ {r['complexity']:<10}{RESET}  "
            f"{CYAN}{r['model']:<15}{RESET}  "
            f"{YELLOW}${r['cost_usd']:.7f}{RESET}  "
            f"{DIM}{r['latency_ms']}ms{RESET}"
        )
        print()

    divider()
    step("Cost simulation  (10,000 req/day)")
    print()
    s = SIMULATED_SAVINGS
    dist_str = " / ".join(f"{k}:{int(v*100)}%" for k, v in s["distribution"].items())
    result("Query distribution", dist_str)
    result("Baseline (all gpt-4o)", f"${s['baseline_daily_cost_usd']:.2f}/day")
    result("With routing",         f"${s['routed_daily_cost_usd']:.2f}/day")
    print()
    print(
        f"  {GREEN}{BOLD}  ${s['daily_savings_usd']:.2f}/day saved  "
        f"({s['savings_percent']}% cheaper)  →  "
        f"${s['annual_savings_usd']:,}/year{RESET}"
    )
    print()

    divider()
    step("Datadog SLO: 99% of requests < 2000ms")
    emit("llm.request.duration_ms",  "p95=1421ms  [model:gpt-4o-mini, complexity:simple]")
    emit("llm.request.duration_ms",  "p95=1893ms  [model:gpt-4o, complexity:complex]")
    emit("llm.cost.savings_percent", "74.9  [ml_app:rag-support-bot]")
    print()

    ok("Dashboard shows cost vs quality side-by-side — answers 'are we spending wisely?'")
    return SIMULATED_SAVINGS


def run_project4(dry_run: bool) -> dict:
    header("Project 4 — AI SRE Triage Bot  (Bits-style PoC)")
    print()
    print(f"  {DIM}Datadog monitor fires → webhook → span fetch → LLM analysis → Slack{RESET}")
    print(f"  {DIM}Pattern: same mechanism as Datadog Bits AI SRE (GA Nov 2024){RESET}")
    print()

    step("Simulating: [RAG Bot] Faithfulness score below threshold fires")
    print(f"  {DIM}  alert_type: error  |  value: 0.52  |  threshold: 0.70{RESET}")
    print()

    thinking("FastAPI /webhook/datadog receives payload")
    time.sleep(0.2)
    thinking("Queuing triage in BackgroundTask (returns 200 immediately)")
    time.sleep(0.2)
    thinking("Fetching failing spans from Datadog Spans API v2")
    time.sleep(0.3)
    result("→ spans fetched", "8 failing spans (last 15 min)")
    result("→ common pattern", "span.name:answer_generator  error_rate=62%")
    print()

    thinking("Sending span summary to GPT-4o for root-cause analysis")
    time.sleep(0.5)
    print()

    divider()
    analysis = SIMULATED_TRIAGE_ANALYSIS
    print(f"  {BOLD}Root-cause hypothesis:{RESET}")
    confidence_color = GREEN if analysis["confidence"] == "high" else YELLOW
    print(f"  {confidence_color}{BOLD}[{analysis['confidence'].upper()} CONFIDENCE]{RESET}")
    print(f"  {WHITE}{analysis['root_cause']}{RESET}")
    print()
    print(f"  {BOLD}Blast radius:{RESET}  {analysis['blast_radius']}")
    print()
    print(f"  {BOLD}Remediation steps:{RESET}")
    for i, step_text in enumerate(analysis["remediation_steps"], 1):
        print(f"    {CYAN}{i}.{RESET} {step_text}")
    print()

    divider()
    step("Slack Block Kit message sent")
    print(f"""
  {BG_DARK}{BOLD} 🚨 AI SRE Triage — [RAG Bot] Faithfulness score below threshold {RESET}
  {BG_DARK} {analysis['summary']:<63} {RESET}
  {BG_DARK} App: rag-support-bot  |  Spans analyzed: 8                          {RESET}
  {BG_DARK}                                                                      {RESET}
  {BG_DARK} Root Cause 🔴              Blast Radius                              {RESET}
  {BG_DARK} Prompt template v1.2 reg.  ~800 req/hour                             {RESET}
  {BG_DARK}                                                                      {RESET}
  {BG_DARK} Remediation Steps                                                    {RESET}
  {BG_DARK} 1. Roll back ANSWER_TEMPLATE to v1.1                                 {RESET}
  {BG_DARK} 2. Filter traces by template_version:v1.2                            {RESET}
  {BG_DARK} 3. Check correlated deploys in last 2h                               {RESET}
""")

    ok("On-call engineer has root cause + fix in Slack within 30s of alert")
    return analysis


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(dry_run: bool) -> None:
    header("Portfolio Summary")
    print()

    rows = [
        ("Project 1", "RAG Trace Pipeline",     "Full span tree  →  DD LLM Explorer"),
        ("Project 2", "Eval Framework",          "Quality metrics  →  DD Monitors"),
        ("Project 3", "Cost Router",             "74.9% cost savings  →  DD SLO"),
        ("Project 4", "AI SRE Triage Bot",       "30s root cause  →  Slack"),
    ]
    for proj, name, outcome in rows:
        print(f"  {GREEN}✓{RESET}  {BOLD}{proj}{RESET}  {name:<25}  {DIM}{outcome}{RESET}")

    print()
    if dry_run:
        warn("Dry-run mode — no real API calls were made")
        print(f"  {DIM}Set DD_API_KEY + OPENAI_API_KEY and remove --dry-run for live traces.{RESET}")
    else:
        ok("All traces visible in Datadog → LLM Observability Explorer")
        ok("All metrics visible in Datadog → Metrics Explorer → llm.*")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Observability Portfolio Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project", type=int, choices=[1, 2, 3, 4],
        help="Run only a specific project (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate all outputs without making real API calls"
    )
    args = parser.parse_args()

    dry = args.dry_run or not os.environ.get("OPENAI_API_KEY")
    if dry and not args.dry_run:
        print(f"\n{YELLOW}⚠  No OPENAI_API_KEY found — running in dry-run mode{RESET}")

    print()
    print(f"{BOLD}{WHITE}LLM Observability Portfolio{RESET}  {DIM}— Datadog PSA Interview Demo{RESET}")
    print(f"{DIM}{datetime.now().strftime('%Y-%m-%d %H:%M')}{RESET}")

    p1_output = SIMULATED_PIPELINE_OUTPUT

    run_all = args.project is None
    p = args.project

    if run_all or p == 1:
        p1_output = run_project1(dry)
    if run_all or p == 2:
        run_project2(dry, p1_output)
    if run_all or p == 3:
        run_project3(dry)
    if run_all or p == 4:
        run_project4(dry)

    if run_all:
        print_summary(dry)


if __name__ == "__main__":
    main()
