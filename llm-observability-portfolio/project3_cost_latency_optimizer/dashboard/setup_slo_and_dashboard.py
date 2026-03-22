"""
Project 3: SLO + Cost Dashboard Setup
Creates the Datadog SLO and cost tracking dashboard via API.

Run once: python dashboard/setup_slo_and_dashboard.py
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


def create_latency_slo() -> Optional[str]:
    """
    Create a metric-based SLO: 95% of requests complete within 2 seconds.
    Returns the SLO ID for use in the dashboard widget.
    """
    slo_payload = {
        "name": "RAG Pipeline p95 Latency SLO",
        "description": (
            "95% of RAG pipeline requests must complete within 2000ms. "
            "Tracks the rag_support_pipeline workflow span."
        ),
        "type": "metric",
        "query": {
            "numerator": (
                "sum:dd.llm.request.count{ml_app:rag-support-bot,"
                "span.name:rag_support_pipeline,env:production,"
                "dd.llm.request.duration:<2000}"
            ),
            "denominator": (
                "sum:dd.llm.request.count{ml_app:rag-support-bot,"
                "span.name:rag_support_pipeline,env:production}"
            ),
        },
        "thresholds": [
            {"target": 99.0, "timeframe": "7d", "warning": 99.5},
            {"target": 99.0, "timeframe": "30d", "warning": 99.5},
        ],
        "tags": ["ml_app:rag-support-bot", "team:ml-platform", "slo:latency"],
    }

    resp = requests.post(
        f"https://api.{DD_SITE}/api/v1/slo",
        headers=HEADERS,
        data=json.dumps(slo_payload),
    )

    if resp.status_code == 200:
        slo_id = resp.json()["data"]["id"]
        print(f"✓ SLO created: {slo_id}")
        return slo_id
    else:
        print(f"✗ SLO creation failed: {resp.status_code} {resp.text}")
        return None


COST_DASHBOARD = {
    "title": "LLM Cost + Quality Dashboard — Cost-Optimized Router",
    "description": (
        "Tracks token cost, model routing distribution, and quality signal side-by-side. "
        "Use this to answer 'are we spending wisely?' and 'is cheaper also worse?'"
    ),
    "layout_type": "ordered",
    "widgets": [
        {
            "definition": {
                "type": "timeseries",
                "title": "Cost per request by model tier (USD)",
                "requests": [
                    {
                        "q": "avg:llm.cost.usd_per_request{ml_app:rag-support-bot} by {model_tier}",
                        "display_type": "line",
                        "style": {"palette": "warm"},
                    }
                ],
            }
        },
        {
            "definition": {
                "type": "timeseries",
                "title": "Request distribution by complexity",
                "requests": [
                    {
                        "q": "sum:llm.routing.simple{ml_app:rag-support-bot}.as_rate()",
                        "display_type": "bars",
                        "style": {"palette": "green"},
                        "metadata": [{"expression": "sum:llm.routing.simple{ml_app:rag-support-bot}.as_rate()", "alias_name": "simple"}],
                    },
                    {
                        "q": "sum:llm.routing.moderate{ml_app:rag-support-bot}.as_rate()",
                        "display_type": "bars",
                        "style": {"palette": "yellow"},
                        "metadata": [{"expression": "sum:llm.routing.moderate{ml_app:rag-support-bot}.as_rate()", "alias_name": "moderate"}],
                    },
                    {
                        "q": "sum:llm.routing.complex{ml_app:rag-support-bot}.as_rate()",
                        "display_type": "bars",
                        "style": {"palette": "orange"},
                        "metadata": [{"expression": "sum:llm.routing.complex{ml_app:rag-support-bot}.as_rate()", "alias_name": "complex"}],
                    },
                ],
            }
        },
        {
            "definition": {
                "type": "query_value",
                "title": "Projected daily cost savings vs all-premium",
                "requests": [
                    {
                        "q": "avg:llm.cost.savings_percent{ml_app:rag-support-bot}",
                        "aggregator": "last",
                        "conditional_formats": [
                            {"comparator": ">", "value": 50, "palette": "white_on_green"},
                            {"comparator": ">", "value": 25, "palette": "white_on_yellow"},
                        ],
                    }
                ],
                "suffix": "% cheaper",
            }
        },
        {
            "definition": {
                "type": "timeseries",
                "title": "p95 latency by model",
                "requests": [
                    {
                        "q": "p95:llm.request.duration_ms{ml_app:rag-support-bot} by {model}",
                        "display_type": "line",
                    }
                ],
                "markers": [
                    {"value": "y = 2000", "display_type": "error dashed", "label": "2s SLO"}
                ],
            }
        },
        {
            "definition": {
                "type": "timeseries",
                "title": "Eval quality score by model tier (cost vs quality tradeoff)",
                "requests": [
                    {
                        "q": "avg:llm.eval.faithfulness.score{ml_app:rag-support-bot} by {model_tier}",
                        "display_type": "line",
                        "metadata": [{"alias_name": "faithfulness"}],
                    },
                    {
                        "q": "avg:llm.eval.relevancy.score{ml_app:rag-support-bot} by {model_tier}",
                        "display_type": "line",
                        "metadata": [{"alias_name": "relevancy"}],
                    },
                ],
                "yaxis": {"min": "0", "max": "1"},
            }
        },
        {
            "definition": {
                "type": "toplist",
                "title": "Total token spend by model (last 24h)",
                "requests": [
                    {
                        "q": "sum:llm.cost.usd_total{ml_app:rag-support-bot} by {model}",
                        "aggregator": "sum",
                    }
                ],
            }
        },
    ],
    "template_variables": [
        {"name": "env", "default": "production", "prefix": "env"},
    ],
}


def create_dashboard() -> None:
    resp = requests.post(
        f"https://api.{DD_SITE}/api/v1/dashboard",
        headers=HEADERS,
        data=json.dumps(COST_DASHBOARD),
    )
    if resp.status_code == 200:
        dashboard_url = resp.json().get("url", "")
        print(f"✓ Dashboard created: https://app.{DD_SITE}{dashboard_url}")
    else:
        print(f"✗ Dashboard creation failed: {resp.status_code} {resp.text}")


if __name__ == "__main__":
    from typing import Optional
    print("Setting up SLO and cost dashboard...")
    slo_id = create_latency_slo()
    create_dashboard()
    if slo_id:
        print(f"\nSLO ID: {slo_id}")
        print("Add this to your cost dashboard widgets to show SLO burn rate alongside cost.")
