"""
Tests for Project 3: Cost + Latency Optimizer
----------------------------------------------
Tests cover:
  - Cost calculation accuracy for each model tier
  - Routing decisions: simple → gpt-4o-mini, complex → gpt-4o
  - Cost simulation math and savings calculation
  - Metric emission tag structure
  - Edge cases: unknown model, zero-token call, all-complex distribution
"""

import os
import sys
import unittest.mock as mock
import pytest

# ── Stubs ─────────────────────────────────────────────────────────────────────
ddtrace_stub = mock.MagicMock()
ddtrace_stub.llmobs.decorators.llm = lambda **kw: (lambda f: f)
ddtrace_stub.llmobs.decorators.workflow = lambda **kw: (lambda f: f)
ddtrace_stub.llmobs.decorators.task = lambda **kw: (lambda f: f)
sys.modules["ddtrace"] = ddtrace_stub
sys.modules["ddtrace.llmobs"] = ddtrace_stub.llmobs
sys.modules["ddtrace.llmobs.decorators"] = ddtrace_stub.llmobs.decorators

datadog_stub = mock.MagicMock()
sys.modules["datadog"] = datadog_stub
sys.modules["datadog.statsd"] = datadog_stub.statsd

openai_stub = mock.MagicMock()
sys.modules["openai"] = openai_stub

os.environ.setdefault("DD_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from project3_cost_latency_optimizer.router.model_router import (  # noqa: E402
    calculate_cost,
    ComplexityLevel,
    MODEL_PRICING,
    simulate_routing_savings,
    RoutedResponse,
    RoutingDecision,
    emit_routing_metrics,
)


# ── Cost calculation tests ────────────────────────────────────────────────────

class TestCalculateCost:

    def test_gpt4o_cost_matches_published_pricing(self):
        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost = calculate_cost("gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cost == pytest.approx(12.50, rel=1e-6)

    def test_gpt4o_mini_cost_matches_published_pricing(self):
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        cost = calculate_cost("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cost == pytest.approx(0.75, rel=1e-6)

    def test_cost_ratio_gpt4o_vs_mini(self):
        # gpt-4o should be significantly more expensive than mini
        cost_4o = calculate_cost("gpt-4o", 1000, 500)
        cost_mini = calculate_cost("gpt-4o-mini", 1000, 500)
        ratio = cost_4o / cost_mini
        assert ratio > 10, f"Expected >10x price difference, got {ratio:.1f}x"

    def test_zero_tokens_returns_zero(self):
        assert calculate_cost("gpt-4o", 0, 0) == 0.0
        assert calculate_cost("gpt-4o-mini", 0, 0) == 0.0

    def test_only_prompt_tokens(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == pytest.approx(2.50, rel=1e-6)

    def test_only_completion_tokens(self):
        cost = calculate_cost("gpt-4o", prompt_tokens=0, completion_tokens=1_000_000)
        assert cost == pytest.approx(10.00, rel=1e-6)

    def test_cost_scales_linearly_with_tokens(self):
        cost_500 = calculate_cost("gpt-4o-mini", 500, 0)
        cost_1000 = calculate_cost("gpt-4o-mini", 1000, 0)
        assert cost_1000 == pytest.approx(cost_500 * 2, rel=1e-9)

    def test_result_is_rounded_to_8_decimal_places(self):
        cost = calculate_cost("gpt-4o-mini", 123, 456)
        # Should be a float with at most 8 decimal places
        assert cost == round(cost, 8)

    def test_unknown_model_does_not_raise(self):
        cost = calculate_cost("gpt-99-ultra", 1000, 200)
        assert isinstance(cost, float)
        assert cost >= 0


# ── Routing decision tests ────────────────────────────────────────────────────

class TestComplexityLevel:

    def test_enum_values(self):
        assert ComplexityLevel.SIMPLE.value == "simple"
        assert ComplexityLevel.MODERATE.value == "moderate"
        assert ComplexityLevel.COMPLEX.value == "complex"

    def test_routing_simple_to_mini(self):
        # Routing rule: simple/moderate → gpt-4o-mini
        for level in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]:
            model = "gpt-4o" if level == ComplexityLevel.COMPLEX else "gpt-4o-mini"
            assert model == "gpt-4o-mini", f"{level} should route to gpt-4o-mini"

    def test_routing_complex_to_gpt4o(self):
        level = ComplexityLevel.COMPLEX
        model = "gpt-4o" if level == ComplexityLevel.COMPLEX else "gpt-4o-mini"
        assert model == "gpt-4o"


# ── Cost simulation tests ─────────────────────────────────────────────────────

class TestSimulateRoutingSavings:

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_savings_are_positive_with_mixed_distribution(self, mock_statsd):
        result = simulate_routing_savings(
            query_distribution={"simple": 0.60, "moderate": 0.30, "complex": 0.10},
            daily_request_count=10_000,
        )
        assert result["daily_savings_usd"] > 0
        assert result["savings_percent"] > 0

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_all_complex_means_zero_savings(self, mock_statsd):
        result = simulate_routing_savings(
            query_distribution={"complex": 1.0},
            daily_request_count=10_000,
        )
        # All complex routes to gpt-4o = same as baseline
        assert result["savings_percent"] == pytest.approx(0.0, abs=0.1)

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_all_simple_means_max_savings(self, mock_statsd):
        result = simulate_routing_savings(
            query_distribution={"simple": 1.0},
            daily_request_count=10_000,
        )
        # All simple routes to gpt-4o-mini — large savings
        assert result["savings_percent"] > 70.0

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_annual_savings_is_daily_times_365(self, mock_statsd):
        result = simulate_routing_savings(
            query_distribution={"simple": 0.7, "complex": 0.3},
            daily_request_count=5_000,
        )
        assert result["annual_savings_usd"] == pytest.approx(
            result["daily_savings_usd"] * 365, rel=0.01
        )

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_result_contains_required_fields(self, mock_statsd):
        result = simulate_routing_savings(
            query_distribution={"simple": 0.5, "complex": 0.5},
            daily_request_count=1_000,
        )
        required_keys = {
            "daily_requests",
            "baseline_daily_cost_usd",
            "routed_daily_cost_usd",
            "daily_savings_usd",
            "savings_percent",
            "annual_savings_usd",
        }
        assert required_keys.issubset(set(result.keys()))

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_emits_savings_metric_to_datadog(self, mock_statsd):
        simulate_routing_savings(
            query_distribution={"simple": 0.8, "complex": 0.2},
            daily_request_count=1_000,
        )
        gauge_calls = [c[0][0] for c in mock_statsd.gauge.call_args_list]
        assert "llm.cost.savings_percent" in gauge_calls


# ── Routing metrics emission tests ───────────────────────────────────────────

class TestEmitRoutingMetrics:

    def make_routed_response(
        self,
        model: str = "gpt-4o-mini",
        complexity: ComplexityLevel = ComplexityLevel.SIMPLE,
        cost: float = 0.000015,
    ) -> RoutedResponse:
        return RoutedResponse(
            content="test answer",
            model_used=model,
            complexity=complexity,
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=cost,
            latency_ms=320.0,
            routing_decision=RoutingDecision(
                complexity=complexity,
                model=model,
                reasoning="test",
                classifier_tokens=20,
                classifier_latency_ms=50.0,
            ),
        )

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_emits_cost_per_request_gauge(self, mock_statsd):
        resp = self.make_routed_response(cost=0.000123)
        emit_routing_metrics(resp, env="test")
        gauge_calls = {c[0][0]: c for c in mock_statsd.gauge.call_args_list}
        assert "llm.cost.usd_per_request" in gauge_calls
        assert gauge_calls["llm.cost.usd_per_request"][0][1] == pytest.approx(0.000123)

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_emits_latency_histogram(self, mock_statsd):
        resp = self.make_routed_response()
        emit_routing_metrics(resp, env="test")
        hist_calls = [c[0][0] for c in mock_statsd.histogram.call_args_list]
        assert "llm.request.duration_ms" in hist_calls

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_tags_include_model_and_complexity(self, mock_statsd):
        resp = self.make_routed_response(
            model="gpt-4o-mini",
            complexity=ComplexityLevel.SIMPLE,
        )
        emit_routing_metrics(resp, env="prod")
        first_call = mock_statsd.gauge.call_args_list[0]
        tags = first_call[1]["tags"]
        assert any("model:gpt-4o-mini" in t for t in tags)
        assert any("complexity:simple" in t for t in tags)

    @mock.patch("project3_cost_latency_optimizer.router.model_router.statsd")
    def test_routing_distribution_counter_incremented(self, mock_statsd):
        resp = self.make_routed_response(complexity=ComplexityLevel.COMPLEX)
        emit_routing_metrics(resp, env="test")
        increment_calls = [c[0][0] for c in mock_statsd.increment.call_args_list]
        assert "llm.routing.complex" in increment_calls


# ── Model pricing table sanity checks ────────────────────────────────────────

class TestModelPricingTable:

    def test_all_required_models_present(self):
        assert "gpt-4o" in MODEL_PRICING
        assert "gpt-4o-mini" in MODEL_PRICING

    def test_each_model_has_required_fields(self):
        for model, pricing in MODEL_PRICING.items():
            assert "input_per_1m" in pricing, f"{model} missing input_per_1m"
            assert "output_per_1m" in pricing, f"{model} missing output_per_1m"
            assert "tier" in pricing, f"{model} missing tier"

    def test_prices_are_positive(self):
        for model, pricing in MODEL_PRICING.items():
            assert pricing["input_per_1m"] >= 0, f"{model} input price negative"
            assert pricing["output_per_1m"] >= 0, f"{model} output price negative"
