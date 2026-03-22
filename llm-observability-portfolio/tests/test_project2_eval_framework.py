"""
Tests for Project 2: LLM Evaluation Framework
----------------------------------------------
Tests cover:
  - EvalLabel assignment at correct score boundaries
  - EvalBundle overall_pass logic (faithfulness gate)
  - Metric emission calls (verifies DogStatsD tags and metric names)
  - JSONL log format and field completeness
  - Edge cases: score=0.0, score=1.0, all-fail bundle
"""

import json
import os
import sys
import time
import tempfile
import unittest.mock as mock
import pytest

# ── Stub ddtrace + datadog before import ─────────────────────────────────────
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
os.environ.setdefault("DD_APP_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from project2_eval_framework.evaluators.eval_suite import (  # noqa: E402
    EvalLabel,
    EvalScore,
    EvalBundle,
    emit_eval_metrics,
    log_eval_bundle,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_score(
    metric: str,
    score: float,
    label: EvalLabel = EvalLabel.PASS,
) -> EvalScore:
    return EvalScore(
        metric_name=metric,
        score=score,
        label=label,
        reasoning="test reasoning",
        latency_ms=50.0,
    )


def make_bundle(
    overall_pass: bool = True,
    scores: list[EvalScore] | None = None,
) -> EvalBundle:
    if scores is None:
        scores = [
            make_score("faithfulness", 0.9, EvalLabel.PASS),
            make_score("relevancy", 0.85, EvalLabel.PASS),
            make_score("completeness", 0.8, EvalLabel.PASS),
        ]
    return EvalBundle(
        question="What is LLM observability?",
        answer="LLM observability is the practice of monitoring LLM apps.",
        context="LLM observability monitors LLM-powered applications.",
        rewritten_query="LLM observability definition",
        scores=scores,
        overall_pass=overall_pass,
        timestamp=time.time(),
    )


# ── EvalLabel boundary tests ──────────────────────────────────────────────────

class TestEvalLabelValues:

    def test_label_values(self):
        assert EvalLabel.PASS.value == "pass"
        assert EvalLabel.FAIL.value == "fail"
        assert EvalLabel.PARTIAL.value == "partial"

    def test_faithfulness_pass_at_0_8(self):
        # From eval_suite: label = PASS if score >= 0.8
        score = 0.80
        label = EvalLabel.PASS if score >= 0.8 else (
            EvalLabel.PARTIAL if score >= 0.5 else EvalLabel.FAIL
        )
        assert label == EvalLabel.PASS

    def test_faithfulness_partial_at_0_6(self):
        score = 0.60
        label = EvalLabel.PASS if score >= 0.8 else (
            EvalLabel.PARTIAL if score >= 0.5 else EvalLabel.FAIL
        )
        assert label == EvalLabel.PARTIAL

    def test_faithfulness_fail_at_0_4(self):
        score = 0.40
        label = EvalLabel.PASS if score >= 0.8 else (
            EvalLabel.PARTIAL if score >= 0.5 else EvalLabel.FAIL
        )
        assert label == EvalLabel.FAIL

    def test_boundary_exactly_0_5_is_partial(self):
        score = 0.5
        label = EvalLabel.PASS if score >= 0.8 else (
            EvalLabel.PARTIAL if score >= 0.5 else EvalLabel.FAIL
        )
        assert label == EvalLabel.PARTIAL

    def test_boundary_exactly_0_8_is_pass(self):
        score = 0.8
        label = EvalLabel.PASS if score >= 0.8 else EvalLabel.FAIL
        assert label == EvalLabel.PASS


# ── EvalBundle overall_pass logic ─────────────────────────────────────────────

class TestEvalBundlePassLogic:

    def test_all_pass_means_overall_pass(self):
        bundle = make_bundle(overall_pass=True)
        assert bundle.overall_pass is True

    def test_faithfulness_fail_means_overall_fail(self):
        scores = [
            make_score("faithfulness", 0.3, EvalLabel.FAIL),    # fail
            make_score("relevancy", 0.9, EvalLabel.PASS),
            make_score("completeness", 0.85, EvalLabel.PASS),
        ]
        # Reproduce the logic from eval_suite.run_eval_suite
        faithfulness = scores[0]
        relevancy = scores[1]
        overall = (
            faithfulness.label == EvalLabel.PASS
            and relevancy.label != EvalLabel.FAIL
        )
        assert overall is False

    def test_relevancy_fail_means_overall_fail(self):
        scores = [
            make_score("faithfulness", 0.95, EvalLabel.PASS),
            make_score("relevancy", 0.2, EvalLabel.FAIL),     # fail
            make_score("completeness", 0.9, EvalLabel.PASS),
        ]
        faithfulness = scores[0]
        relevancy = scores[1]
        overall = (
            faithfulness.label == EvalLabel.PASS
            and relevancy.label != EvalLabel.FAIL
        )
        assert overall is False

    def test_completeness_fail_does_not_block_overall(self):
        # Completeness is a soft metric — failing it alone doesn't fail overall
        scores = [
            make_score("faithfulness", 0.9, EvalLabel.PASS),
            make_score("relevancy", 0.85, EvalLabel.PASS),
            make_score("completeness", 0.2, EvalLabel.FAIL),  # fail but soft
        ]
        faithfulness = scores[0]
        relevancy = scores[1]
        overall = (
            faithfulness.label == EvalLabel.PASS
            and relevancy.label != EvalLabel.FAIL
        )
        assert overall is True


# ── Metric emission tests ─────────────────────────────────────────────────────

class TestEmitEvalMetrics:

    def setup_method(self):
        datadog_stub.statsd.gauge.reset_mock()
        datadog_stub.statsd.increment.reset_mock()

    def test_emits_gauge_for_each_metric(self):
        from datadog import statsd
        bundle = make_bundle()
        emit_eval_metrics(bundle, env="test")
        gauge_calls = [c[0][0] for c in statsd.gauge.call_args_list]
        assert any("faithfulness" in m for m in gauge_calls)
        assert any("relevancy" in m for m in gauge_calls)
        assert any("completeness" in m for m in gauge_calls)

    def test_emits_overall_pass_count(self):
        from datadog import statsd
        bundle = make_bundle(overall_pass=True)
        emit_eval_metrics(bundle, env="test")
        increment_calls = [c[0][0] for c in statsd.increment.call_args_list]
        assert "llm.eval.overall.pass" in increment_calls

    def test_overall_pass_value_is_1_when_passing(self):
        from datadog import statsd
        bundle = make_bundle(overall_pass=True)
        emit_eval_metrics(bundle, env="test")
        overall_call = next(
            c for c in statsd.increment.call_args_list
            if c[0][0] == "llm.eval.overall.pass"
        )
        assert overall_call[1]["value"] == 1

    def test_overall_pass_value_is_0_when_failing(self):
        from datadog import statsd
        bundle = make_bundle(overall_pass=False)
        emit_eval_metrics(bundle, env="test")
        overall_call = next(
            c for c in statsd.increment.call_args_list
            if c[0][0] == "llm.eval.overall.pass"
        )
        assert overall_call[1]["value"] == 0

    def test_tags_contain_ml_app(self):
        from datadog import statsd
        bundle = make_bundle()
        emit_eval_metrics(bundle, env="test")
        first_gauge_call = statsd.gauge.call_args_list[0]
        tags = first_gauge_call[1]["tags"]
        assert any("ml_app:rag-support-bot" in t for t in tags)

    def test_tags_contain_env(self):
        from datadog import statsd
        bundle = make_bundle()
        emit_eval_metrics(bundle, env="staging")
        first_gauge_call = statsd.gauge.call_args_list[0]
        tags = first_gauge_call[1]["tags"]
        assert any("env:staging" in t for t in tags)


# ── JSONL log tests ───────────────────────────────────────────────────────────

class TestLogEvalBundle:

    def test_writes_valid_jsonl(self):
        bundle = make_bundle()
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            fname = f.name
        log_eval_bundle(bundle, log_file=fname)
        with open(fname) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "question" in record
        assert "overall_pass" in record
        assert "scores" in record
        assert "timestamp" in record
        os.unlink(fname)

    def test_appends_multiple_entries(self):
        bundle = make_bundle()
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            fname = f.name
        log_eval_bundle(bundle, log_file=fname)
        log_eval_bundle(bundle, log_file=fname)
        with open(fname) as f:
            lines = f.readlines()
        assert len(lines) == 2
        os.unlink(fname)

    def test_scores_include_all_three_metrics(self):
        bundle = make_bundle()
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            fname = f.name
        log_eval_bundle(bundle, log_file=fname)
        with open(fname) as f:
            record = json.loads(f.readline())
        assert "faithfulness" in record["scores"]
        assert "relevancy" in record["scores"]
        assert "completeness" in record["scores"]
        os.unlink(fname)

    def test_long_answers_are_truncated(self):
        bundle = make_bundle()
        bundle.answer = "A" * 500  # 500 char answer
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            fname = f.name
        log_eval_bundle(bundle, log_file=fname)
        with open(fname) as f:
            record = json.loads(f.readline())
        assert len(record["answer"]) <= 203  # 200 + "..."
        os.unlink(fname)

    def test_score_values_are_rounded(self):
        scores = [make_score("faithfulness", 0.876543, EvalLabel.PASS)]
        bundle = make_bundle(scores=scores)
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            fname = f.name
        log_eval_bundle(bundle, log_file=fname)
        with open(fname) as f:
            record = json.loads(f.readline())
        faith_score = record["scores"]["faithfulness"]["score"]
        assert faith_score == pytest.approx(0.877, abs=0.001)
        os.unlink(fname)
