"""
Tests for Project 1: RAG Support Bot Pipeline
----------------------------------------------
Strategy: mock all external calls (OpenAI, Datadog) so tests run
without credentials and execute in CI in < 5 seconds.

Tests cover:
  - Cost calculator math
  - Vector store similarity ranking
  - Pipeline routing logic (correct model per stage)
  - BotResponse structure and field types
  - Edge cases: empty store, no matching docs
"""

import pytest
from unittest.mock import patch, MagicMock, call
from dataclasses import dataclass
import sys
import os

# ── Stub out ddtrace before importing the module ──────────────────────────────
# ddtrace decorators are no-ops in test; we test the underlying logic.
import unittest.mock as mock

ddtrace_stub = mock.MagicMock()
ddtrace_stub.llmobs.LLMObs = mock.MagicMock()
ddtrace_stub.llmobs.decorators.llm = lambda **kw: (lambda f: f)
ddtrace_stub.llmobs.decorators.workflow = lambda **kw: (lambda f: f)
ddtrace_stub.llmobs.decorators.task = lambda **kw: (lambda f: f)
ddtrace_stub.llmobs.decorators.embedding = lambda **kw: (lambda f: f)
ddtrace_stub.patch_all = mock.MagicMock()

sys.modules["ddtrace"] = ddtrace_stub
sys.modules["ddtrace.llmobs"] = ddtrace_stub.llmobs
sys.modules["ddtrace.llmobs.decorators"] = ddtrace_stub.llmobs.decorators

# Stub openai
openai_stub = mock.MagicMock()
sys.modules["openai"] = openai_stub

os.environ.setdefault("DD_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Now safe to import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from project1_llm_trace_pipeline.app.rag_support_bot import (  # noqa: E402
    Document,
    InMemoryVectorStore,
    RetrievalResult,
    BotResponse,
    calculate_cost,
    SAMPLE_DOCS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_doc(doc_id: str, embedding: list[float]) -> Document:
    return Document(id=doc_id, content=f"Content for {doc_id}", embedding=embedding)


@pytest.fixture
def populated_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add_documents([
        make_doc("doc-a", [1.0, 0.0, 0.0]),
        make_doc("doc-b", [0.0, 1.0, 0.0]),
        make_doc("doc-c", [0.0, 0.0, 1.0]),
        make_doc("doc-d", [0.7, 0.7, 0.0]),   # close to doc-a and doc-b
    ])
    return store


# ── Vector store tests ────────────────────────────────────────────────────────

class TestInMemoryVectorStore:

    def test_search_returns_top_k(self, populated_store):
        query = [1.0, 0.0, 0.0]  # identical to doc-a
        result = populated_store.search(query, top_k=2)
        assert len(result.documents) == 2
        assert len(result.scores) == 2

    def test_search_ranks_by_similarity(self, populated_store):
        query = [1.0, 0.0, 0.0]  # identical to doc-a
        result = populated_store.search(query, top_k=3)
        # doc-a should be ranked first (cosine sim = 1.0)
        assert result.documents[0].id == "doc-a"
        assert result.scores[0] == pytest.approx(1.0, abs=1e-6)

    def test_search_descending_scores(self, populated_store):
        query = [1.0, 0.0, 0.0]
        result = populated_store.search(query, top_k=4)
        scores = result.scores
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), \
            f"Scores not descending: {scores}"

    def test_search_empty_store(self):
        store = InMemoryVectorStore()
        result = store.search([1.0, 0.0], top_k=3)
        assert result.documents == []
        assert result.scores == []

    def test_search_top_k_greater_than_doc_count(self, populated_store):
        result = populated_store.search([1.0, 0.0, 0.0], top_k=100)
        assert len(result.documents) == 4  # only 4 docs exist

    def test_search_zero_vector_handled(self, populated_store):
        # Zero vector → all similarities should be 0 (no ZeroDivisionError)
        result = populated_store.search([0.0, 0.0, 0.0], top_k=2)
        assert all(s == 0.0 for s in result.scores)

    def test_add_documents_increments_count(self):
        store = InMemoryVectorStore()
        store.add_documents([make_doc("x", [1.0])])
        store.add_documents([make_doc("y", [0.0])])
        result = store.search([1.0], top_k=10)
        assert len(result.documents) == 2

    def test_cosine_similarity_orthogonal(self):
        sim = InMemoryVectorStore._cosine_similarity([1, 0], [0, 1])
        assert sim == pytest.approx(0.0, abs=1e-9)

    def test_cosine_similarity_identical(self):
        sim = InMemoryVectorStore._cosine_similarity([3, 4], [3, 4])
        assert sim == pytest.approx(1.0, abs=1e-9)

    def test_cosine_similarity_opposite(self):
        sim = InMemoryVectorStore._cosine_similarity([1, 0], [-1, 0])
        assert sim == pytest.approx(-1.0, abs=1e-9)

    def test_retrieval_result_has_latency(self, populated_store):
        result = populated_store.search([1.0, 0.0, 0.0], top_k=1)
        assert result.latency_ms >= 0


# ── Document model tests ──────────────────────────────────────────────────────

class TestDocumentModel:

    def test_document_defaults(self):
        doc = Document(id="test", content="hello")
        assert doc.metadata == {}
        assert doc.embedding is None

    def test_sample_docs_have_content(self):
        for doc in SAMPLE_DOCS:
            assert doc.id
            assert len(doc.content) > 20, f"Doc {doc.id} content too short"
            assert "source" in doc.metadata

    def test_bot_response_fields(self):
        resp = BotResponse(
            answer="test answer",
            sources=["kb-001", "kb-002"],
            tokens_used=150,
            total_latency_ms=342.5,
            rewritten_query="rewritten question",
        )
        assert resp.tokens_used == 150
        assert len(resp.sources) == 2
        assert resp.total_latency_ms == pytest.approx(342.5)


# ── Cost calculator tests ─────────────────────────────────────────────────────

class TestCostCalculator:

    def test_gpt4o_cost_calculation(self):
        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_gpt4o_mini_cheaper_than_gpt4o(self):
        cost_mini = calculate_cost("gpt-4o-mini", 1000, 500)
        cost_full = calculate_cost("gpt-4o", 1000, 500)
        assert cost_mini < cost_full

    def test_zero_tokens_zero_cost(self):
        assert calculate_cost("gpt-4o", 0, 0) == 0.0

    def test_unknown_model_defaults_to_mini(self):
        # Should not raise; falls back to gpt-4o-mini pricing
        cost = calculate_cost("unknown-model-xyz", 1000, 200)
        assert cost > 0

    def test_cost_scales_linearly(self):
        cost_1k = calculate_cost("gpt-4o-mini", 1000, 0)
        cost_2k = calculate_cost("gpt-4o-mini", 2000, 0)
        assert cost_2k == pytest.approx(cost_1k * 2, rel=1e-9)


# ── Integration-style pipeline tests (all external calls mocked) ──────────────

class TestRagPipelineMocked:
    """
    Tests the full pipeline flow with all I/O mocked.
    Verifies correct orchestration without making real API calls.
    """

    @patch("project1_llm_trace_pipeline.app.rag_support_bot.embed_text")
    @patch("project1_llm_trace_pipeline.app.rag_support_bot.rewrite_query")
    @patch("project1_llm_trace_pipeline.app.rag_support_bot.retrieve_documents")
    @patch("project1_llm_trace_pipeline.app.rag_support_bot.generate_answer")
    def test_pipeline_calls_all_stages(
        self,
        mock_generate,
        mock_retrieve,
        mock_rewrite,
        mock_embed,
    ):
        mock_rewrite.return_value = "rewritten query"
        mock_embed.return_value = [0.1] * 1536
        mock_retrieve.return_value = RetrievalResult(
            documents=[make_doc("kb-001", [0.1] * 1536)],
            scores=[0.92],
            latency_ms=12.0,
        )
        mock_generate.return_value = ("The answer is X.", 180)

        from project1_llm_trace_pipeline.app.rag_support_bot import run_rag_pipeline
        store = InMemoryVectorStore()
        result = run_rag_pipeline("What is observability?", store)

        mock_rewrite.assert_called_once()
        mock_embed.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()

        assert result.answer == "The answer is X."
        assert result.tokens_used == 180
        assert "kb-001" in result.sources
        assert result.rewritten_query == "rewritten query"

    @patch("project1_llm_trace_pipeline.app.rag_support_bot.embed_text")
    @patch("project1_llm_trace_pipeline.app.rag_support_bot.rewrite_query")
    @patch("project1_llm_trace_pipeline.app.rag_support_bot.retrieve_documents")
    @patch("project1_llm_trace_pipeline.app.rag_support_bot.generate_answer")
    def test_pipeline_handles_empty_retrieval(
        self,
        mock_generate,
        mock_retrieve,
        mock_rewrite,
        mock_embed,
    ):
        mock_rewrite.return_value = "rewritten"
        mock_embed.return_value = [0.0] * 1536
        mock_retrieve.return_value = RetrievalResult(
            documents=[], scores=[], latency_ms=5.0
        )

        from project1_llm_trace_pipeline.app.rag_support_bot import run_rag_pipeline
        store = InMemoryVectorStore()
        result = run_rag_pipeline("Unknown topic", store)

        mock_generate.assert_not_called()
        assert "don't have enough information" in result.answer.lower()
        assert result.tokens_used == 0
        assert result.sources == []

    @patch("project1_llm_trace_pipeline.app.rag_support_bot.embed_text")
    @patch("project1_llm_trace_pipeline.app.rag_support_bot.rewrite_query")
    @patch("project1_llm_trace_pipeline.app.rag_support_bot.retrieve_documents")
    @patch("project1_llm_trace_pipeline.app.rag_support_bot.generate_answer")
    def test_pipeline_returns_correct_latency_type(
        self,
        mock_generate,
        mock_retrieve,
        mock_rewrite,
        mock_embed,
    ):
        mock_rewrite.return_value = "q"
        mock_embed.return_value = [1.0]
        mock_retrieve.return_value = RetrievalResult(
            documents=[make_doc("d1", [1.0])], scores=[0.9], latency_ms=10.0
        )
        mock_generate.return_value = ("answer", 100)

        from project1_llm_trace_pipeline.app.rag_support_bot import run_rag_pipeline
        result = run_rag_pipeline("test", InMemoryVectorStore())
        assert isinstance(result.total_latency_ms, float)
        assert result.total_latency_ms > 0
