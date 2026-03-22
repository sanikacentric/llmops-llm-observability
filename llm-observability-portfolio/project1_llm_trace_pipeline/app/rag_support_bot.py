"""
Project 1: LLM Trace Pipeline with Datadog LLM Observability
------------------------------------------------------------
A production-style RAG (Retrieval-Augmented Generation) support bot
fully instrumented with Datadog LLM Observability spans, prompt templates,
token cost tracking, and structured logging.

Architecture:
  User Query → Query Rewriter → Vector Retrieval → LLM Answer → Response
                     ↓                 ↓                ↓
              [DD LLM Span]    [DD Custom Span]  [DD LLM Span]
"""

import os
import time
import json
import logging
from typing import Optional
from dataclasses import dataclass, field

import openai
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm, workflow, task, embedding
from ddtrace import tracer, patch_all
import numpy as np

# ── Bootstrap ────────────────────────────────────────────────────────────────
patch_all()  # auto-instrument all supported libraries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Datadog LLM Observability — enable before any traced calls
LLMObs.enable(
    ml_app="rag-support-bot",          # groups traces in DD UI
    agentless_enabled=True,            # no DD Agent needed for dev
    api_key=os.environ["DD_API_KEY"],
    site=os.environ.get("DD_SITE", "datadoghq.com"),
)

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class RetrievalResult:
    documents: list[Document]
    scores: list[float]
    latency_ms: float


@dataclass
class BotResponse:
    answer: str
    sources: list[str]
    tokens_used: int
    total_latency_ms: float
    rewritten_query: str


# ── Prompt templates (versioned — tracked in DD) ──────────────────────────────
QUERY_REWRITE_TEMPLATE = """You are a search query optimizer.
Rewrite the user's question into a clear, keyword-rich search query.
Return ONLY the rewritten query, no explanation.

Original question: {question}"""

ANSWER_TEMPLATE = """You are a helpful support assistant. Use ONLY the provided 
context to answer the question. If the context doesn't contain the answer, 
say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


# ── Vector store (in-memory for demo; swap for Pinecone/Weaviate in prod) ────
class InMemoryVectorStore:
    """
    Simulates a vector store. In production, replace with:
      - Pinecone, Weaviate, pgvector, or Datadog-monitored Redis
    """

    def __init__(self):
        self._docs: list[Document] = []

    def add_documents(self, docs: list[Document]) -> None:
        self._docs.extend(docs)
        logger.info("Added %d documents to vector store", len(docs))

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        return float(np.dot(a_arr, b_arr) / denom) if denom > 0 else 0.0

    def search(self, query_embedding: list[float], top_k: int = 3) -> RetrievalResult:
        start = time.perf_counter()
        if not self._docs or not self._docs[0].embedding:
            # Return dummy results if store is empty
            return RetrievalResult(documents=[], scores=[], latency_ms=0)

        scored = [
            (doc, self._cosine_similarity(query_embedding, doc.embedding))
            for doc in self._docs
            if doc.embedding
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            documents=[d for d, _ in top],
            scores=[s for _, s in top],
            latency_ms=latency_ms,
        )


# ── Core traced functions ─────────────────────────────────────────────────────

@embedding(model_name="text-embedding-3-small", model_provider="openai")
def embed_text(text: str) -> list[float]:
    """
    Embed text using OpenAI. Decorated with @embedding so Datadog
    automatically captures model name, input, and vector dimensions.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    LLMObs.annotate(
        input_data=[{"text": text}],
        output_data=str(response.data[0].embedding[:5]) + "...",
        metadata={"total_tokens": response.usage.total_tokens, "dimension": 1536},
    )
    return response.data[0].embedding


@llm(model_name="gpt-4o-mini", model_provider="openai", name="query_rewriter")
def rewrite_query(original_question: str) -> str:
    """
    Rewrite the user's question for better retrieval.
    Decorated with @llm — Datadog captures prompt, completion, and token usage.
    """
    prompt = QUERY_REWRITE_TEMPLATE.format(question=original_question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    rewritten = response.choices[0].message.content.strip()

    # Annotate with structured data for the DD LLM Observability UI
    LLMObs.annotate(
        input_data=[{"role": "user", "content": prompt}],
        output_data=[{"role": "assistant", "content": rewritten}],
        metadata={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "template_version": "v1.0",
        },
        tags={"operation": "query_rewrite"},
    )
    return rewritten


@task(name="vector_retrieval")
def retrieve_documents(
    query_embedding: list[float],
    vector_store: InMemoryVectorStore,
    top_k: int = 3,
) -> RetrievalResult:
    """
    Retrieve top-k documents by cosine similarity.
    Decorated with @task — Datadog tracks this as a non-LLM span in the trace.
    """
    result = vector_store.search(query_embedding, top_k=top_k)

    LLMObs.annotate(
        input_data=[{"content": f"Embedding query (dim={len(query_embedding)})"}],
        output_data=[{"content": f"Retrieved {len(result.documents)} docs"}],
        metadata={
            "top_k": top_k,
            "retrieval_latency_ms": round(result.latency_ms, 2),
            "scores": [round(s, 4) for s in result.scores],
        },
    )
    return result


@llm(model_name="gpt-4o", model_provider="openai", name="answer_generator")
def generate_answer(question: str, context_docs: list[Document]) -> tuple[str, int]:
    """
    Generate the final answer from retrieved context.
    Uses gpt-4o (more capable) while rewriting used gpt-4o-mini (cheaper).
    This cost routing strategy is demonstrated in Project 3.
    """
    context = "\n\n---\n\n".join(
        f"[Source: {doc.id}]\n{doc.content}" for doc in context_docs
    )
    prompt = ANSWER_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise support assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    answer = response.choices[0].message.content.strip()
    total_tokens = response.usage.total_tokens

    LLMObs.annotate(
        input_data=[
            {"role": "system", "content": "You are a precise support assistant."},
            {"role": "user", "content": prompt},
        ],
        output_data=[{"role": "assistant", "content": answer}],
        metadata={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": total_tokens,
            "context_doc_count": len(context_docs),
            "template_version": "v1.0",
        },
        tags={"operation": "answer_generation", "model_tier": "premium"},
    )
    return answer, total_tokens


@workflow(name="rag_support_pipeline")
def run_rag_pipeline(
    user_question: str,
    vector_store: InMemoryVectorStore,
) -> BotResponse:
    """
    Full RAG pipeline. Decorated with @workflow — Datadog shows the complete
    trace tree: workflow → query_rewriter → vector_retrieval → answer_generator.

    This is the entry point customers would call in production.
    """
    pipeline_start = time.perf_counter()
    logger.info("RAG pipeline started | question='%s'", user_question)

    # Annotate the workflow span with the raw input
    LLMObs.annotate(
        input_data=[{"role": "user", "content": user_question}],
        tags={"env": os.environ.get("DD_ENV", "dev"), "version": "1.0.0"},
    )

    # Step 1: Rewrite query for better retrieval
    rewritten = rewrite_query(user_question)
    logger.info("Query rewritten: '%s'", rewritten)

    # Step 2: Embed the rewritten query
    query_embedding = embed_text(rewritten)

    # Step 3: Retrieve relevant documents
    retrieval = retrieve_documents(query_embedding, vector_store)

    # Step 4: Generate grounded answer
    if not retrieval.documents:
        answer = "I don't have enough information in my knowledge base to answer that."
        tokens_used = 0
    else:
        answer, tokens_used = generate_answer(user_question, retrieval.documents)

    total_latency_ms = (time.perf_counter() - pipeline_start) * 1000

    # Annotate the workflow with final output and cost metadata
    LLMObs.annotate(
        output_data=[{"role": "assistant", "content": answer}],
        metadata={
            "total_latency_ms": round(total_latency_ms, 2),
            "total_tokens": tokens_used,
            "docs_retrieved": len(retrieval.documents),
            "retrieval_latency_ms": round(retrieval.latency_ms, 2),
            "rewritten_query": rewritten,
        },
    )

    logger.info(
        "RAG pipeline complete | latency=%.0fms tokens=%d docs=%d",
        total_latency_ms,
        tokens_used,
        len(retrieval.documents),
    )

    return BotResponse(
        answer=answer,
        sources=[doc.id for doc in retrieval.documents],
        tokens_used=tokens_used,
        total_latency_ms=total_latency_ms,
        rewritten_query=rewritten,
    )


# ── Sample knowledge base ─────────────────────────────────────────────────────
SAMPLE_DOCS = [
    Document(
        id="kb-001",
        content=(
            "Datadog LLM Observability provides end-to-end tracing for LLM-powered "
            "applications. It captures spans for every LLM call, including model name, "
            "prompt, completion, token usage, latency, and errors. Traces can be viewed "
            "in the LLM Observability Explorer."
        ),
        metadata={"source": "docs", "topic": "overview"},
    ),
    Document(
        id="kb-002",
        content=(
            "To instrument a LangChain application, install ddtrace and call "
            "LLMObs.enable() before initializing your chain. The dd-trace library "
            "automatically patches LangChain, LlamaIndex, OpenAI, and Anthropic clients."
        ),
        metadata={"source": "docs", "topic": "instrumentation"},
    ),
    Document(
        id="kb-003",
        content=(
            "Token cost tracking in Datadog uses custom metrics. Emit "
            "llm.tokens.prompt, llm.tokens.completion, and llm.cost.usd as gauges "
            "using DogStatsD. You can then build cost dashboards and set budget alerts."
        ),
        metadata={"source": "docs", "topic": "cost"},
    ),
    Document(
        id="kb-004",
        content=(
            "LLM evaluations measure output quality. Common metrics include faithfulness "
            "(does the answer match the context?), relevancy (is the answer on-topic?), "
            "and correctness. Scores can be submitted to Datadog via LLMObs.submit_evaluation()."
        ),
        metadata={"source": "docs", "topic": "evaluations"},
    ),
    Document(
        id="kb-005",
        content=(
            "Prompt template versioning in Datadog LLM Observability allows you to track "
            "which prompt version was used for each trace. Pass template_version in the "
            "metadata when annotating spans. This enables A/B testing of prompts."
        ),
        metadata={"source": "docs", "topic": "prompt-management"},
    ),
]


def seed_vector_store(store: InMemoryVectorStore) -> None:
    """Embed and load sample documents into the vector store."""
    logger.info("Seeding vector store with %d documents...", len(SAMPLE_DOCS))
    for doc in SAMPLE_DOCS:
        doc.embedding = embed_text(doc.content)
    store.add_documents(SAMPLE_DOCS)
    logger.info("Vector store ready.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    store = InMemoryVectorStore()
    seed_vector_store(store)

    test_questions = [
        "How do I set up Datadog LLM Observability for my OpenAI app?",
        "How can I track token costs in Datadog?",
        "What is prompt template versioning?",
        "How do LLM evaluations work with Datadog?",
    ]

    print("\n" + "=" * 60)
    print("RAG Support Bot — Datadog LLM Observability Demo")
    print("=" * 60 + "\n")

    for question in test_questions:
        print(f"Q: {question}")
        response = run_rag_pipeline(question, store)
        print(f"A: {response.answer}")
        print(f"   Sources: {response.sources}")
        print(f"   Tokens: {response.tokens_used} | Latency: {response.total_latency_ms:.0f}ms")
        print(f"   Rewritten query: '{response.rewritten_query}'")
        print()
