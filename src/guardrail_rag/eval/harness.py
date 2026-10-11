"""Eval harness: build a wired system, ingest the fixture corpus and detect cross-tenant leaks."""

from __future__ import annotations

from dataclasses import dataclass

from guardrail_rag.config import Settings
from guardrail_rag.eval.datasets import LeakCase, load_corpus
from guardrail_rag.generation import GuardedRAG, get_llm
from guardrail_rag.generation.llm import LLMBackend
from guardrail_rag.ingest import (
    Embedder,
    IngestionPipeline,
    IngestStats,
    TenantVectorStore,
    get_embedder,
)
from guardrail_rag.models import TenantIsolationError, validate_tenant_id
from guardrail_rag.retrieval import (
    HybridRetriever,
    Mode,
    NoopReranker,
    Reranker,
    get_reranker,
)


@dataclass
class System:
    settings: Settings
    store: TenantVectorStore
    embedder: Embedder
    retriever: HybridRetriever
    rag: GuardedRAG


def build_system(
    settings: Settings | None = None,
    mode: Mode = "hybrid",
    rerank: bool = True,
    llm: LLMBackend | None = None,
) -> System:
    """Wire store, embedder, retriever and GuardedRAG.

    With no settings, the store is in-memory (Qdrant ':memory:') and the defaults from the
    environment are used. Pass llm to inject a backend such as FakeLLM.
    """
    if settings is None:
        settings = Settings(_env_file=None, qdrant_url=None, qdrant_path=":memory:")
    store = TenantVectorStore(settings)
    embedder = get_embedder(settings)
    reranker: Reranker = get_reranker(settings) if rerank else NoopReranker()
    retriever = HybridRetriever(
        store,
        embedder,
        reranker=reranker,
        candidate_k=settings.candidate_k,
        top_k=settings.top_k,
        mode=mode,
    )
    backend = llm if llm is not None else get_llm(settings)
    rag = GuardedRAG(retriever, backend, top_k=settings.top_k)
    return System(
        settings=settings,
        store=store,
        embedder=embedder,
        retriever=retriever,
        rag=rag,
    )


def ingest_fixture_corpus(system: System) -> IngestStats:
    """Ingest every tenant's synthetic corpus through the normal ingestion pipeline."""
    pipeline = IngestionPipeline(system.settings, system.store, system.embedder)
    return pipeline.ingest(load_corpus())


def leak_findings(system: System, case: LeakCase) -> list[str]:
    """Run the attacker's query and describe every cross-tenant leak found.

    A leak is any of: a retrieved chunk whose tenant is not the attacker; the target canary in
    a retrieved chunk; the target canary in the answer text; or a TenantIsolationError raised
    while retrieving (recorded as a finding rather than propagated).
    """
    validate_tenant_id(case.attacker_tenant)
    validate_tenant_id(case.target_tenant)
    try:
        answer = system.rag.answer(case.attacker_tenant, case.query)
    except TenantIsolationError as exc:
        return [f"TenantIsolationError: {exc}"]

    findings: list[str] = []
    for item in answer.sources:
        chunk = item.chunk
        if chunk.tenant_id != case.attacker_tenant:
            findings.append(
                f"retrieved chunk {chunk.chunk_id!r} belongs to tenant {chunk.tenant_id!r}"
            )
        if case.canary in chunk.text:
            findings.append(f"target canary found in retrieved chunk {chunk.chunk_id!r}")
    if case.canary in answer.text:
        findings.append("target canary found in answer text")
    return findings
