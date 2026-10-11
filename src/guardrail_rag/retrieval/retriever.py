"""Hybrid retrieval: dense and sparse legs, RRF fusion, optional rerank, isolation check."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from guardrail_rag.ingest.embeddings import Embedder
from guardrail_rag.ingest.sparse import sparse_encode
from guardrail_rag.ingest.store import TenantVectorStore
from guardrail_rag.models import ScoredChunk, TenantIsolationError, validate_tenant_id
from guardrail_rag.retrieval.fusion import rrf
from guardrail_rag.retrieval.rerank import Reranker

if TYPE_CHECKING:
    from llama_index.core.retrievers import BaseRetriever

Mode = Literal["dense", "sparse", "hybrid"]
_MODES: frozenset[str] = frozenset({"dense", "sparse", "hybrid"})


class HybridRetriever:
    def __init__(
        self,
        store: TenantVectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
        top_k: int = 5,
        mode: Mode = "hybrid",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown retrieval mode: {mode!r}")
        if candidate_k <= 0 or top_k <= 0:
            raise ValueError("candidate_k and top_k must be positive")
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.mode = mode

    def retrieve(self, tenant_id: str, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        validate_tenant_id(tenant_id)
        k = self.top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k must be positive")

        ranked_lists: list[list[ScoredChunk]] = []
        if self.mode in ("dense", "hybrid"):
            vector = self.embedder.embed([query])[0]
            ranked_lists.append(self.store.search_dense(tenant_id, vector, self.candidate_k))
        if self.mode in ("sparse", "hybrid"):
            indices, values = sparse_encode(query)
            ranked_lists.append(
                self.store.search_sparse(tenant_id, indices, values, self.candidate_k)
            )

        fused = rrf(ranked_lists)
        if self.reranker is not None:
            results = self.reranker.rerank(query, fused, k)
        else:
            results = fused[:k]
        self._assert_isolation(tenant_id, results)
        return results

    @staticmethod
    def _assert_isolation(tenant_id: str, chunks: list[ScoredChunk]) -> None:
        for item in chunks:
            if item.chunk.tenant_id != tenant_id:
                raise TenantIsolationError(
                    f"chunk {item.chunk.chunk_id!r} belongs to tenant "
                    f"{item.chunk.tenant_id!r}, not {tenant_id!r}"
                )


def as_llamaindex_retriever(retriever: HybridRetriever, tenant_id: str) -> BaseRetriever:
    """Wrap a HybridRetriever as a LlamaIndex BaseRetriever bound to one tenant."""
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    validate_tenant_id(tenant_id)

    class TenantBoundRetriever(BaseRetriever):
        def __init__(self) -> None:
            super().__init__()
            self._retriever = retriever
            self._tenant_id = tenant_id

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            chunks = self._retriever.retrieve(self._tenant_id, query_bundle.query_str)
            return [
                NodeWithScore(
                    node=TextNode(
                        id_=c.chunk.chunk_id,
                        text=c.chunk.text,
                        metadata={"tenant_id": c.chunk.tenant_id, "doc_id": c.chunk.doc_id},
                    ),
                    score=c.score,
                )
                for c in chunks
            ]

    return TenantBoundRetriever()
