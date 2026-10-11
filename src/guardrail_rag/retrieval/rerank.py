"""Rerankers: order a fused candidate list and keep the top_k."""

from __future__ import annotations

from typing import Protocol

from guardrail_rag.config import Settings
from guardrail_rag.ingest.sparse import tokenize
from guardrail_rag.models import ConfigError, ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class NoopReranker:
    """Keeps the incoming order and truncates to top_k."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return list(chunks[:top_k])


class LexicalReranker:
    """score = 0.7 * query-term overlap + 0.3 * prior, where prior is the chunk score
    divided by the best score in the list (0 when that best score is not positive).
    The returned ScoredChunk objects carry the rerank score. The sort is stable.
    """

    OVERLAP_WEIGHT = 0.7
    PRIOR_WEIGHT = 0.3

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        query_terms = set(tokenize(query))
        best = max(c.score for c in chunks)
        scored: list[ScoredChunk] = []
        for item in chunks:
            text_terms = set(tokenize(item.chunk.text))
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            prior = item.score / best if best > 0 else 0.0
            score = self.OVERLAP_WEIGHT * overlap + self.PRIOR_WEIGHT * prior
            scored.append(ScoredChunk(chunk=item.chunk, score=score))
        scored.sort(key=lambda c: -c.score)
        return scored[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker from sentence-transformers (optional extra 'rerank')."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ConfigError("install extra 'rerank'") from exc
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        pairs = [(query, c.chunk.text) for c in chunks]
        scores = self._model.predict(pairs)
        scored = [
            ScoredChunk(chunk=c.chunk, score=float(s)) for c, s in zip(chunks, scores, strict=True)
        ]
        scored.sort(key=lambda c: -c.score)
        return scored[:top_k]


def get_reranker(settings: Settings) -> Reranker:
    if settings.rerank_backend == "none":
        return NoopReranker()
    if settings.rerank_backend == "lexical":
        return LexicalReranker()
    if settings.rerank_backend == "cross_encoder":
        return CrossEncoderReranker(settings.rerank_model)
    raise ConfigError(f"unknown rerank backend: {settings.rerank_backend!r}")
