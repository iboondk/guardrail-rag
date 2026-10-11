"""Reciprocal Rank Fusion over ranked lists of scored chunks."""

from __future__ import annotations

from guardrail_rag.models import ScoredChunk


def rrf(ranked_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    """Fuse ranked lists with score = sum(1 / (k + rank)), rank starting at 1.

    Chunks are deduplicated by chunk_id. Each list contributes at most once per chunk
    (its first occurrence). The result is sorted by fused score, descending, with ties
    broken by chunk_id. Returned objects are new ScoredChunk instances carrying the fused score.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, float] = {}
    first_seen: dict[str, ScoredChunk] = {}
    for ranked in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            cid = item.chunk.chunk_id
            if cid in seen_in_list:
                continue
            seen_in_list.add(cid)
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(cid, item)
    ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))
    return [ScoredChunk(chunk=first_seen[cid].chunk, score=fused[cid]) for cid in ordered]
