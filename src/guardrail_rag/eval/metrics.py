"""Retrieval and faithfulness metrics for the eval harness.

Retrieval metrics work on document ids. The retrieved list is de-duplicated (keeping the first
occurrence) before any slicing, so several chunks from one document count as one hit.
"""

from __future__ import annotations

import re

from guardrail_rag.generation.llm import LLMBackend
from guardrail_rag.generation.prompt import REFUSAL
from guardrail_rag.ingest.sparse import tokenize

_CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)")


def _dedupe(doc_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    """Fraction of relevant documents found in the top k distinct retrieved documents.

    Returns 0.0 when there are no relevant documents.
    """
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    top = _dedupe(retrieved_doc_ids)[:k]
    return len(set(top) & relevant) / len(relevant)


def mrr(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant document (1-based, distinct docs). 0.0 if none."""
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(_dedupe(retrieved_doc_ids), start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def faithfulness_heuristic(answer: str, contexts: list[str], threshold: float = 0.6) -> float:
    """Share of answer sentences that are supported by the contexts (token coverage).

    Citation markers such as [1] are removed first. A sentence is supported when at least
    `threshold` of its tokens (see tokenize) appear in the union of the context tokens.
    Sentences without any tokens make no claim and are skipped. A refusal, an empty answer,
    or an answer with no claim-bearing sentence returns 1.0.
    """
    cleaned = _CITATION_RE.sub("", answer).strip()
    if not cleaned or cleaned == REFUSAL:
        return 1.0
    context_tokens: set[str] = set()
    for context in contexts:
        context_tokens.update(tokenize(context))

    total = 0
    supported = 0
    for sentence in _split_sentences(cleaned):
        tokens = tokenize(sentence)
        if not tokens:
            continue
        total += 1
        covered = sum(1 for t in tokens if t in context_tokens)
        if covered / len(tokens) >= threshold:
            supported += 1
    if total == 0:
        return 1.0
    return supported / total


JUDGE_SYSTEM = (
    "You are a strict fact-checking judge. Score how well the ANSWER is supported by the "
    "CONTEXT. Reply with a single number between 0 and 1 and nothing else, where 1 means "
    "every claim is supported and 0 means none are."
)


def llm_judge_faithfulness(judge: LLMBackend, answer: str, contexts: list[str]) -> float:
    """Ask a judge LLM for a faithfulness score in [0, 1].

    The first number in the reply is used and clamped to [0, 1]. Raises ValueError when the
    reply has no number.
    """
    context_block = "\n\n".join(f"<doc>\n{c}\n</doc>" for c in contexts)
    user = f"CONTEXT:\n{context_block}\n\nANSWER:\n{answer}\n\nScore (0 to 1):"
    reply = judge.complete(JUDGE_SYSTEM, user)
    match = _NUMBER_RE.search(reply or "")
    if match is None:
        raise ValueError(f"judge reply contains no number: {reply!r}")
    value = float(match.group(0))
    return min(1.0, max(0.0, value))
