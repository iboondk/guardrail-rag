"""GuardedRAG: screen the query, retrieve, sanitize context, generate, and check the output."""

from __future__ import annotations

from guardrail_rag.generation.llm import LLMBackend
from guardrail_rag.generation.prompt import REFUSAL, SYSTEM_MARKER, build_messages
from guardrail_rag.guardrails.injection import screen_query
from guardrail_rag.models import Answer, validate_tenant_id
from guardrail_rag.pii import redact
from guardrail_rag.retrieval.retriever import HybridRetriever


class GuardedRAG:
    def __init__(self, retriever: HybridRetriever, llm: LLMBackend, top_k: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def answer(self, tenant_id: str, query: str) -> Answer:
        validate_tenant_id(tenant_id)
        flags = [f"query_injection:{name}" for name in screen_query(query)]

        chunks = self.retriever.retrieve(tenant_id, query, self.top_k)
        if not chunks:
            return Answer(REFUSAL, [], flags + ["no_context"])

        system, user, context_hits = build_messages(query, chunks)
        flags += [f"context_injection:{name}" for name in context_hits]

        raw = self.llm.complete(system, user)
        if SYSTEM_MARKER in raw:
            text = REFUSAL
            flags.append("system_prompt_leak_blocked")
        else:
            text = raw

        text, pii_counts = redact(text)
        if pii_counts:
            flags.append("output_pii_redacted")
        return Answer(text, chunks, flags, pii_counts)
