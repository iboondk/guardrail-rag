# Guardrail-RAG: Write-up

## Problem

A retrieval system shared by several customers needs two guarantees. One tenant's documents must
never appear in another tenant's answers, and untrusted text must not hijack the model or leak
personal data. A single missing filter leaks silently, so Guardrail-RAG makes both properties
explicit and checks them with a repeatable harness.

## Architecture

Documents are validated, redacted for PII, chunked, embedded as dense and sparse vectors, and
upserted into one Qdrant collection. Each point carries a `tenant_id` payload. The store builds
the tenant filter itself, so callers cannot search without one.

A question passes the same tenant check and an injection screen. A hybrid retriever runs a dense
and a sparse query, both filtered by tenant, fuses them with Reciprocal Rank Fusion, and reranks
the result. The retriever asserts that every chunk belongs to the requesting tenant. Context is
sanitized, placed in a hardened prompt, and sent to a pluggable LLM backend. The answer is checked
for the system-prompt marker and redacted for PII before it is returned with sources and flags.

## Key decisions

- **Own the tenant filter.** LlamaIndex supplies chunking and LLM adapters. Qdrant is called
  directly, so the isolation code stays short and auditable (ADR-0001).
- **Layered isolation.** Validation, a store with no unfiltered search, a per-chunk assertion and
  tenant-scoped point IDs must all fail before a leak can happen (ADR-0002).
- **Hybrid search with RRF.** Ranks are fused, not scores, so no score normalization is needed
  (ADR-0003).
- **Offline defaults.** Tests and the eval run need no network, keys or paid APIs (ADR-0004).

## Results

Numbers come from `python -m guardrail_rag.eval.run`, as recorded in `reports/eval_report.md`.
The corpus is synthetic: three tenants, 24 golden questions, 36 leak cases and 13 injection cases.

| Metric | Result |
|---|---|
| Recall@5 (hybrid + rerank) | 1.000 |
| MRR (hybrid + rerank) | 1.000 |
| Cross-tenant leaks | 0 of 36 cases |
| Injection pass rate | 13 of 13 |
| PII in answers | 0 |

Dense-only MRR is 0.889. The default embedder is lexical feature hashing and the LLM is
extractive, so these numbers show that the pipeline and isolation work on this corpus, not how a
semantic model would score.

## What I would do next

- Run the same harness with a semantic embedder and a hosted LLM, and publish both reports.
- Take the tenant from an authenticated principal instead of a function argument.
- Run the suites against Qdrant server mode in CI, where payload indexes take effect.
- Replace the regex PII and injection checks with a model-based detector, and measure its false
  positive rate on realistic documents.
