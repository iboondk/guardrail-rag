# Guardrail-RAG: Architecture

A RAG platform that keeps tenants apart, with an eval harness that checks that it does.
Several tenants share one Qdrant collection. Every read goes through one retrieval path that
always applies the tenant filter. An eval harness measures retrieval quality and faithfulness,
and runs a cross-tenant leak suite that must report **0 leaks** plus a prompt-injection suite.

## 1. Goals and non-goals

Goals
- Ingest per-tenant documents, redact PII before anything is stored, and index dense and sparse vectors in Qdrant.
- Retrieve with a hybrid search (dense + sparse, fused with RRF), then rerank. The tenant filter is applied inside Qdrant on every query, and the result is re-checked in Python.
- Generate answers with a pluggable LLM backend (`fake`, `ollama`, `anthropic`, `groq`), chosen through env config.
- Guardrails: PII redaction on the way in and on the way out, prompt-injection detection and context sanitizing, and a hardened prompt template.
- Eval harness: retrieval recall@k and MRR, faithfulness, a cross-tenant leak count (must be 0), an injection pass rate, and PII leaks in answers. It writes a Markdown and JSON report.

Non-goals
- No real cloud deployment, no paid API calls in tests, and no real credentials (only `.env.example`).
- No auth server or UI. The tenant identity is an explicit, validated argument. A real system would take it from an authenticated principal, and this is documented.

## 2. Tech choices

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11 (virtualenv at `.venv`) | Reproducible dev environment via `pip install -e ".[dev]"` |
| Framework | **LlamaIndex** (`llama-index-core`) | `SentenceSplitter` for chunking, LLM integrations for Ollama, Anthropic and Groq, and a `BaseRetriever` adapter. We call `qdrant-client` directly (not `llama-index-vector-stores-qdrant`) so that tenant filtering is code we own and can audit |
| Vector DB | Qdrant through `qdrant-client`. **Local mode** (`:memory:` or a path) for tests and evals, server mode (`GR_QDRANT_URL`) through docker-compose | Tests run without Docker |
| Dense embeddings | `HashEmbedder` (deterministic feature hashing, no downloads) by default; `OllamaEmbedder` (`all-minilm:l6-v2`, 384-d, already pulled) as an option | Offline and reproducible CI, plus a real semantic option |
| Sparse | A hashed-token TF sparse encoder we write ourselves; Qdrant sparse vectors with `Modifier.IDF` | BM25-like keyword leg with no model download |
| Fusion | Reciprocal Rank Fusion in Python (k=60) over two tenant-filtered Qdrant queries | Simple and deterministic; works the same in local and server mode |
| Reranker | `LexicalReranker` (default, no deps); `CrossEncoderReranker` (optional extra `rerank`, sentence-transformers) | Optional heavy dependency |
| PII | Regex + Luhn detectors (EMAIL, PHONE, SSN, CREDIT_CARD, IPV4, IBAN) | No spaCy or Presidio weight; easy to test |
| LLM backends | `FakeLLM` (deterministic, extractive), plus `llama-index-llms-ollama`, `-anthropic`, `-groq`, imported lazily | Tests never touch the network |
| Config | `pydantic-settings`, env prefix `GR_` | One place for the env config |
| Tests and lint | `pytest`, `ruff` | |

Setup and checks (`make all` runs the same steps end to end):
```
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest -q && .venv/bin/ruff check .
```

## 3. Architecture

```
            ingest(tenant, docs)                              answer(tenant, query)
                   |                                                   |
        validate_tenant_id                                   validate_tenant_id
                   |                                          screen_query (injection flags)
        pii.redact(text)                                               |
                   |                                    HybridRetriever.retrieve(tenant, q)
        SentenceSplitter (LlamaIndex)                   |-- store.search_dense(tenant, ...)  --+
                   |                                    |-- store.search_sparse(tenant, ...) --+--> rrf()
        Embedder.embed + sparse_encode                  |                                       |
                   |                                    |   Reranker.rerank -> top_k            |
        TenantVectorStore.upsert                        |   assert all chunk.tenant_id == tenant (TenantIsolationError)
        (payload: tenant_id, doc_id, chunk_id, text)    |
                   |                                    sanitize_context (injection) -> build_messages
              Qdrant collection                         LLMBackend.complete(system, user)
   named vectors: "dense" (cosine), "sparse" (IDF)      pii.redact(output) + output checks -> Answer(text, sources, flags)
   payload index: tenant_id (keyword, is_tenant)
```

Tenant isolation, layered:
1. `TenantVectorStore` has no public search method that runs without a `tenant_id`. The filter `must: tenant_id == X` is built inside the store, never by callers.
2. `validate_tenant_id` rejects anything outside `^[a-z0-9][a-z0-9_-]{1,63}$` (including empty, `*`, whitespace, and injection-like strings).
3. The retriever checks every returned chunk's tenant again and raises `TenantIsolationError` rather than filter results quietly.
4. Point IDs are `uuid5(tenant:doc:idx)`, so two tenants with the same `doc_id` never collide or overwrite each other.
5. The leak suite plants a **canary** token in each tenant's corpus and runs adversarial queries from every other tenant. Any canary in retrieved chunks or answers counts as a leak, and the target is 0.

Prompt-injection defence:
- `detect_injection` uses regex patterns ("ignore (all|previous) instructions", "system prompt", "you are now", role tags, "reveal", fake `</doc>` delimiters, and so on).
- `sanitize_context` replaces matching lines with `[REMOVED: possible prompt injection]` and escapes `<doc`/`</doc` tags inside chunk text.
- The system prompt says that context is untrusted data, says to answer only from it, and includes the fixed refusal string.
- Output checks: the answer must not contain the system-prompt marker `GR-SYS-7f3a`, and PII is redacted from the output.

## 4. Directory layout

```
guardrail-rag/
  pyproject.toml            # package "guardrail-rag", src layout, extras: dev, ollama, anthropic, groq, rerank
  .env.example  .gitignore  LICENSE (MIT, iboondk)  docker-compose.yml (qdrant only)
  docs/                     # architecture.md (this file) and ADR-*.md decision records
  src/guardrail_rag/
    __init__.py
    config.py               # Settings, get_settings
    models.py               # Document, Chunk, ScoredChunk, Answer, errors, validate_tenant_id
    pii.py                  # find_pii, redact
    ingest/
      __init__.py
      embeddings.py         # Embedder protocol, HashEmbedder, OllamaEmbedder, get_embedder
      sparse.py             # tokenize, sparse_encode
      store.py              # TenantVectorStore
      pipeline.py           # IngestionPipeline, IngestStats, load_directory
    retrieval/
      __init__.py
      fusion.py             # rrf
      rerank.py             # Reranker protocol, NoopReranker, LexicalReranker, CrossEncoderReranker, get_reranker
      retriever.py          # HybridRetriever, TenantIsolationError, as_llamaindex_retriever
    guardrails/
      __init__.py
      injection.py          # detect_injection, sanitize_context, screen_query
    generation/
      __init__.py
      prompt.py             # SYSTEM_MARKER, REFUSAL, build_messages
      llm.py                # LLMBackend protocol, FakeLLM, OllamaLLM, AnthropicLLM, GroqLLM, get_llm
      pipeline.py           # GuardedRAG
    cli.py                  # ingest / ask subcommands
    eval/
      __init__.py
      datasets.py           # loaders for fixtures
      harness.py            # build_system(settings) -> (store, ingestion, rag), ingest_fixture_corpus
      metrics.py            # recall_at_k, mrr, faithfulness_heuristic
      run.py                # python -m guardrail_rag.eval.run -> reports/eval_report.{md,json}
      data/
        corpus/<tenant>/*.md        # acme, globex, initech (synthetic)
        golden.jsonl
        leak_cases.jsonl
        injection_cases.jsonl
  tests/
    test_config.py test_pii.py test_ingest.py test_store.py test_retrieval.py
    test_guardrails.py test_generation.py test_leak_suite.py test_injection_suite.py
    test_metrics.py test_eval_run.py
  reports/                  # generated; eval_report.md is committed
```

## 5. Components and how each is verified

| Layer | Scope | Check |
|---|---|---|
| 1 | Config, shared models, PII redaction | `pytest tests/test_config.py tests/test_pii.py`; ruff clean |
| 2 | Ingestion: embedder, sparse encoder, tenant-scoped Qdrant store | Per-tenant counts correct; no PII in stored payloads |
| 3 | Hybrid retrieval, RRF, reranker, isolation assertion | Right doc ranks first; tenants never see each other's chunks |
| 4 | Guardrails and generation: injection sanitizer, prompt, LLM backends, CLI | Fake-backend answer with citations; injected chunks neutralized |
| 5 | Eval datasets, leak and injection suites | Leak suite reports 0 leaks; injection suite passes |
| 6 | Metrics, eval runner, report | `python -m guardrail_rag.eval.run` writes `reports/eval_report.md` with computed numbers and an ablation table; non-zero exit if leaks > 0 |

## 6. Known limitations and notes

- Some Qdrant local-mode features (sparse `Modifier.IDF`, payload indexes) can differ from server mode. Local mode ignores payload indexes with a warning, which is fine. If `Modifier.IDF` is rejected in local mode, the store falls back to raw TF values, and the report notes it.
- `HashEmbedder` is not semantic, so recall numbers with it reflect lexical overlap. The report states which embedder and LLM produced the numbers. All numbers come from `eval/run.py`.
- Faithfulness defaults to a transparent token-coverage heuristic. An LLM judge is used only when `GR_JUDGE_BACKEND` is set to a working backend. The report labels which method ran.
- Docker is optional: `docker-compose.yml` starts a Qdrant server, but local mode works without it.
