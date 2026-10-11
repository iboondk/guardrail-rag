# ADR-0001: Use LlamaIndex only for chunking and LLM adapters

- Status: Accepted
- Date: 2026-10-11

## Context

The project needs chunking, LLM adapters (Ollama, Anthropic, Groq) and a retriever interface. A
full framework such as LlamaIndex could also own the vector store. Tenant isolation is the core
property being tested, so the code that builds tenant filters must be small and auditable.

## Decision

- Use `llama-index-core` for `SentenceSplitter`, the optional LLM integrations (imported lazily
  from their own extras) and the `BaseRetriever` adapter (`as_llamaindex_retriever`).
- Call `qdrant-client` directly for all storage and search. Do not use
  `llama-index-vector-stores-qdrant`.

## Consequences

- Every filter that reaches Qdrant is built in `TenantVectorStore`, in one file that a reviewer
  can read in full.
- We own the RRF fusion, the sparse encoder and the reranker interface. This is more code than a
  framework would provide, but it is all tested.
- The LlamaIndex dependency stays narrow. Importing the package never requires the LLM extras.
