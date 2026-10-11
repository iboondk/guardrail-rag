# ADR-0003: Hybrid retrieval with Reciprocal Rank Fusion

- Status: Accepted
- Date: 2026-10-11

## Context

Dense vectors capture meaning but can miss exact terms such as product names or policy numbers.
A keyword leg covers those cases. Combining scores from different systems needs a normalization
step that is hard to get right without a model.

## Decision

- Run two tenant-filtered Qdrant queries: a dense query (cosine) and a sparse query (hashed-token
  TF with `Modifier.IDF`).
- Fuse the two ranked lists with Reciprocal Rank Fusion, `score = sum(1 / (60 + rank))`, in Python.
- Rerank the fused list with `LexicalReranker` by default. `CrossEncoderReranker` is an optional
  extra.

RRF needs only ranks, so it works the same way in local and server mode and needs no tuning of
score scales.

## Consequences

- The ablation in `reports/eval_report.md` (dense, sparse, hybrid, hybrid + rerank) can be
  reproduced by one command.
- If local Qdrant rejects `Modifier.IDF`, the store falls back to raw TF and logs a warning. The
  report records which mode ran.
- The default embedder is lexical, so the dense leg is not semantic. Results must not be read as
  a measure of a semantic embedder (see ADR-0004).
