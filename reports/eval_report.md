# Guardrail-RAG Eval Report

## Configuration

- Generated (UTC): 2026-10-13T21:05:14+00:00
- Embedder: hash (lexical, not semantic)
- LLM: fake (extractive, deterministic)
- Reranker: lexical
- Faithfulness method: token-coverage heuristic
- Qdrant mode: local (:memory:), collection `guardrail_rag`
- Sparse IDF modifier: on
- Retrieval: k = 5 distinct documents, from 20 candidate chunks
- Answers: top_k = 5 chunks
- Python 3.11.15; qdrant-client 1.19.1; llama-index-core 0.14.25

## Results

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Retrieval recall@5 (hybrid+rerank) | 1.000 | >=0.80 (informational) | MET |
| MRR (hybrid+rerank) | 1.000 | - (informational) | INFO |
| Faithfulness (token-coverage heuristic) | 1.000 | >=0.80 | PASS |
| Cross-tenant leaks | 0 | ==0 | PASS |
| Injection pass rate | 1.000 | ==1.00 | PASS |
| PII in answers | 0 | ==0 | PASS |
| Refusal rate (golden set) | 0.000 | - (informational) | INFO |

## Retrieval ablation

| Config | recall@k | MRR |
|---|---|---|
| dense (no rerank) | 1.000 | 0.889 |
| sparse (no rerank) | 1.000 | 1.000 |
| hybrid (no rerank) | 1.000 | 0.979 |
| hybrid + rerank | 1.000 | 1.000 |

## Cross-tenant leak matrix

Each cell is `cases / leak findings` for an attacker (row) querying a target tenant (column).

| Attacker \ Target | acme | globex | initech |
|---|---|---|---|
| acme | - | 6 / 0 | 6 / 0 |
| globex | 6 / 0 | - | 6 / 0 |
| initech | 6 / 0 | 6 / 0 | - |

## Notes

- These numbers come from this run (`python -m guardrail_rag.eval.run`) with the embedder and LLM listed above. They are not copied from any other run.
- The hash embedder is lexical (feature hashing over words and character trigrams), not semantic. Recall with it reflects lexical overlap between query and chunk, not semantic similarity, so it does not estimate a semantic embedder.
- The fake LLM is extractive: it returns the sentence with the most question-token overlap. Faithfulness and refusal rate therefore describe that backend.
- Golden set: 24 questions. Leak suite: 36 cases. Injection suite: 13 cases.
