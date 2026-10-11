# ADR-0004: Offline defaults and heuristic guardrails

- Status: Accepted
- Date: 2026-10-11

## Context

Tests and CI must be reproducible and must not spend money or need keys. The guardrails must be
simple enough to read and test, and they must not add heavy NLP dependencies such as spaCy or
Presidio.

## Decision

- Defaults are `GR_EMBED_BACKEND=hash` (feature hashing over words and character trigrams, no
  downloads), `GR_LLM_BACKEND=fake` (deterministic, extractive) and `GR_RERANK_BACKEND=lexical`.
  Semantic and hosted backends are opt-in through environment variables and imported lazily.
- PII detection uses regex with a Luhn check for cards and format checks for IBAN, SSN and IPv4.
- Injection detection uses a fixed set of regex patterns. `sanitize_context` removes matching
  lines and escapes `<doc` tags. The system prompt carries a marker that must not appear in
  answers.

## Consequences

- The eval numbers describe the defaults, and the report says so. Recall with the hash embedder
  reflects lexical overlap, and faithfulness uses a token-coverage heuristic. An LLM judge runs
  only when `GR_JUDGE_BACKEND` names a working backend.
- Regex detectors miss some PII and some paraphrased injections. The injection and PII suites
  cover only the cases in the repository, and the documentation says so.
- Replacing a detector means changing one module and rerunning the suites.
