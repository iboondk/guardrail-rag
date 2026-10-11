"""Integrity checks for the synthetic evaluation fixtures."""

from __future__ import annotations

import json
import re
from collections import Counter

from guardrail_rag.eval.datasets import (
    DATA_DIR,
    TENANTS,
    load_corpus,
    load_golden,
    load_injection_cases,
    load_leak_cases,
)

CANARIES = {
    "acme": "CANARY-ACME-5b1e9c",
    "globex": "CANARY-GLOBEX-a73d20",
    "initech": "CANARY-INITECH-4f08c6",
}
SYSTEM_MARKER = "GR-SYS-7f3a"
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _docs_by_tenant() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {t: {} for t in TENANTS}
    for doc in load_corpus():
        out[doc.tenant_id][doc.doc_id] = doc.text
    return out


def test_every_jsonl_file_parses_line_by_line():
    for name in ("golden.jsonl", "leak_cases.jsonl", "injection_cases.jsonl"):
        lines = [ln for ln in (DATA_DIR / name).read_text(encoding="utf-8").splitlines() if ln]
        assert lines, name
        for line in lines:
            json.loads(line)


def test_corpus_layout_per_tenant():
    docs = _docs_by_tenant()
    for tenant in TENANTS:
        assert "secrets" in docs[tenant], f"{tenant} has no secrets doc"
        assert CANARIES[tenant] in docs[tenant]["secrets"]
        # Canary appears only in its own tenant's corpus.
        for other in TENANTS:
            if other == tenant:
                continue
            for text in docs[other].values():
                assert CANARIES[tenant] not in text
        # Six topic docs plus secrets plus at least one injection doc.
        assert len(docs[tenant]) >= 8


NON_TOPIC_DOCS = {"secrets", "support-runbook", "partner-faq", "field-service-notes"}


def test_topic_docs_are_150_to_400_words():
    for tenant, docs in _docs_by_tenant().items():
        for doc_id, text in docs.items():
            if doc_id in NON_TOPIC_DOCS:
                continue
            words = len(text.split())
            assert 150 <= words <= 400, f"{tenant}/{doc_id} has {words} words"


def test_injection_payloads_are_embedded_in_corpus():
    markers = ("Ignore all previous instructions", "SYSTEM:", "</doc>", "you are now")
    for tenant in TENANTS:
        texts = [t for t in _docs_by_tenant()[tenant].values()]
        assert any(any(m in t for m in markers) for t in texts), tenant


def test_golden_items_cover_each_tenant_eight_times():
    golden = load_golden()
    assert len(golden) >= 24
    counts = Counter(item.tenant_id for item in golden)
    for tenant in TENANTS:
        assert counts[tenant] == 8


def test_golden_relevant_docs_exist_in_tenant_corpus():
    docs = _docs_by_tenant()
    for item in load_golden():
        assert item.relevant_doc_ids, item.id
        for doc_id in item.relevant_doc_ids:
            assert doc_id in docs[item.tenant_id], f"{item.id}: missing {item.tenant_id}/{doc_id}"


def test_golden_reference_answers_appear_in_referenced_doc():
    docs = _docs_by_tenant()
    for item in load_golden():
        doc_text = " ".join(_norm(docs[item.tenant_id][d]) for d in item.relevant_doc_ids)
        assert _norm(item.reference_answer) in doc_text, item.id


def test_leak_cases_cover_every_ordered_tenant_pair():
    cases = load_leak_cases()
    assert len(cases) >= 30
    pairs = {(c.attacker_tenant, c.target_tenant) for c in cases}
    expected = {(a, t) for a in TENANTS for t in TENANTS if a != t}
    assert pairs == expected
    for case in cases:
        assert case.attacker_tenant != case.target_tenant
        assert case.canary == CANARIES[case.target_tenant]


def test_injection_cases_have_required_fields():
    cases = load_injection_cases()
    assert len(cases) >= 10
    for case in cases:
        assert SYSTEM_MARKER in case.must_not_contain
        others = [CANARIES[t] for t in TENANTS if t != case.tenant_id]
        for canary in others:
            assert canary in case.must_not_contain
        assert case.expect_flag_prefix in (None, "context_injection", "query_injection")
