"""Prompt-injection suite and stored-payload hygiene checks."""

from __future__ import annotations

import json
import re

import pytest

from guardrail_rag.config import Settings
from guardrail_rag.eval.datasets import InjectionCase, load_corpus, load_injection_cases
from guardrail_rag.eval.harness import System, build_system, ingest_fixture_corpus
from guardrail_rag.generation import FakeLLM
from guardrail_rag.generation.prompt import SYSTEM_MARKER

INJECTION_CASES = load_injection_cases()
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RAW_CARD_FORMS = ("4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111")


@pytest.fixture(scope="module")
def system() -> System:
    settings = Settings(
        _env_file=None,
        qdrant_url=None,
        qdrant_path=":memory:",
        embed_backend="hash",
        llm_backend="fake",
    )
    built = build_system(settings=settings, llm=FakeLLM())
    ingest_fixture_corpus(built)
    return built


@pytest.mark.parametrize("case", INJECTION_CASES, ids=[c.id for c in INJECTION_CASES])
def test_injection_case(system: System, case: InjectionCase) -> None:
    answer = system.rag.answer(case.tenant_id, case.query)
    for needle in case.must_not_contain:
        assert needle not in answer.text, f"{needle!r} leaked into the answer"
    assert SYSTEM_MARKER not in answer.text
    if case.expect_flag_prefix:
        assert any(flag.startswith(case.expect_flag_prefix) for flag in answer.flags), (
            f"no flag starting with {case.expect_flag_prefix!r} in {answer.flags}"
        )


def test_stored_payloads_hold_no_raw_card_or_corpus_email(system: System) -> None:
    corpus_emails = {m.group(0) for doc in load_corpus() for m in EMAIL_RE.finditer(doc.text)}
    assert corpus_emails, "the corpus is expected to contain fake email addresses"

    forbidden = [*RAW_CARD_FORMS, *corpus_emails]
    store = system.store
    offset = None
    scanned = 0
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            scanned += 1
            payload_text = json.dumps(point.payload, ensure_ascii=False)
            for needle in forbidden:
                assert needle not in payload_text, f"stored payload contains {needle!r}"
        if offset is None:
            break
    assert scanned > 0
