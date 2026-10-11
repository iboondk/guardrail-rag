"""Cross-tenant leak suite. Every LeakCase must produce zero leak findings."""

from __future__ import annotations

import pytest

from guardrail_rag.config import Settings
from guardrail_rag.eval.datasets import TENANTS, load_leak_cases
from guardrail_rag.eval.harness import System, build_system, ingest_fixture_corpus, leak_findings
from guardrail_rag.generation import FakeLLM
from guardrail_rag.ingest import HashEmbedder
from guardrail_rag.models import InvalidTenantError

LEAK_CASES = load_leak_cases()

# Canary per tenant, as planted in each tenant's secrets.md.
OWN_CANARIES = {
    "acme": "CANARY-ACME-5b1e9c",
    "globex": "CANARY-GLOBEX-a73d20",
    "initech": "CANARY-INITECH-4f08c6",
}


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
    assert isinstance(built.embedder, HashEmbedder)
    stats = ingest_fixture_corpus(built)
    assert stats.documents > 0
    return built


@pytest.mark.parametrize("case", LEAK_CASES, ids=[c.id for c in LEAK_CASES])
def test_no_cross_tenant_leak(system: System, case) -> None:
    assert leak_findings(system, case) == []


@pytest.mark.parametrize("tenant", TENANTS)
def test_tenant_can_retrieve_its_own_canary(system: System, tenant: str) -> None:
    # Proves the canary is indexed, so the zero-leak result above is meaningful.
    canary = OWN_CANARIES[tenant]
    answer = system.rag.answer(tenant, f"What is the internal canary token {canary}?")
    retrieved = [item.chunk.text for item in answer.sources]
    assert any(canary in text for text in retrieved)
    assert all(item.chunk.tenant_id == tenant for item in answer.sources)


@pytest.mark.parametrize("bad", ["*", "", "acme,globex", "ACME"])
def test_invalid_tenant_ids_are_rejected(system: System, bad: str) -> None:
    with pytest.raises(InvalidTenantError):
        system.rag.answer(bad, "What is the refund window?")
