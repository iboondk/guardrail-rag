import sys

import pytest

from guardrail_rag.config import Settings
from guardrail_rag.ingest import HashEmbedder, IngestionPipeline, TenantVectorStore
from guardrail_rag.models import (
    Chunk,
    ConfigError,
    Document,
    InvalidTenantError,
    ScoredChunk,
    TenantIsolationError,
)
from guardrail_rag.retrieval import (
    HybridRetriever,
    LexicalReranker,
    NoopReranker,
    as_llamaindex_retriever,
    get_reranker,
    rrf,
)

DIM = 128
SHARED = "Our offices are closed on public holidays."
UNIQUE_ACME = "The zephyrium alloy powers the acme reactor."

ACME_DOCS = [
    Document("holidays", "acme", SHARED),
    Document("reactor", "acme", UNIQUE_ACME),
    Document("pricing", "acme", "Acme enterprise pricing is billed annually in advance."),
]
GLOBEX_DOCS = [
    Document("holidays", "globex", SHARED),
    Document("widgets", "globex", "Globex ships widgets worldwide every single week."),
    Document("pricing", "globex", "Globex pricing is billed monthly with no contract."),
]


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, qdrant_path=":memory:", embed_dim=DIM, **overrides)


@pytest.fixture
def store() -> TenantVectorStore:
    s = TenantVectorStore(_settings())
    pipeline = IngestionPipeline(_settings(), s, HashEmbedder(dim=DIM))
    pipeline.ingest(ACME_DOCS + GLOBEX_DOCS)
    return s


def _sc(chunk_id: str, score: float, tenant: str = "acme", text: str = "t") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id=chunk_id, tenant_id=tenant, text=text),
        score=score,
    )


def _retriever(store, mode, reranker=None, top_k=5) -> HybridRetriever:
    return HybridRetriever(
        store,
        HashEmbedder(dim=DIM),
        reranker=reranker,
        candidate_k=10,
        top_k=top_k,
        mode=mode,
    )


# ----- fusion ----------------------------------------------------------------------------------


def test_rrf_ranks_shared_item_first_and_keeps_three_unique():
    a, b, c = _sc("a", 1.0), _sc("b", 0.9), _sc("c", 0.8)
    fused = rrf([[a, b], [b, c]])
    assert next(x.chunk.chunk_id for x in fused) == "b"
    assert {x.chunk.chunk_id for x in fused} == {"a", "b", "c"}
    assert len(fused) == 3
    expected_b = 1 / 62 + 1 / 61
    assert fused[0].score == pytest.approx(expected_b)


def test_rrf_tie_breaks_by_chunk_id():
    fused = rrf([[_sc("z", 1.0)], [_sc("m", 1.0)]])
    assert [x.chunk.chunk_id for x in fused] == ["m", "z"]


# ----- rerankers -------------------------------------------------------------------------------


def test_noop_reranker_truncates():
    chunks = [_sc(f"c{i}", 1.0 - i / 10) for i in range(5)]
    out = NoopReranker().rerank("q", chunks, 2)
    assert [x.chunk.chunk_id for x in out] == ["c0", "c1"]


def test_lexical_reranker_moves_full_match_to_top():
    query = "zephyrium reactor cooling"
    partial = _sc("partial", 0.9, text="The reactor is cooled by water.")
    full = _sc("full", 0.1, text="zephyrium reactor cooling loop, installed in 2024.")
    out = LexicalReranker().rerank(query, [partial, full], 2)
    assert out[0].chunk.chunk_id == "full"


def test_get_reranker_cross_encoder_without_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ConfigError, match="install extra 'rerank'"):
        get_reranker(_settings(rerank_backend="cross_encoder"))


def test_get_reranker_backends():
    assert isinstance(get_reranker(_settings(rerank_backend="none")), NoopReranker)
    assert isinstance(get_reranker(_settings(rerank_backend="lexical")), LexicalReranker)


# ----- retrieval -------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["sparse", "hybrid"])
def test_keyword_query_ranks_acme_doc_first(store, mode):
    results = _retriever(store, mode, LexicalReranker()).retrieve("acme", "zephyrium")
    assert results
    assert results[0].chunk.text == UNIQUE_ACME
    assert results[0].chunk.tenant_id == "acme"


def test_other_tenant_cannot_see_unique_keyword(store):
    for mode in ["dense", "sparse", "hybrid"]:
        results = _retriever(store, mode).retrieve("globex", "zephyrium")
        assert all(r.chunk.tenant_id == "globex" for r in results)
        assert all("zephyrium" not in r.chunk.text.lower() for r in results)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
@pytest.mark.parametrize(
    "tenant,query",
    [("acme", "zephyrium"), ("acme", SHARED), ("globex", SHARED)],
)
def test_every_result_belongs_to_requested_tenant(store, mode, tenant, query):
    r = _retriever(store, mode, LexicalReranker())
    results = r.retrieve(tenant, query)
    assert results
    assert all(item.chunk.tenant_id == tenant for item in results)


def test_retrieve_rejects_invalid_tenant(store):
    r = _retriever(store, "hybrid")
    with pytest.raises(InvalidTenantError):
        r.retrieve("ACME", "zephyrium")
    with pytest.raises(InvalidTenantError):
        r.retrieve("", "zephyrium")


def test_assert_isolation_raises_on_foreign_chunk():
    with pytest.raises(TenantIsolationError):
        HybridRetriever._assert_isolation("acme", [_sc("x", 1.0, tenant="globex")])
    HybridRetriever._assert_isolation("acme", [_sc("y", 1.0, tenant="acme")])


def test_llamaindex_adapter_returns_tenant_nodes(store):
    retriever = _retriever(store, "hybrid", LexicalReranker())
    adapter = as_llamaindex_retriever(retriever, "acme")
    nodes = adapter.retrieve("zephyrium")
    assert nodes
    assert all(n.node.metadata["tenant_id"] == "acme" for n in nodes)
    assert nodes[0].node.text == UNIQUE_ACME
    assert nodes[0].node.id_ == "acme:reactor:0"
