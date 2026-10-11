import math
import shutil
import tempfile
from pathlib import Path

import pytest

from guardrail_rag.config import Settings
from guardrail_rag.ingest import (
    HashEmbedder,
    IngestionPipeline,
    OllamaEmbedder,
    TenantVectorStore,
    get_embedder,
    load_directory,
    sparse_encode,
    tokenize,
)
from guardrail_rag.ingest.sparse import STOPWORDS, token_index
from guardrail_rag.models import ConfigError, Document, InvalidTenantError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, qdrant_path=":memory:", **kw)


# ----- embedders -------------------------------------------------------------------------------


def test_hash_embedder_deterministic():
    e1, e2 = HashEmbedder(dim=128), HashEmbedder(dim=128)
    text = "Refund policy: returns accepted within 30 days."
    assert e1.embed([text]) == e2.embed([text])
    assert e1.embed([text]) == e1.embed([text])


def test_hash_embedder_shape_and_unit_norm():
    emb = HashEmbedder(dim=384)
    (vec,) = emb.embed(["quarterly revenue grew in Europe"])
    assert len(vec) == emb.dim == 384
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)


def test_hash_embedder_empty_text_is_zero_vector():
    (vec,) = HashEmbedder(dim=64).embed([""])
    assert len(vec) == 64
    assert all(x == 0.0 for x in vec)


def test_similar_texts_score_higher_than_unrelated():
    emb = HashEmbedder(dim=384)
    anchor, similar, unrelated = emb.embed(
        [
            "The refund policy allows returns within thirty days.",
            "Refunds: returns are allowed within 30 days under the policy.",
            "Penguins live in the Antarctic and eat small fish.",
        ]
    )
    assert _cos(anchor, similar) > _cos(anchor, unrelated)


def test_ollama_unreachable_raises_config_error():
    # Nothing listens on port 9 locally, so the connection is refused without any network.
    emb = OllamaEmbedder(base_url="http://127.0.0.1:9", model="all-minilm:l6-v2", dim=384)
    with pytest.raises(ConfigError, match="cannot reach Ollama"):
        emb.embed(["hello"])


def test_get_embedder_selects_backend():
    assert isinstance(get_embedder(_settings(embed_backend="hash", embed_dim=32)), HashEmbedder)
    emb = get_embedder(_settings(embed_backend="ollama", embed_dim=384))
    assert isinstance(emb, OllamaEmbedder)
    assert emb.dim == 384


# ----- sparse encoder --------------------------------------------------------------------------


def test_tokenize_drops_stopwords_and_single_chars():
    assert tokenize("The cat, a dog: x and 42!") == ["cat", "dog", "42"]
    assert "the" in STOPWORDS and len(STOPWORDS) >= 35


def test_token_index_stable_and_non_negative():
    assert token_index("cat") == token_index("cat")
    assert 0 <= token_index("cat") <= 0x7FFFFFFF


def test_sparse_encode_repeated_term_weighted_higher():
    indices, values = sparse_encode("the cat cat sat")
    assert len(indices) == 2
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    cat_w = values[indices.index(token_index("cat"))]
    sat_w = values[indices.index(token_index("sat"))]
    assert cat_w > sat_w
    assert math.isclose(cat_w, 1 + math.log(2))
    assert math.isclose(sat_w, 1.0)


def test_sparse_encode_empty_text():
    assert sparse_encode("the and of") == ([], [])


# ----- pipeline --------------------------------------------------------------------------------


def _pipeline(settings: Settings | None = None):
    settings = settings or _settings(embed_dim=128)
    store = TenantVectorStore(settings)
    return IngestionPipeline(settings, store, HashEmbedder(dim=settings.embed_dim)), store


def _docs():
    return [
        Document("refunds", "acme", "Refunds are accepted within 30 days of purchase."),
        Document("shipping", "acme", "Standard shipping takes five business days."),
        Document("refunds", "globex", "Globex refunds take ten business days to clear."),
    ]


def test_ingest_counts_per_tenant():
    pipeline, store = _pipeline()
    stats = pipeline.ingest(_docs())
    assert stats.documents == 3
    assert stats.chunks >= 3
    assert store.count("acme") >= 2
    assert store.count("globex") >= 1


def test_same_doc_id_two_tenants_no_overwrite():
    pipeline, store = _pipeline()
    pipeline.ingest(
        [
            Document("shared-id", "acme", "Acme handbook text about onboarding."),
            Document("shared-id", "globex", "Globex handbook text about onboarding."),
        ]
    )
    assert store.count("acme") == 1
    assert store.count("globex") == 1
    points, _ = store.client.scroll("guardrail_rag", limit=100, with_payload=True)
    assert len(points) == 2
    assert {p.id for p in points} and len({p.id for p in points}) == 2


def test_pii_is_not_stored_and_counted():
    pipeline, store = _pipeline()
    doc = Document(
        "contact",
        "acme",
        "Reach jane@example.com for billing. Card on file: 4111111111111111.",
    )
    stats = pipeline.ingest([doc])
    assert stats.pii_redactions.get("EMAIL") == 1
    assert stats.pii_redactions.get("CREDIT_CARD") == 1

    points, _ = store.client.scroll("guardrail_rag", limit=100, with_payload=True)
    assert points
    for p in points:
        blob = repr(p.payload)
        assert "jane@example.com" not in blob
        assert "4111111111111111" not in blob
        assert "[REDACTED_EMAIL]" in p.payload["text"]


def test_ingest_empty_list_is_noop():
    pipeline, store = _pipeline()
    stats = pipeline.ingest([])
    assert stats.documents == 0 and stats.chunks == 0
    assert not store.client.collection_exists("guardrail_rag")


def test_ingest_rejects_invalid_tenant():
    pipeline, _ = _pipeline()
    with pytest.raises(InvalidTenantError):
        pipeline.ingest([Document("x", "*", "text")])


def test_ingest_metadata_has_source_and_chunk_ids():
    pipeline, store = _pipeline()
    pipeline.ingest([Document("d1", "acme", "Some text.", metadata={"source": "d1.md"})])
    points, _ = store.client.scroll("guardrail_rag", limit=10, with_payload=True)
    (p,) = points
    assert p.payload["chunk_id"] == "acme:d1:0"
    assert p.payload["metadata"]["source"] == "d1.md"


def test_load_directory_reads_md_and_txt_sorted():
    scratch = Path(tempfile.mkdtemp(dir=PROJECT_ROOT / ".pytest_cache"))
    try:
        (scratch / "b.txt").write_text("Second file.", encoding="utf-8")
        (scratch / "a.md").write_text("# First\nFirst file.", encoding="utf-8")
        (scratch / "ignored.pdf").write_text("nope", encoding="utf-8")
        docs = load_directory(scratch, "acme")
        assert [d.doc_id for d in docs] == ["a", "b"]
        assert docs[0].metadata == {"source": "a.md"}
        assert all(d.tenant_id == "acme" for d in docs)
        assert "First file." in docs[0].text
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_load_directory_rejects_bad_tenant():
    with pytest.raises(InvalidTenantError):
        load_directory(PROJECT_ROOT, "")
