import shutil
import tempfile
from pathlib import Path

import pytest

from guardrail_rag.config import Settings
from guardrail_rag.ingest import (
    DENSE,
    HashEmbedder,
    IngestionPipeline,
    TenantVectorStore,
    sparse_encode,
)
from guardrail_rag.models import Chunk, ConfigError, Document, InvalidTenantError

DIM = 128
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _store() -> TenantVectorStore:
    return TenantVectorStore(Settings(_env_file=None, qdrant_path=":memory:", embed_dim=DIM))


def _seeded_store() -> TenantVectorStore:
    store = _store()
    settings = Settings(_env_file=None, qdrant_path=":memory:", embed_dim=DIM)
    pipeline = IngestionPipeline(settings, store, HashEmbedder(dim=DIM))
    pipeline.ingest(
        [
            Document("pricing", "acme", "Acme enterprise pricing is billed annually in advance."),
            Document("pricing", "globex", "Globex pricing is billed monthly with no contract."),
            Document("holidays", "acme", "Acme offices close for winter holidays in December."),
        ]
    )
    return store


def test_ensure_collection_is_idempotent():
    store = _store()
    store.ensure_collection(DIM)
    store.ensure_collection(DIM)
    assert store.client.collection_exists(store.collection)
    assert store.count("acme") == 0
    info = store.client.get_collection(store.collection)
    assert DENSE in info.config.params.vectors


def test_ensure_collection_dim_mismatch_raises():
    store = _store()
    store.ensure_collection(DIM)
    with pytest.raises(ConfigError, match="dense size"):
        store.ensure_collection(DIM * 2)


def test_upsert_validates_tenant_and_lengths():
    store = _store()
    store.ensure_collection(DIM)
    bad = Chunk(chunk_id="*:d:0", doc_id="d", tenant_id="*", text="x")
    with pytest.raises(InvalidTenantError):
        store.upsert([bad], [[0.0] * DIM], [([], [])])
    good = Chunk(chunk_id="acme:d:0", doc_id="d", tenant_id="acme", text="x")
    with pytest.raises(ValueError):
        store.upsert([good], [], [([], [])])


def test_upsert_then_count_and_point_id_is_uuid5():
    import uuid

    store = _store()
    store.ensure_collection(DIM)
    chunk = Chunk(chunk_id="acme:d:0", doc_id="d", tenant_id="acme", text="hello world")
    vec = HashEmbedder(dim=DIM).embed(["hello world"])
    n = store.upsert([chunk], vec, [sparse_encode("hello world")])
    assert n == 1
    assert store.count("acme") == 1
    points, _ = store.client.scroll(store.collection, limit=10)
    assert points[0].id == str(uuid.uuid5(uuid.NAMESPACE_URL, "acme:d:0"))


def test_search_dense_returns_only_requested_tenant():
    store = _seeded_store()
    query = HashEmbedder(dim=DIM).embed(["billing pricing enterprise"])[0]
    hits = store.search_dense("acme", query, limit=10)
    assert hits
    assert all(h.chunk.tenant_id == "acme" for h in hits)
    assert not any("Globex" in h.chunk.text for h in hits)


def test_search_sparse_returns_only_requested_tenant():
    store = _seeded_store()
    indices, values = sparse_encode("pricing billed")
    hits = store.search_sparse("globex", indices, values, limit=10)
    assert hits
    assert all(h.chunk.tenant_id == "globex" for h in hits)
    assert all("Acme" not in h.chunk.text for h in hits)


def test_search_sparse_empty_indices_returns_empty():
    store = _seeded_store()
    assert store.search_sparse("acme", [], [], limit=5) == []


@pytest.mark.parametrize("bad", ["*", "", " acme", "acme OR 1=1"])
def test_bad_tenant_raises_on_every_read(bad):
    store = _seeded_store()
    vec = [0.0] * DIM
    with pytest.raises(InvalidTenantError):
        store.search_dense(bad, vec, 5)
    with pytest.raises(InvalidTenantError):
        store.search_sparse(bad, [1], [1.0], 5)
    with pytest.raises(InvalidTenantError):
        store.count(bad)
    with pytest.raises(InvalidTenantError):
        store.delete_tenant(bad)


def test_tenant_filter_requires_valid_id():
    store = _store()
    flt = store._tenant_filter("acme")
    assert flt.must[0].key == "tenant_id"
    assert flt.must[0].match.value == "acme"
    with pytest.raises(InvalidTenantError):
        store._tenant_filter("*")


def test_delete_tenant_removes_only_that_tenant():
    store = _seeded_store()
    assert store.count("acme") == 2
    assert store.count("globex") == 1
    store.delete_tenant("acme")
    assert store.count("acme") == 0
    assert store.count("globex") == 1


def test_sparse_idf_flag_recorded():
    store = _store()
    store.ensure_collection(DIM)
    # Whichever mode local Qdrant accepted, the flag must reflect it.
    assert isinstance(store.sparse_idf, bool)


def test_search_on_missing_collection_is_empty():
    store = _store()
    assert store.count("acme") == 0
    assert store.search_dense("acme", [0.0] * DIM, 3) == []
    assert store.search_sparse("acme", [1], [1.0], 3) == []


def test_store_built_from_settings_path_uses_local_directory():
    scratch = Path(tempfile.mkdtemp(dir=PROJECT_ROOT / ".pytest_cache"))
    try:
        settings = Settings(_env_file=None, qdrant_path=str(scratch / "qdrant"), embed_dim=DIM)
        store = TenantVectorStore(settings)
        store.ensure_collection(DIM)
        assert store.count("acme") == 0
    finally:
        store.client.close()
        shutil.rmtree(scratch, ignore_errors=True)
