"""Tenant-scoped Qdrant store. Every query is built with a tenant filter inside this class."""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient, models

from guardrail_rag.config import Settings
from guardrail_rag.models import (
    Chunk,
    ConfigError,
    ScoredChunk,
    TenantIsolationError,
    validate_tenant_id,
)

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"


class TenantVectorStore:
    """Named dense and sparse vectors in one collection, with a tenant payload index.

    There is intentionally no public search method that runs without a tenant_id.
    """

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        if client is None:
            if settings.qdrant_url:
                client = QdrantClient(url=settings.qdrant_url)
            elif settings.qdrant_path == ":memory:":
                client = QdrantClient(location=":memory:")
            else:
                client = QdrantClient(path=settings.qdrant_path)
        self.client = client
        self.collection = settings.collection
        # True when the sparse vector uses Modifier.IDF. Set to False if local mode rejects IDF
        # and the store falls back to raw TF values (see ensure_collection).
        self.sparse_idf = True

    # ----- collection management -------------------------------------------------------------

    def ensure_collection(self, dim: int) -> None:
        """Create the collection if missing. Safe to call repeatedly."""
        if not self.client.collection_exists(self.collection):
            self._create_collection(dim)
        else:
            self._check_dim(dim)
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="tenant_id",
                field_schema=models.KeywordIndexParams(type="keyword", is_tenant=True),
            )
        except Exception as exc:  # noqa: BLE001 - local mode and server mode raise varied errors
            log.warning("tenant_id payload index not created: %s", exc)

    def _create_collection(self, dim: int) -> None:
        vectors = {DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)}
        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors,
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            self.sparse_idf = True
        except Exception as exc:  # noqa: BLE001 - local mode rejects IDF with varied errors
            # Fallback for local mode if Modifier.IDF is rejected: raw TF weights only.
            # Retrieval still works; BM25-style IDF weighting is then not applied.
            log.warning("sparse Modifier.IDF rejected (%s); falling back to raw TF", exc)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors,
                sparse_vectors_config={SPARSE: models.SparseVectorParams()},
            )
            self.sparse_idf = False

    def _check_dim(self, dim: int) -> None:
        info = self.client.get_collection(self.collection)
        existing = info.config.params.vectors
        if isinstance(existing, dict) and DENSE in existing and existing[DENSE].size != dim:
            raise ConfigError(
                f"collection {self.collection!r} has dense size {existing[DENSE].size}, "
                f"but the embedder produces {dim}"
            )

    # ----- writes ----------------------------------------------------------------------------

    def upsert(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[tuple[list[int], list[float]]],
    ) -> int:
        if not (len(chunks) == len(dense) == len(sparse)):
            raise ValueError("chunks, dense and sparse must have the same length")
        points: list[models.PointStruct] = []
        for chunk, vec, (indices, values) in zip(chunks, dense, sparse, strict=True):
            validate_tenant_id(chunk.tenant_id)
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    vector={
                        DENSE: list(vec),
                        SPARSE: models.SparseVector(indices=list(indices), values=list(values)),
                    },
                    payload={
                        "tenant_id": chunk.tenant_id,
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "metadata": dict(chunk.metadata),
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def delete_tenant(self, tenant_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._tenant_filter(tenant_id)),
            wait=True,
        )

    # ----- reads -----------------------------------------------------------------------------

    def search_dense(self, tenant_id: str, vector: list[float], limit: int) -> list[ScoredChunk]:
        flt = self._tenant_filter(tenant_id)
        if not self.client.collection_exists(self.collection):
            return []
        res = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            using=DENSE,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return [self._to_scored(p, tenant_id) for p in res.points]

    def search_sparse(
        self,
        tenant_id: str,
        indices: list[int],
        values: list[float],
        limit: int,
    ) -> list[ScoredChunk]:
        flt = self._tenant_filter(tenant_id)
        if not indices:
            return []
        if not self.client.collection_exists(self.collection):
            return []
        res = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=list(indices), values=list(values)),
            using=SPARSE,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return [self._to_scored(p, tenant_id) for p in res.points]

    def count(self, tenant_id: str) -> int:
        flt = self._tenant_filter(tenant_id)
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(
            collection_name=self.collection, count_filter=flt, exact=True
        ).count

    # ----- internals -------------------------------------------------------------------------

    def _tenant_filter(self, tenant_id: str) -> models.Filter:
        validate_tenant_id(tenant_id)
        return models.Filter(
            must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
            ]
        )

    @staticmethod
    def _to_scored(point: models.ScoredPoint, tenant_id: str) -> ScoredChunk:
        payload = point.payload or {}
        if payload.get("tenant_id") != tenant_id:
            # Defence in depth: the filter should make this unreachable.
            raise TenantIsolationError(
                f"result tenant {payload.get('tenant_id')!r} does not match {tenant_id!r}"
            )
        chunk = Chunk(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            tenant_id=payload["tenant_id"],
            text=payload["text"],
            metadata=dict(payload.get("metadata", {})),
        )
        return ScoredChunk(chunk=chunk, score=float(point.score))
