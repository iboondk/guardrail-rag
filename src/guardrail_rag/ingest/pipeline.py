"""Ingestion: PII redaction, chunking, embedding and tenant-scoped upsert."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter

from guardrail_rag.config import Settings
from guardrail_rag.ingest.embeddings import Embedder
from guardrail_rag.ingest.sparse import sparse_encode
from guardrail_rag.ingest.store import TenantVectorStore
from guardrail_rag.models import Chunk, ConfigError, Document, validate_tenant_id
from guardrail_rag.pii import redact


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    pii_redactions: dict[str, int] = field(default_factory=dict)


def _merge_counts(into: dict[str, int], counts: dict[str, int]) -> None:
    for kind, n in counts.items():
        into[kind] = into.get(kind, 0) + n


class IngestionPipeline:
    def __init__(self, settings: Settings, store: TenantVectorStore, embedder: Embedder) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self._splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ingest(self, docs: list[Document]) -> IngestStats:
        stats = IngestStats()
        for doc in docs:
            validate_tenant_id(doc.tenant_id)
        if not docs:
            return stats
        self.store.ensure_collection(self.embedder.dim)

        for doc in docs:
            chunks = self._chunk(doc, stats)
            stats.documents += 1
            if not chunks:
                continue
            texts = [c.text for c in chunks]
            dense = self.embedder.embed(texts)
            if len(dense) != len(texts):
                raise ConfigError("embedder returned a different number of vectors than inputs")
            sparse = [sparse_encode(t) for t in texts]
            self.store.upsert(chunks, dense, sparse)
            stats.chunks += len(chunks)
        return stats

    def _chunk(self, doc: Document, stats: IngestStats) -> list[Chunk]:
        redacted, counts = redact(doc.text)
        _merge_counts(stats.pii_redactions, counts)

        metadata: dict = {}
        for key, value in doc.metadata.items():
            if isinstance(value, str):
                value, meta_counts = redact(value)
                _merge_counts(stats.pii_redactions, meta_counts)
            metadata[key] = value
        metadata["source"] = metadata.get("source", "")

        pieces = [p for p in self._splitter.split_text(redacted) if p.strip()]
        return [
            Chunk(
                chunk_id=f"{doc.tenant_id}:{doc.doc_id}:{i}",
                doc_id=doc.doc_id,
                tenant_id=doc.tenant_id,
                text=piece,
                metadata=dict(metadata),
            )
            for i, piece in enumerate(pieces)
        ]


def load_directory(path: str | Path, tenant_id: str) -> list[Document]:
    """Load *.md and *.txt files (sorted by name). doc_id is the file stem."""
    validate_tenant_id(tenant_id)
    root = Path(path)
    files = sorted(
        f for f in root.iterdir() if f.is_file() and f.suffix.lower() in {".md", ".txt"}
    )
    docs: list[Document] = []
    seen: set[str] = set()
    for f in files:
        if f.stem in seen:
            raise ConfigError(f"duplicate doc_id {f.stem!r} in {root}")
        seen.add(f.stem)
        docs.append(
            Document(
                doc_id=f.stem,
                tenant_id=tenant_id,
                text=f.read_text(encoding="utf-8"),
                metadata={"source": f.name},
            )
        )
    return docs
