"""Shared domain models, errors and tenant-id validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


class GuardrailError(Exception):
    """Base class for all Guardrail-RAG errors."""


class InvalidTenantError(GuardrailError, ValueError):
    """Raised when a tenant id does not match TENANT_RE."""


class TenantIsolationError(GuardrailError):
    """Raised when a result crosses a tenant boundary."""


class ConfigError(GuardrailError):
    """Raised for invalid or unusable configuration."""


def validate_tenant_id(tenant_id: str) -> str:
    """Return tenant_id if valid, otherwise raise InvalidTenantError."""
    if not isinstance(tenant_id, str) or not TENANT_RE.fullmatch(tenant_id):
        raise InvalidTenantError(f"invalid tenant id: {tenant_id!r}")
    return tenant_id


@dataclass(frozen=True)
class Document:
    doc_id: str
    tenant_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    tenant_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    sources: list[ScoredChunk]
    flags: list[str] = field(default_factory=list)
    pii_redactions: dict[str, int] = field(default_factory=dict)
