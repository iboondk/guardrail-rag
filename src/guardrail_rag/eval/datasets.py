"""Loaders for the synthetic evaluation fixtures under eval/data.

All data is synthetic: fictional companies, example.com addresses, 555 phone numbers and the
standard test card 4111111111111111.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from guardrail_rag.ingest.pipeline import load_directory
from guardrail_rag.models import Document

DATA_DIR: Path = Path(__file__).parent / "data"
CORPUS_DIR: Path = DATA_DIR / "corpus"
TENANTS = ("acme", "globex", "initech")


@dataclass(frozen=True)
class GoldenItem:
    id: str
    tenant_id: str
    question: str
    relevant_doc_ids: list[str]
    reference_answer: str


@dataclass(frozen=True)
class LeakCase:
    id: str
    attacker_tenant: str
    target_tenant: str
    query: str
    canary: str


@dataclass(frozen=True)
class InjectionCase:
    id: str
    tenant_id: str
    query: str
    must_not_contain: list[str]
    expect_flag_prefix: str | None


def _read_jsonl(name: str) -> list[dict]:
    rows: list[dict] = []
    text = (DATA_DIR / name).read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_corpus() -> list[Document]:
    """Load every tenant's markdown corpus. doc_id is the file stem."""
    docs: list[Document] = []
    for tenant in TENANTS:
        docs.extend(load_directory(CORPUS_DIR / tenant, tenant))
    return docs


def load_golden() -> list[GoldenItem]:
    return [GoldenItem(**row) for row in _read_jsonl("golden.jsonl")]


def load_leak_cases() -> list[LeakCase]:
    return [LeakCase(**row) for row in _read_jsonl("leak_cases.jsonl")]


def load_injection_cases() -> list[InjectionCase]:
    return [InjectionCase(**row) for row in _read_jsonl("injection_cases.jsonl")]
