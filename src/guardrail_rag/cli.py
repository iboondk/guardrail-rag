"""Command line: `ingest --tenant T --path DIR` and `ask --tenant T QUESTION`.

Run with `python -m guardrail_rag.cli` or the `guardrail-rag` console script.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from guardrail_rag.config import Settings, get_settings
from guardrail_rag.generation import GuardedRAG, get_llm
from guardrail_rag.ingest import IngestionPipeline, TenantVectorStore, get_embedder, load_directory
from guardrail_rag.models import GuardrailError
from guardrail_rag.retrieval import HybridRetriever, get_reranker


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    store = TenantVectorStore(settings)
    embedder = get_embedder(settings)
    pipeline = IngestionPipeline(settings, store, embedder)
    docs = load_directory(args.path, args.tenant)
    stats = pipeline.ingest(docs)
    print(json.dumps(dataclasses.asdict(stats), indent=2))
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    store = TenantVectorStore(settings)
    embedder = get_embedder(settings)
    retriever = HybridRetriever(
        store,
        embedder,
        reranker=get_reranker(settings),
        candidate_k=settings.candidate_k,
        top_k=settings.top_k,
    )
    rag = GuardedRAG(retriever, get_llm(settings), top_k=settings.top_k)
    result = rag.answer(args.tenant, args.question)
    payload = {
        "answer": result.text,
        "sources": [
            {"doc_id": s.chunk.doc_id, "chunk_id": s.chunk.chunk_id, "score": s.score}
            for s in result.sources
        ],
        "flags": result.flags,
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="guardrail-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest *.md and *.txt files for one tenant")
    ingest.add_argument("--tenant", required=True)
    ingest.add_argument("--path", required=True, help="directory containing the documents")
    ingest.set_defaults(handler=_cmd_ingest)

    ask = sub.add_parser("ask", help="ask a question over one tenant's documents")
    ask.add_argument("--tenant", required=True)
    ask.add_argument("question")
    ask.set_defaults(handler=_cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args, get_settings())
    except GuardrailError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
