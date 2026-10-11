"""Dense embedders: a deterministic offline hash embedder and an Ollama client."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

from guardrail_rag.config import Settings
from guardrail_rag.models import ConfigError


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _features(text: str) -> list[str]:
    """Word tokens plus character trigrams of each token (boundary-padded with '#')."""
    feats: list[str] = []
    for token in re.findall(r"\w+", text.lower()):
        feats.append("w:" + token)
        padded = f"#{token}#"
        for i in range(len(padded) - 2):
            feats.append("g:" + padded[i : i + 3])
    return feats


class HashEmbedder:
    """Feature-hashing embedder. Deterministic and network-free; lexical, not semantic."""

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feature in _features(text):
            h = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
            bucket = h % self.dim
            # Sign comes from the top bit of the same hash, independent of the bucket index.
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec  # empty text (or fully cancelled features) -> zero vector
        return [x / norm for x in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OllamaEmbedder:
    """Calls Ollama's /api/embed endpoint. Requires a running Ollama server."""

    def __init__(self, base_url: str, model: str, dim: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/api/embed"
        try:
            resp = httpx.post(
                url,
                json={"model": self.model, "input": texts},
                timeout=60.0,
            )
        except httpx.ConnectError as exc:
            raise ConfigError(
                f"cannot reach Ollama at {self.base_url} (is the server running?): {exc}"
            ) from exc
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        for vec in embeddings:
            if len(vec) != self.dim:
                raise ConfigError(
                    f"Ollama model {self.model!r} returned dim {len(vec)}, expected {self.dim}"
                )
        return embeddings


def get_embedder(settings: Settings) -> Embedder:
    if settings.embed_backend == "hash":
        return HashEmbedder(dim=settings.embed_dim)
    if settings.embed_backend == "ollama":
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.embed_model,
            dim=settings.embed_dim,
        )
    raise ConfigError(f"unknown embed backend: {settings.embed_backend!r}")
