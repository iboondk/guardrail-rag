"""LLM backends. Only FakeLLM is network-free; the others import their packages lazily.

Constructing a backend never makes a network call. Tests use FakeLLM or stubs.
"""

from __future__ import annotations

import re
from typing import Protocol

from guardrail_rag.config import Settings
from guardrail_rag.generation.prompt import REFUSAL
from guardrail_rag.ingest.sparse import tokenize
from guardrail_rag.models import ConfigError

_DOC_RE = re.compile(r'<doc id="([^"]*)">(.*?)</doc>', re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class LLMBackend(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


def _parse_question(user: str) -> str:
    idx = user.rfind("Question:")
    return user[idx + len("Question:") :].strip() if idx >= 0 else ""


def _doc_sort_key(doc_id: str) -> tuple[int, int | str]:
    return (0, int(doc_id)) if doc_id.isdigit() else (1, doc_id)


class FakeLLM:
    """Deterministic extractive answerer for tests and offline runs.

    It picks the sentence from the <doc> blocks with the largest token overlap with the
    question. Ties go to the lowest doc id, then the earliest sentence. It returns REFUSAL when
    there are no docs or when no token overlaps. It never reads the system prompt.
    """

    name = "fake"

    def complete(self, system: str, user: str) -> str:
        question_tokens = set(tokenize(_parse_question(user)))
        docs = sorted(_DOC_RE.findall(user), key=lambda d: _doc_sort_key(d[0]))
        best_score = 0
        best_text: str | None = None
        for doc_id, body in docs:
            for sentence in _split_sentences(body):
                if "[REMOVED:" in sentence:
                    continue
                overlap = len(question_tokens & set(tokenize(sentence)))
                if overlap > best_score:
                    best_score = overlap
                    best_text = f"{sentence} [{doc_id}]"
        return best_text if best_text is not None else REFUSAL


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _chat(llm, system: str, user: str) -> str:
    from llama_index.core.llms import ChatMessage

    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
    return llm.chat(messages).message.content


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str, base_url: str) -> None:
        try:
            from llama_index.llms.ollama import Ollama
        except ImportError as exc:
            raise ConfigError("install extra 'ollama' (pip install -e '.[ollama]')") from exc
        self.model = model
        self._llm = Ollama(model=model, base_url=base_url, request_timeout=120.0)

    def complete(self, system: str, user: str) -> str:
        return _chat(self._llm, system, user)


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str, api_key: str) -> None:
        try:
            from llama_index.llms.anthropic import Anthropic
        except ImportError as exc:
            raise ConfigError("install extra 'anthropic' (pip install -e '.[anthropic]')") from exc
        self.model = model
        self._llm = Anthropic(model=model, api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        return _chat(self._llm, system, user)


class GroqLLM:
    name = "groq"

    def __init__(self, model: str, api_key: str) -> None:
        try:
            from llama_index.llms.groq import Groq
        except ImportError as exc:
            raise ConfigError("install extra 'groq' (pip install -e '.[groq]')") from exc
        self.model = model
        self._llm = Groq(model=model, api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        return _chat(self._llm, system, user)


def get_llm(settings: Settings, backend: str | None = None) -> LLMBackend:
    name = backend or settings.llm_backend
    if name == "fake":
        return FakeLLM()
    if name == "ollama":
        if not settings.ollama_model:
            raise ConfigError("GR_OLLAMA_MODEL is not set")
        return OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_base_url)
    if name == "anthropic":
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
        if not key:
            raise ConfigError("ANTHROPIC_API_KEY is not set (required for llm backend 'anthropic')")
        if not settings.anthropic_model:
            raise ConfigError(
                "GR_ANTHROPIC_MODEL is not set (required for llm backend 'anthropic')"
            )
        return AnthropicLLM(model=settings.anthropic_model, api_key=key)
    if name == "groq":
        key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
        if not key:
            raise ConfigError("GROQ_API_KEY is not set (required for llm backend 'groq')")
        if not settings.groq_model:
            raise ConfigError("GR_GROQ_MODEL is not set (required for llm backend 'groq')")
        return GroqLLM(model=settings.groq_model, api_key=key)
    raise ConfigError(f"unknown llm backend: {name!r}")
