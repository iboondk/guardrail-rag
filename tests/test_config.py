import pytest

from guardrail_rag.config import Settings, get_settings
from guardrail_rag.models import InvalidTenantError, validate_tenant_id


def test_defaults(monkeypatch):
    for name in ("GR_LLM_BACKEND", "GR_QDRANT_PATH", "GR_QDRANT_URL"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_backend == "fake"
    assert s.qdrant_path == ":memory:"
    assert s.qdrant_url is None
    assert s.embed_backend == "hash"
    assert s.rerank_backend == "lexical"


def test_env_override(monkeypatch):
    monkeypatch.setenv("GR_LLM_BACKEND", "ollama")
    s = get_settings()
    assert s.llm_backend == "ollama"


def test_anthropic_key_alias(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-real")
    s = get_settings()
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "test-not-real"


def test_populate_by_name_for_key():
    s = get_settings(anthropic_api_key="x")
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "x"


@pytest.mark.parametrize("ok", ["acme", "globex-2", "a_b", "x1"])
def test_tenant_accepts(ok):
    assert validate_tenant_id(ok) == ok


@pytest.mark.parametrize("bad", ["", "*", "Acme", "a", "acme OR 1=1", None, " acme", "acme;drop", 123])
def test_tenant_rejects(bad):
    with pytest.raises(InvalidTenantError):
        validate_tenant_id(bad)
