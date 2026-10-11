"""Prompt-injection detection (regex) and line-level context sanitizing.

These patterns are a heuristic first line of defence. They are not a complete classifier:
the system prompt also tells the model that retrieved text is untrusted data.
"""

from __future__ import annotations

import re

REMOVED_MARKER = "[REMOVED: possible prompt injection]"

_FLAGS = re.IGNORECASE | re.MULTILINE

INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:all|any|the|previous|prior|above)\s+.*instructions",
        _FLAGS,
    ),
    "system_prompt": re.compile(
        r"\b(?:reveal|print|show|repeat)\b.{0,40}(?:system prompt|instructions|hidden)",
        _FLAGS,
    ),
    "role_override": re.compile(
        r"you are now|\bact as\b|new instructions:",
        _FLAGS,
    ),
    "role_tags": re.compile(
        r"^\s*(?:system|assistant)\s*:|<\|?(?:system|im_start)\|?>",
        _FLAGS,
    ),
    "delimiter_escape": re.compile(r"</?doc\b", _FLAGS),
    "exfiltration": re.compile(
        r"\b(?:send|post|upload)\b.{0,40}\bhttps?://",
        _FLAGS,
    ),
    "other_tenant": re.compile(
        r"\b(?:other|another|all|every)\s+(?:tenant|customer|client)s?'?s?\s+(?:data|documents|records)",
        _FLAGS,
    ),
}

_DOC_TAG_RE = re.compile(r"<(/?)doc", re.IGNORECASE)


def detect_injection(text: str) -> list[str]:
    """Return sorted unique names of injection patterns that match text."""
    return sorted(name for name, pattern in INJECTION_PATTERNS.items() if pattern.search(text))


def sanitize_context(text: str) -> tuple[str, list[str]]:
    """Remove injected lines, escape leftover doc tags, and return (clean_text, names).

    Lines matching any injection pattern are replaced with REMOVED_MARKER. Any remaining
    '<doc' or '</doc' (for example '<docx') is escaped to '&lt;doc' or '&lt;/doc'.
    """
    names: set[str] = set()
    kept: list[str] = []
    for line in text.splitlines():
        hits = detect_injection(line)
        if hits:
            names.update(hits)
            kept.append(REMOVED_MARKER)
        else:
            kept.append(line)
    clean = "\n".join(kept)
    clean = _DOC_TAG_RE.sub(r"&lt;\1doc", clean)
    return clean, sorted(names)


def screen_query(query: str) -> list[str]:
    """Check a user query for injection patterns. Returns sorted pattern names."""
    return detect_injection(query)
