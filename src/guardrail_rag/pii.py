"""Regex + Luhn PII detection and redaction."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass

PII_KINDS = ("EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IPV4", "IBAN")


@dataclass(frozen=True)
class PIIMatch:
    kind: str
    start: int
    end: int
    value: str


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SSN_RE = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")
# Candidate card numbers: 13-19 digits with optional single space/dash separators.
_CARD_RE = re.compile(r"(?<![\d])\d(?:[ -]?\d){12,18}(?![\d])")
_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
# Phone: optional country code, then 10+ digits in common US/intl groupings.
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{2,4}(?:[ .-]\(?\d{2,4}\)?)*"
    r"(?:[ .-]\d{2,4})+(?![\w])"
)
_IBAN_RE = re.compile(
    r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b"
)


def luhn_valid(digits: str) -> bool:
    """Return True if digits (a string of 0-9 only) passes the Luhn checksum."""
    if not digits or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _phone_digits_ok(value: str) -> bool:
    return 10 <= sum(c.isdigit() for c in value) <= 15


def _iban_ok(value: str) -> bool:
    compact = value.replace(" ", "")
    if not 15 <= len(compact) <= 34:
        return False
    # Standard mod-97 check.
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1 if numeric.isdigit() else False


def _candidates() -> list[tuple[str, re.Pattern[str]]]:
    # Order defines precedence for overlapping spans at equal start.
    return [
        ("EMAIL", _EMAIL_RE),
        ("SSN", _SSN_RE),
        ("IBAN", _IBAN_RE),
        ("CREDIT_CARD", _CARD_RE),
        ("IPV4", _IPV4_RE),
        ("PHONE", _PHONE_RE),
    ]


def _accept(kind: str, value: str) -> bool:
    if kind == "EMAIL":
        return True
    if kind == "SSN":
        return True
    if kind == "IBAN":
        return _iban_ok(value)
    if kind == "CREDIT_CARD":
        digits = re.sub(r"[ -]", "", value)
        return 13 <= len(digits) <= 19 and luhn_valid(digits)
    if kind == "IPV4":
        try:
            ipaddress.IPv4Address(value)
        except ValueError:
            return False
        return True
    if kind == "PHONE":
        return _phone_digits_ok(value)
    return False


def find_pii(text: str) -> list[PIIMatch]:
    """Return non-overlapping PII matches sorted by start.

    Earlier-starting spans win; for equal starts the longer span wins.
    """
    found: list[PIIMatch] = []
    for kind, pattern in _candidates():
        for m in pattern.finditer(text):
            value = m.group(0)
            if _accept(kind, value):
                found.append(PIIMatch(kind, m.start(), m.end(), value))

    found.sort(key=lambda x: (x.start, -(x.end - x.start)))
    result: list[PIIMatch] = []
    last_end = -1
    for match in found:
        if match.start >= last_end:
            result.append(match)
            last_end = match.end
    return result


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace PII spans with [REDACTED_<KIND>] and return counts per kind."""
    matches = find_pii(text)
    if not matches:
        return text, {}
    counts: dict[str, int] = {}
    pieces: list[str] = []
    pos = 0
    for match in matches:
        pieces.append(text[pos : match.start])
        pieces.append(f"[REDACTED_{match.kind}]")
        counts[match.kind] = counts.get(match.kind, 0) + 1
        pos = match.end
    pieces.append(text[pos:])
    return "".join(pieces), counts
