import itertools

import pytest

from guardrail_rag.pii import find_pii, luhn_valid, redact


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("")
    assert not luhn_valid("12a4")


def test_redact_mixed_sample():
    text = (
        "mail bob@example.com or 555-123-4567, card 4111 1111 1111 1111, "
        "ssn 123-45-6789, ip 10.0.0.1"
    )
    out, counts = redact(text)
    for raw in ("bob@example.com", "555-123-4567", "4111 1111 1111 1111", "123-45-6789", "10.0.0.1"):
        assert raw not in out
    assert counts == {"EMAIL": 1, "PHONE": 1, "CREDIT_CARD": 1, "SSN": 1, "IPV4": 1}
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_CREDIT_CARD]" in out


def test_luhn_invalid_card_not_redacted():
    kinds = {m.kind for m in find_pii("num 4111 1111 1111 1112 end")}
    assert "CREDIT_CARD" not in kinds


def test_invalid_ipv4_not_matched():
    assert all(m.kind != "IPV4" for m in find_pii("version 999.1.1.1 here"))


def test_no_pii_unchanged():
    text = "The quick brown fox jumps over 2 lazy dogs."
    assert redact(text) == (text, {})


@pytest.mark.parametrize(
    "phone",
    ["+1 (555) 123-4567", "555-123-4567", "+44 20 7946 0958"],
)
def test_phone_formats(phone):
    matches = find_pii(f"call {phone} today")
    assert [m.kind for m in matches] == ["PHONE"]
    assert matches[0].value == phone


def test_ssn_not_phone():
    kinds = [m.kind for m in find_pii("ssn 123-45-6789")]
    assert kinds == ["SSN"]


def test_iban_detected():
    matches = find_pii("pay to GB82 WEST 1234 5698 7654 32 please")
    assert [m.kind for m in matches] == ["IBAN"]


def test_spans_sorted_and_non_overlapping():
    matches = find_pii("a@b.com 10.0.0.1 4111111111111111")
    starts = [m.start for m in matches]
    assert starts == sorted(starts)
    for prev, cur in itertools.pairwise(matches):
        assert prev.end <= cur.start
