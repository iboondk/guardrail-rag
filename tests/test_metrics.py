"""Unit tests for retrieval and faithfulness metrics."""

from __future__ import annotations

import pytest

from guardrail_rag.eval.metrics import (
    faithfulness_heuristic,
    llm_judge_faithfulness,
    mrr,
    recall_at_k,
)
from guardrail_rag.generation.prompt import REFUSAL


class StubJudge:
    name = "stub"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


# ----- recall_at_k -----------------------------------------------------------------------------


def test_recall_at_k_counts_relevant_docs_in_top_k():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=2) == 0.5


def test_recall_at_k_empty_relevant_is_zero():
    assert recall_at_k(["a", "b"], [], k=5) == 0.0


def test_recall_at_k_empty_retrieved_is_zero():
    assert recall_at_k([], ["a"], k=5) == 0.0


def test_recall_at_k_deduplicates_before_slicing():
    # Three chunks from "a" must not push "b" out of the top 2 distinct docs.
    retrieved = ["a", "a", "a", "b"]
    assert recall_at_k(retrieved, ["b"], k=2) == 1.0
    assert recall_at_k(retrieved, ["b"], k=1) == 0.0


def test_recall_at_k_duplicate_relevant_ids_use_set_size():
    assert recall_at_k(["a"], ["a", "a"], k=1) == 1.0


def test_recall_at_k_k_larger_than_list():
    assert recall_at_k(["a"], ["a", "b"], k=10) == 0.5


# ----- mrr -------------------------------------------------------------------------------------


def test_mrr_first_relevant_rank():
    assert mrr(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)
    assert mrr(["a", "b"], ["b"]) == pytest.approx(0.5)
    assert mrr(["a", "b"], ["a"]) == 1.0


def test_mrr_no_relevant_retrieved_is_zero():
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_mrr_empty_relevant_is_zero():
    assert mrr(["a"], []) == 0.0


def test_mrr_empty_retrieved_is_zero():
    assert mrr([], ["a"]) == 0.0


def test_mrr_ranks_distinct_docs_not_chunks():
    # Chunks of "x" come first; the rank of "a" is 2 among distinct docs, not 3 among chunks.
    assert mrr(["x", "x", "a"], ["a"]) == pytest.approx(0.5)


# ----- faithfulness_heuristic ------------------------------------------------------------------

CONTEXT = ["Acme offers a full refund for monthly subscriptions within 30 days of the charge."]


def test_faithfulness_fully_supported_answer():
    answer = "Acme offers a full refund for monthly subscriptions within 30 days."
    assert faithfulness_heuristic(answer, CONTEXT) == 1.0


def test_faithfulness_unsupported_sentence_lowers_score():
    answer = (
        "Acme offers a full refund for monthly subscriptions within 30 days. "
        "Dragons guard the vault under the mountain."
    )
    assert faithfulness_heuristic(answer, CONTEXT) == pytest.approx(0.5)


def test_faithfulness_strips_citation_markers():
    answer = "Acme offers a full refund for monthly subscriptions within 30 days [1]."
    assert faithfulness_heuristic(answer, CONTEXT) == 1.0


def test_faithfulness_strips_multi_citation_markers():
    answer = "Acme offers a full refund for monthly subscriptions [1, 2]."
    assert faithfulness_heuristic(answer, CONTEXT) == 1.0


def test_faithfulness_refusal_makes_no_claims():
    assert faithfulness_heuristic(REFUSAL, CONTEXT) == 1.0


def test_faithfulness_empty_answer_is_one():
    assert faithfulness_heuristic("", CONTEXT) == 1.0
    assert faithfulness_heuristic("   ", CONTEXT) == 1.0


def test_faithfulness_answer_with_only_citations_is_one():
    assert faithfulness_heuristic("[1]", CONTEXT) == 1.0


def test_faithfulness_no_contexts_scores_zero_for_claims():
    assert faithfulness_heuristic("Acme offers refunds.", []) == 0.0


def test_faithfulness_threshold_controls_support():
    answer = "Acme offers a refund for banana hovercraft."
    # Tokens: acme, offers, refund, banana, hovercraft -> 3 of 5 supported (0.6).
    assert faithfulness_heuristic(answer, CONTEXT, threshold=0.6) == 1.0
    assert faithfulness_heuristic(answer, CONTEXT, threshold=0.7) == 0.0


def test_faithfulness_is_between_zero_and_one():
    answer = "Random words here. Acme offers refunds. Nothing matches zebra quartz."
    score = faithfulness_heuristic(answer, CONTEXT)
    assert 0.0 <= score <= 1.0


# ----- llm_judge_faithfulness ------------------------------------------------------------------


def test_llm_judge_parses_first_float():
    judge = StubJudge("Score: 0.75")
    assert llm_judge_faithfulness(judge, "answer", CONTEXT) == pytest.approx(0.75)
    assert judge.calls, "judge must be called"


def test_llm_judge_prompt_contains_answer_and_context():
    judge = StubJudge("0.5")
    llm_judge_faithfulness(judge, "THE-ANSWER", ["THE-CONTEXT"])
    system, user = judge.calls[0]
    assert "THE-ANSWER" in user
    assert "THE-CONTEXT" in user
    assert "0" in system and "1" in system


def test_llm_judge_clamps_above_one():
    assert llm_judge_faithfulness(StubJudge("Score: 1.7"), "a", CONTEXT) == 1.0


def test_llm_judge_clamps_below_zero():
    assert llm_judge_faithfulness(StubJudge("-0.3"), "a", CONTEXT) == 0.0


def test_llm_judge_integer_reply():
    assert llm_judge_faithfulness(StubJudge("1"), "a", CONTEXT) == 1.0


def test_llm_judge_raises_without_number():
    with pytest.raises(ValueError):
        llm_judge_faithfulness(StubJudge("no idea"), "a", CONTEXT)


def test_llm_judge_raises_on_empty_reply():
    with pytest.raises(ValueError):
        llm_judge_faithfulness(StubJudge(""), "a", CONTEXT)
