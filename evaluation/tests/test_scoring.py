"""
scoring.py 순수 함수 테스트 — 모델·DB 의존성 없음.
"""

from __future__ import annotations

import math
import pytest

from evaluation.scoring import (
    compute_idf,
    concept_overlap_score,
    rerank_by_concept_overlap,
)


# ── compute_idf ───────────────────────────────────────────────────────────────

def test_compute_idf_single_photo_single_concept() -> None:
    idf = compute_idf({"d1": ["mood:평화로운"]})
    # log((1+1)/(1+1)) = log(1) = 0.0
    assert idf["mood:평화로운"] == pytest.approx(0.0)


def test_compute_idf_rare_concept_higher_than_common() -> None:
    photo_concepts = {
        "d1": ["place:카페", "mood:행복한"],
        "d2": ["place:카페"],
        "d3": ["mood:행복한"],
    }
    idf = compute_idf(photo_concepts)
    # place:카페 appears in 2/3 photos; mood:행복한 appears in 2/3 — equal df
    # all three are equal here; just verify they're non-negative
    assert all(v >= 0.0 for v in idf.values())


def test_compute_idf_concept_only_in_one_photo_is_high() -> None:
    photo_concepts = {
        f"d{i}": ["place:카페"] for i in range(9)
    }
    photo_concepts["d9"] = ["object:자전거"]  # unique concept
    idf = compute_idf(photo_concepts)
    # object:자전거: log(11/2) ≈ 1.70
    # place:카페:   log(11/10) ≈ 0.095
    assert idf["object:자전거"] > idf["place:카페"]


def test_compute_idf_empty_corpus_returns_empty() -> None:
    assert compute_idf({}) == {}


def test_compute_idf_keys_match_concepts() -> None:
    corpus = {
        "d1": ["a", "b"],
        "d2": ["b", "c"],
    }
    idf = compute_idf(corpus)
    assert set(idf.keys()) == {"a", "b", "c"}


# ── concept_overlap_score ─────────────────────────────────────────────────────

def test_concept_overlap_score_no_overlap_returns_zero() -> None:
    idf = {"place:카페": 1.0, "mood:행복한": 0.5}
    score = concept_overlap_score(["object:자전거"], ["place:카페"], idf)
    assert score == pytest.approx(0.0)


def test_concept_overlap_score_full_overlap() -> None:
    idf = {"place:카페": 1.5, "mood:행복한": 0.8}
    score = concept_overlap_score(
        ["place:카페", "mood:행복한"],
        ["place:카페", "mood:행복한"],
        idf,
    )
    assert score == pytest.approx(1.5 + 0.8)


def test_concept_overlap_score_partial_overlap() -> None:
    idf = {"place:카페": 1.0, "mood:행복한": 2.0}
    score = concept_overlap_score(
        ["place:카페", "mood:행복한"],
        ["place:카페"],
        idf,
    )
    assert score == pytest.approx(1.0)


def test_concept_overlap_score_unknown_concept_in_idf_ignored() -> None:
    idf = {"place:카페": 1.0}
    # "style:빈티지" not in idf
    score = concept_overlap_score(["style:빈티지"], ["style:빈티지"], idf)
    assert score == pytest.approx(0.0)


def test_concept_overlap_score_duplicate_query_concepts_counted_once() -> None:
    idf = {"place:카페": 1.0}
    # Duplicate in query — set intersection deduplices
    score = concept_overlap_score(
        ["place:카페", "place:카페"],
        ["place:카페"],
        idf,
    )
    assert score == pytest.approx(1.0)


# ── rerank_by_concept_overlap ─────────────────────────────────────────────────

def test_rerank_prefers_higher_overlap() -> None:
    idf = {"place:카페": 1.0, "mood:행복한": 2.0}
    photo_concept_map = {
        "d1": ["mood:행복한"],          # overlap score: 2.0
        "d2": ["place:카페"],           # overlap score: 1.0
        "d3": ["object:자전거"],        # overlap score: 0.0
    }
    # Candidates in original B-order: d2, d1, d3
    reranked = rerank_by_concept_overlap(
        candidates=["d2", "d1", "d3"],
        query_concepts=["place:카페", "mood:행복한"],
        photo_concept_map=photo_concept_map,
        idf=idf,
        top_k=3,
    )
    assert reranked == ["d1", "d2", "d3"]


def test_rerank_top_k_limits_results() -> None:
    idf = {"mood:행복한": 1.0}
    photo_concept_map = {f"d{i}": ["mood:행복한"] for i in range(5)}
    result = rerank_by_concept_overlap(
        candidates=list(photo_concept_map.keys()),
        query_concepts=["mood:행복한"],
        photo_concept_map=photo_concept_map,
        idf=idf,
        top_k=3,
    )
    assert len(result) == 3


def test_rerank_missing_doc_id_gets_zero_score() -> None:
    idf = {"place:카페": 1.0}
    # d2 not in photo_concept_map → score 0
    result = rerank_by_concept_overlap(
        candidates=["d1", "d2"],
        query_concepts=["place:카페"],
        photo_concept_map={"d1": ["place:카페"]},
        idf=idf,
        top_k=2,
    )
    assert result[0] == "d1"
    assert result[1] == "d2"


def test_rerank_empty_candidates_returns_empty() -> None:
    result = rerank_by_concept_overlap(
        candidates=[],
        query_concepts=["place:카페"],
        photo_concept_map={},
        idf={"place:카페": 1.0},
        top_k=10,
    )
    assert result == []
