"""
Pure scoring utilities for concept-overlap re-ranking (B+C).

All functions are stateless and side-effect-free to remain unit-testable
without any model or database dependency.
"""

from __future__ import annotations

import math
from collections import Counter


# ── IDF ───────────────────────────────────────────────────────────────────────

def compute_idf(
    photo_concepts: dict[str, list[str]],
) -> dict[str, float]:
    """
    Computes smoothed IDF weights for each concept across the corpus.

    IDF(c) = log((N + 1) / (df(c) + 1))
    where N is the number of photos and df(c) is the number of photos
    containing concept c.

    Args:
        photo_concepts: mapping of {doc_id -> [concept_id, ...]}.

    Returns:
        Mapping of {concept_id -> idf_weight}; concepts absent from the
        corpus are not included.
    """
    n = len(photo_concepts)
    df: Counter[str] = Counter()
    for concepts in photo_concepts.values():
        df.update(set(concepts))  # count each concept at most once per photo

    return {concept: math.log((n + 1) / (freq + 1)) for concept, freq in df.items()}


# ── Concept overlap score ──────────────────────────────────────────────────────

def concept_overlap_score(
    query_concepts: list[str],
    photo_concepts: list[str],
    idf: dict[str, float],
) -> float:
    """
    Computes the IDF-weighted concept overlap between a query and a photo.

    Score = sum of IDF(c) for every concept c in the intersection of
    query_concepts and photo_concepts.  Concepts not present in the IDF
    table (unseen at index time) contribute 0.

    Args:
        query_concepts:  concepts extracted from the query (normalised IDs).
        photo_concepts:  concepts stored for the photo (normalised IDs).
        idf:             IDF table produced by :func:`compute_idf`.

    Returns:
        Non-negative float; 0.0 when there is no overlap.
    """
    query_set = set(query_concepts)
    photo_set = set(photo_concepts)
    return sum(idf.get(c, 0.0) for c in query_set & photo_set)


# ── Re-ranker ─────────────────────────────────────────────────────────────────

def rerank_by_concept_overlap(
    candidates: list[str],
    query_concepts: list[str],
    photo_concept_map: dict[str, list[str]],
    idf: dict[str, float],
    top_k: int,
) -> list[str]:
    """
    Re-ranks candidate doc_ids by IDF-weighted concept overlap and returns
    the top-k results.

    Candidates with no entry in photo_concept_map receive a score of 0.0
    but are still included (they stay at the bottom of the re-ranked list).

    Args:
        candidates:        doc_ids from the first-stage retriever (B), in
                           retrieval-score order.
        query_concepts:    normalised concept IDs extracted from the query.
        photo_concept_map: mapping of {doc_id -> [concept_id, ...]}.
        idf:               IDF weights from :func:`compute_idf`.
        top_k:             number of results to return; must be ≥ 1.

    Returns:
        Re-ranked list of doc_ids, length ≤ top_k.
    """
    scored = [
        (
            doc_id,
            concept_overlap_score(
                query_concepts,
                photo_concept_map.get(doc_id, []),
                idf,
            ),
        )
        for doc_id in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scored[:top_k]]
