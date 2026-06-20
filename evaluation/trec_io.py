"""
TREC-format I/O utilities.

TREC qrels format (one line per judgement):
    query_id  0  doc_id  relevance

TREC run format (one line per result):
    query_id  0  doc_id  rank  score  method_name

All functions are pure (no side effects) to stay testable.
"""

from __future__ import annotations

from pathlib import Path


# ── Qrels ─────────────────────────────────────────────────────────────────────

def write_qrels(qrels: dict[str, dict[str, int]], path: Path) -> None:
    """
    Writes relevance judgements in TREC qrels format.

    Args:
        qrels: mapping of {query_id -> {doc_id -> relevance_score}}.
               Relevance is typically 0 (irrelevant) to 3 (highly relevant).
        path:  destination file path; parent directories must exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for qid, docs in sorted(qrels.items()):
        for doc_id, rel in sorted(docs.items()):
            lines.append(f"{qid} 0 {doc_id} {rel}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """
    Reads a TREC qrels file.

    Args:
        path: path to qrels file.

    Returns:
        Mapping of {query_id -> {doc_id -> relevance_score}}.

    Raises:
        ValueError: if any line does not have exactly 4 whitespace-separated fields.
    """
    qrels: dict[str, dict[str, int]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"qrels line {lineno}: expected 4 fields, got {len(parts)}: {line!r}")
        qid, _, doc_id, rel = parts
        qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


# ── Run files ─────────────────────────────────────────────────────────────────

def write_run(
    results: dict[str, list[str]],
    method_name: str,
    path: Path,
) -> None:
    """
    Writes ranked search results in TREC run format.

    Args:
        results:     mapping of {query_id -> [doc_id, ...]} in rank order.
        method_name: tag written in the 6th column (identifies the system).
        path:        destination file path; parent directories must exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for qid, doc_ids in sorted(results.items()):
        for rank, doc_id in enumerate(doc_ids, 1):
            score = 1.0 / rank  # reciprocal-rank proxy score
            lines.append(f"{qid} 0 {doc_id} {rank} {score:.6f} {method_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_run(path: Path) -> dict[str, list[str]]:
    """
    Reads a TREC run file.

    Args:
        path: path to run file.

    Returns:
        Mapping of {query_id -> [doc_id, ...]} sorted by ascending rank.

    Raises:
        ValueError: if any line does not have exactly 6 whitespace-separated fields.
    """
    rows: dict[str, list[tuple[int, str]]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"run line {lineno}: expected 6 fields, got {len(parts)}: {line!r}")
        qid, _, doc_id, rank, _score, _tag = parts
        rows.setdefault(qid, []).append((int(rank), doc_id))
    return {qid: [d for _, d in sorted(entries)] for qid, entries in rows.items()}
