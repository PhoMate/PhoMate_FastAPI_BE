"""
TREC I/O 순수 함수 테스트 — 모델·DB 의존성 없음.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from evaluation.trec_io import write_qrels, read_qrels, write_run, read_run


# ── write_qrels / read_qrels ──────────────────────────────────────────────────

def test_write_and_read_qrels_roundtrip(tmp_path: Path) -> None:
    qrels = {
        "q1": {"doc_a": 3, "doc_b": 1},
        "q2": {"doc_c": 2},
    }
    path = tmp_path / "qrels.txt"
    write_qrels(qrels, path)
    recovered = read_qrels(path)
    assert recovered == qrels


def test_write_qrels_file_format(tmp_path: Path) -> None:
    qrels = {"q1": {"d1": 3}}
    path = tmp_path / "qrels.txt"
    write_qrels(qrels, path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert lines[0] == "q1 0 d1 3"


def test_read_qrels_empty_lines_ignored(tmp_path: Path) -> None:
    path = tmp_path / "qrels.txt"
    path.write_text("q1 0 d1 1\n\n\nq1 0 d2 0\n")
    result = read_qrels(path)
    assert result == {"q1": {"d1": 1, "d2": 0}}


def test_read_qrels_invalid_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("q1 d1 1\n")  # 3 fields instead of 4
    with pytest.raises(ValueError, match="4 fields"):
        read_qrels(path)


def test_read_qrels_multiple_queries(tmp_path: Path) -> None:
    path = tmp_path / "qrels.txt"
    path.write_text(
        "q1 0 d1 3\n"
        "q1 0 d2 0\n"
        "q2 0 d3 2\n"
    )
    result = read_qrels(path)
    assert result["q1"]["d1"] == 3
    assert result["q1"]["d2"] == 0
    assert result["q2"]["d3"] == 2


# ── write_run / read_run ──────────────────────────────────────────────────────

def test_write_and_read_run_roundtrip(tmp_path: Path) -> None:
    results = {
        "q1": ["doc_a", "doc_b", "doc_c"],
        "q2": ["doc_x"],
    }
    path = tmp_path / "run.txt"
    write_run(results, "method_A", path)
    recovered = read_run(path)
    assert recovered == results


def test_write_run_file_format(tmp_path: Path) -> None:
    results = {"q1": ["d1", "d2"]}
    path = tmp_path / "run.txt"
    write_run(results, "A", path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    # first line: q1 0 d1 1 <score> A
    parts = lines[0].split()
    assert parts[0] == "q1"
    assert parts[2] == "d1"
    assert parts[3] == "1"
    assert parts[5] == "A"


def test_write_run_scores_decrease_by_rank(tmp_path: Path) -> None:
    results = {"q1": ["d1", "d2", "d3"]}
    path = tmp_path / "run.txt"
    write_run(results, "A", path)
    lines = path.read_text().strip().splitlines()
    scores = [float(line.split()[4]) for line in lines]
    # reciprocal rank: 1/1, 1/2, 1/3 — strictly decreasing
    assert scores == sorted(scores, reverse=True)


def test_read_run_invalid_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("q1 0 d1 1\n")  # 4 fields instead of 6
    with pytest.raises(ValueError, match="6 fields"):
        read_run(path)


def test_read_run_preserves_rank_order(tmp_path: Path) -> None:
    path = tmp_path / "run.txt"
    # Intentionally write out of rank order
    path.write_text(
        "q1 0 d3 3 0.333333 A\n"
        "q1 0 d1 1 1.000000 A\n"
        "q1 0 d2 2 0.500000 A\n"
    )
    result = read_run(path)
    assert result["q1"] == ["d1", "d2", "d3"]
