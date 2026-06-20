"""
evaluate.py — ranx로 검색 성능 지표를 산출하고 비교 리포트를 출력합니다

사전 조건:
    - make_qrels.py 실행 완료 (qrels.txt 존재)
    - run_search.py 실행 완료 (runs/*.txt 존재)

사용법:
    cd PhoMate_FastAPI_BE
    python -m evaluation.evaluate

    # 특정 run 파일만 평가
    python -m evaluation.evaluate --runs A.txt A_prime.txt

출력:
    콘솔: 지표 비교 테이블
    evaluation/data/results_summary.csv   — 방식별 집계 지표
    evaluation/data/results_per_query.csv — 쿼리별 세부 지표

지표:
    recall@1, recall@5, recall@10
    precision@5, precision@10
    mrr@10
    ndcg@10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ranx import Qrels, Run, compare, evaluate

import evaluation.config as cfg


METRICS = [
    "recall@1",
    "recall@5",
    "recall@10",
    "precision@5",
    "precision@10",
    "mrr@10",
    "ndcg@10",
]


def _load_qrels() -> Qrels:
    if not cfg.QRELS_PATH.exists():
        raise FileNotFoundError(
            f"qrels.txt 없음: {cfg.QRELS_PATH}\n"
            "먼저 make_qrels.py를 실행하세요."
        )
    return Qrels.from_file(str(cfg.QRELS_PATH), kind="trec")


def _load_runs(run_names: list[str] | None) -> list[Run]:
    """
    runs/ 디렉토리에서 TREC run 파일을 읽어 ranx Run 객체 목록을 반환합니다.

    Args:
        run_names: 파일명 목록 (None이면 전체 *.txt).

    Returns:
        List of ranx Run objects.

    Raises:
        FileNotFoundError: if no run files are found.
    """
    if run_names:
        paths = [cfg.RUNS_DIR / n for n in run_names]
    else:
        paths = sorted(cfg.RUNS_DIR.glob("*.txt"))

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"run 파일 없음: {missing}\n"
            "먼저 run_search.py를 실행하세요."
        )
    if not paths:
        raise FileNotFoundError(
            f"runs/ 에 .txt 파일이 없습니다: {cfg.RUNS_DIR}\n"
            "먼저 run_search.py를 실행하세요."
        )

    return [Run.from_file(str(p), kind="trec") for p in paths]


def _write_summary(results: dict[str, dict[str, float]], path: Path) -> None:
    """
    방식별 집계 지표를 CSV로 저장합니다.

    Args:
        results: {method_name -> {metric -> score}} mapping.
        path:    destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method"] + METRICS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, scores in sorted(results.items()):
            writer.writerow({"method": method, **{m: f"{scores.get(m, 0):.4f}" for m in METRICS}})


def _write_per_query(
    qrels: Qrels,
    runs: list[Run],
    path: Path,
) -> None:
    """
    쿼리별 세부 지표를 CSV로 저장합니다.

    Args:
        qrels: ranx Qrels object.
        runs:  list of ranx Run objects.
        path:  destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run in runs:
        method = run.name or "unknown"
        # return_mean=False 시 ranx는 {metric: np.ndarray} 반환
        # 배열 순서는 run.run 의 key 순서와 동일
        per_query = evaluate(qrels, run, METRICS, return_mean=False)
        query_ids_ordered = list(run.run.keys())
        for i, qid in enumerate(query_ids_ordered):
            row: dict = {"method": method, "query_id": qid}
            for metric in METRICS:
                arr = per_query.get(metric, [])
                score = float(arr[i]) if i < len(arr) else 0.0
                row[metric] = f"{score:.4f}"
            rows.append(row)

    fieldnames = ["method", "query_id"] + METRICS
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(run_names: list[str] | None = None) -> None:
    """
    전체 평가 파이프라인을 실행합니다.

    Args:
        run_names: 특정 run 파일명 목록 (None이면 전체 *.txt).
    """
    qrels = _load_qrels()
    runs = _load_runs(run_names)

    print(f"qrels: {len(qrels.qrels)}개 쿼리")
    print(f"runs:  {len(runs)}개 방식\n")

    # ── per-method aggregate scores
    summary: dict[str, dict[str, float]] = {}
    for run in runs:
        scores = evaluate(qrels, run, METRICS)
        method = run.name or "unknown"
        summary[method] = dict(scores)

    # ── comparison table (ranx built-in)
    if len(runs) >= 2:
        print("── 방식 비교 ──")
        report = compare(
            qrels,
            runs,
            METRICS,
            max_p=0.05,  # statistical significance threshold
        )
        print(report)
    else:
        print("── 단일 방식 결과 ──")
        method, scores = next(iter(summary.items()))
        for metric, score in scores.items():
            print(f"  {metric:<20} {score:.4f}")

    # ── write CSVs
    summary_path = cfg.DATA_DIR / "results_summary.csv"
    per_query_path = cfg.DATA_DIR / "results_per_query.csv"

    _write_summary(summary, summary_path)
    _write_per_query(qrels, runs, per_query_path)

    print(f"\n결과 저장:")
    print(f"  집계: {summary_path}")
    print(f"  쿼리별: {per_query_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="검색 성능 지표를 산출하고 방식별로 비교합니다.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        metavar="FILE",
        default=None,
        help="평가할 run 파일명 (e.g. A.txt A_prime.txt). 생략 시 runs/ 전체.",
    )
    args = parser.parse_args()
    run_evaluation(run_names=args.runs)


if __name__ == "__main__":
    main()
