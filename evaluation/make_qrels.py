"""
make_qrels.py — 정답(qrels.txt) 생성

두 가지 방식으로 qrels를 구성합니다:

  known_item:  queries.csv 에서 type=known_item 인 쿼리의 정답을 doc_id 기반으로 자동 생성.
               query_ko 가 doc_id 와 일치하면 relevance=3 (exact match).

  vlm_judge:   모든 방식의 run 파일을 pooling하여 후보를 뽑고, VLM으로 0-3 relevance 레이블.
               (TODO: VLM 판단 구현)

사전 조건:
    - prepare_corpus.py 실행 완료 (id_map.csv 존재)
    - build_queries.py 실행 완료 (queries.csv 존재)
    - vlm_judge 사용 시: run_search.py 실행 완료 (runs/ 존재)

사용법:
    # known_item 쿼리만 (VLM 불필요, 즉시 실행 가능)
    python -m evaluation.make_qrels --mode known_item

    # VLM pooling + 판단 (TODO)
    python -m evaluation.make_qrels --mode vlm_judge
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import evaluation.config as cfg
from evaluation.trec_io import write_qrels, read_run


# ── known_item qrels ──────────────────────────────────────────────────────────

def build_known_item_qrels() -> dict[str, dict[str, int]]:
    """
    known_item 타입 쿼리에 대해 qrels를 자동 생성합니다.

    queries.csv 의 query_ko 값을 doc_id 와 비교하여 exact match인 경우
    relevance=3 을 부여합니다.  일치하지 않으면 수동으로 편집해야 합니다.

    Returns:
        {query_id -> {doc_id -> relevance}} mapping.

    Raises:
        FileNotFoundError: if queries.csv or id_map.csv are missing.
    """
    for path in (cfg.QUERIES_PATH, cfg.ID_MAP_PATH):
        if not path.exists():
            raise FileNotFoundError(f"파일 없음: {path}")

    with cfg.ID_MAP_PATH.open(encoding="utf-8") as f:
        doc_ids = {row["doc_id"] for row in csv.DictReader(f)}

    qrels: dict[str, dict[str, int]] = {}
    with cfg.QUERIES_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["type"] != "known_item":
                continue
            qid = row["query_id"]
            query_ko = row["query_ko"]
            # doc_id 의 __ 구분자를 원래 label 형태와 비교
            matched = [d for d in doc_ids if d.replace("__", " / ").replace("_", " ") == query_ko]
            if matched:
                qrels[qid] = {doc_id: 3 for doc_id in matched}
            else:
                # 매치 없음 — 빈 항목으로 남겨 수동 편집 유도
                qrels[qid] = {}

    return qrels


# ── VLM pooling + judge (TODO) ────────────────────────────────────────────────

def build_vlm_judge_qrels(pool_depth: int = 10) -> dict[str, dict[str, int]]:
    """
    모든 run 파일을 pooling하여 후보를 수집하고, VLM으로 relevance를 판단합니다.

    TODO: VLM 판단 구현.

    Args:
        pool_depth: 각 run 파일에서 pooling 할 깊이.

    Returns:
        {query_id -> {doc_id -> relevance}} mapping.

    Raises:
        FileNotFoundError: if runs/ directory is empty.
        NotImplementedError: until VLM judge is implemented.
    """
    if not cfg.RUNS_DIR.exists() or not list(cfg.RUNS_DIR.glob("*.txt")):
        raise FileNotFoundError(
            f"run 파일 없음: {cfg.RUNS_DIR}\n"
            "먼저 run_search.py를 실행하세요."
        )

    # Pool top-N candidates from all runs
    pool: dict[str, set[str]] = {}  # {query_id -> set of doc_ids}
    for run_path in cfg.RUNS_DIR.glob("*.txt"):
        run = read_run(run_path)
        for qid, doc_ids in run.items():
            pool.setdefault(qid, set()).update(doc_ids[:pool_depth])

    # TODO: call VLM to judge each (query, doc) pair
    # qrels: dict[str, dict[str, int]] = {}
    # for qid, candidates in pool.items():
    #     qrels[qid] = {}
    #     query_ko = _load_query_text(qid)
    #     for doc_id in candidates:
    #         image_path = _doc_id_to_path(doc_id)
    #         relevance = _vlm_judge(query_ko, image_path)  # 0, 1, 2, or 3
    #         qrels[qid][doc_id] = relevance
    # return qrels
    raise NotImplementedError(
        "VLM judge 미구현. VLM_API_KEY 설정 후 build_vlm_judge_qrels 코드를 주석 해제하세요."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="평가용 정답(qrels.txt)을 생성합니다.",
    )
    parser.add_argument(
        "--mode",
        choices=["known_item", "vlm_judge"],
        default="known_item",
        help="qrels 생성 방식 (기본: known_item)",
    )
    parser.add_argument(
        "--pool-depth",
        type=int,
        default=10,
        help="vlm_judge 모드에서 각 run 파일의 pooling 깊이 (기본: 10)",
    )
    args = parser.parse_args()

    if args.mode == "known_item":
        qrels = build_known_item_qrels()
    else:
        qrels = build_vlm_judge_qrels(pool_depth=args.pool_depth)

    write_qrels(qrels, cfg.QRELS_PATH)

    total_judgements = sum(len(v) for v in qrels.values())
    print(f"\n완료: {len(qrels)}개 쿼리, {total_judgements}건 판단 → {cfg.QRELS_PATH}")

    empty = [qid for qid, docs in qrels.items() if not docs]
    if empty:
        print(f"  [경고] 정답이 없는 쿼리 {len(empty)}건: {empty}")
        print("  → qrels.txt 를 수동으로 편집하거나 --mode vlm_judge 를 사용하세요.")


if __name__ == "__main__":
    main()
