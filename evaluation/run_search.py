"""
run_search.py — 각 검색 방식으로 쿼리를 실행하고 TREC run 파일을 생성합니다

사전 조건:
    - prepare_corpus.py 실행 완료 (eval_siglip2 인덱싱)
    - make_qrels.py 실행 완료 (queries.csv, qrels.txt 존재)

사용법:
    # 모든 방식 실행
    cd PhoMate_FastAPI_BE
    python -m evaluation.run_search

    # 특정 방식만 실행
    python -m evaluation.run_search --methods A A_prime

출력:
    evaluation/data/runs/<method>.txt  (TREC run 포맷)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import evaluation.config as cfg
from evaluation.methods import get_method, list_methods
from evaluation.trec_io import write_run


def load_queries() -> list[dict]:
    """
    queries.csv 를 읽어 쿼리 목록을 반환합니다.

    Returns:
        List of dicts with keys: query_id, query_ko, query_en, type.

    Raises:
        FileNotFoundError: if queries.csv does not exist.
    """
    if not cfg.QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"queries.csv 없음: {cfg.QUERIES_PATH}\n"
            "먼저 build_queries.py를 실행하세요."
        )
    with cfg.QUERIES_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_method(method_name: str, queries: list[dict]) -> None:
    """
    지정된 방식으로 모든 쿼리를 검색하고 TREC run 파일로 저장합니다.

    A' 방식은 query_en 필드를 우선 사용하고 없으면 런타임 번역을 시도합니다.

    Args:
        method_name: 레지스트리에 등록된 방식 이름 (e.g. "A", "A_prime").
        queries:     쿼리 목록 (load_queries() 반환값).
    """
    method = get_method(method_name)
    results: dict[str, list[str]] = {}
    errors: list[str] = []

    for row in tqdm(queries, desc=f"[{method_name}] 검색"):
        qid = row["query_id"]
        # A_prime 방식은 query_en 이 있으면 번역 API를 건너뜁니다
        if method_name == "A_prime" and row.get("query_en"):
            query = row["query_en"]
        else:
            query = row["query_ko"]

        try:
            doc_ids = method.search(query, top_k=cfg.TOP_K)
            results[qid] = doc_ids
        except NotImplementedError as exc:
            errors.append(f"  qid={qid}: {exc}")
            results[qid] = []
        except Exception as exc:
            errors.append(f"  qid={qid}: {type(exc).__name__}: {exc}")
            results[qid] = []

    out_path = cfg.RUNS_DIR / f"{method_name}.txt"
    write_run(results, method_name, out_path)

    success = sum(1 for v in results.values() if v)
    print(f"  저장: {out_path} ({success}/{len(queries)} 쿼리 성공)")
    if errors:
        print(f"  [경고] {len(errors)}건 오류:")
        for e in errors[:5]:
            print(e)
        if len(errors) > 5:
            print(f"  ... 외 {len(errors) - 5}건")


def main() -> None:
    available = list_methods()
    parser = argparse.ArgumentParser(
        description="평가용 검색을 실행하고 TREC run 파일을 생성합니다.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=available,
        default=available,
        metavar="METHOD",
        help=f"실행할 방식 (기본: 전체). 선택 가능: {available}",
    )
    args = parser.parse_args()

    queries = load_queries()
    print(f"쿼리 {len(queries)}건 로드 완료\n")

    cfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    for method_name in args.methods:
        print(f"── {method_name} ──")
        run_method(method_name, queries)
        print()

    print("전체 완료.")


if __name__ == "__main__":
    main()
