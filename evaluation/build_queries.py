"""
build_queries.py — 평가용 쿼리 CSV 생성 헬퍼

queries.csv 를 직접 편집해도 되지만, 이 스크립트를 사용하면
id_map.csv 에서 known-item 쿼리를 자동으로 생성할 수 있습니다.

쿼리 유형:
    known_item: 특정 이미지 파일명에서 쿼리 자동 생성 (doc_id → 쿼리)
    broad:      폭넓은 의미 검색 쿼리 (수동 작성)

사용법:
    # 자동 생성 (id_map.csv 기반 known-item 쿼리)
    python -m evaluation.build_queries --auto

    # 기존 queries.csv 에 수동 쿼리 추가
    python -m evaluation.build_queries --add "카페에서 커피 마시는 사람" broad

출력:
    evaluation/data/queries.csv — (query_id, query_ko, query_en, type)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import evaluation.config as cfg


FIELDNAMES = ["query_id", "query_ko", "query_en", "type"]


def _load_existing() -> list[dict]:
    if not cfg.QUERIES_PATH.exists():
        return []
    with cfg.QUERIES_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(rows: list[dict]) -> None:
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with cfg.QUERIES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _next_id(rows: list[dict]) -> int:
    if not rows:
        return 1
    return max(int(r["query_id"]) for r in rows) + 1


def generate_known_item_queries() -> None:
    """
    id_map.csv 의 각 이미지 파일명에서 known-item 쿼리를 생성합니다.
    query_ko 는 doc_id를 사람이 읽기 쉬운 형태로 변환한 값으로 초기화되며,
    실제 평가에 쓰기 전에 수동으로 편집해야 합니다.

    query_en 은 비워두고 A' 방식에서 번역을 수행합니다.
    """
    if not cfg.ID_MAP_PATH.exists():
        raise FileNotFoundError(
            f"id_map.csv 없음: {cfg.ID_MAP_PATH}\n"
            "먼저 prepare_corpus.py를 실행하세요."
        )

    with cfg.ID_MAP_PATH.open(encoding="utf-8") as f:
        id_rows = list(csv.DictReader(f))

    existing = _load_existing()
    existing_doc_ids = {r.get("query_ko", "") for r in existing}
    next_id = _next_id(existing)

    new_rows: list[dict] = []
    for row in id_rows:
        doc_id = row["doc_id"]
        label = doc_id.replace("__", " / ").replace("_", " ")
        if label in existing_doc_ids:
            continue
        new_rows.append(
            {
                "query_id": next_id,
                "query_ko": label,
                "query_en": "",
                "type": "known_item",
            }
        )
        next_id += 1

    _save(existing + new_rows)
    print(f"쿼리 {len(new_rows)}건 추가 (합계 {len(existing) + len(new_rows)}건)")
    print(f"  → {cfg.QUERIES_PATH}")
    print("  ※ query_ko 값을 실제 검색 쿼리로 수동 편집하세요.")


def add_query(query_ko: str, query_type: str) -> None:
    """
    쿼리를 queries.csv 에 한 건 추가합니다.

    Args:
        query_ko:   한국어 쿼리 문자열.
        query_type: 'known_item' 또는 'broad'.
    """
    rows = _load_existing()
    rows.append(
        {
            "query_id": _next_id(rows),
            "query_ko": query_ko,
            "query_en": "",
            "type": query_type,
        }
    )
    _save(rows)
    print(f"쿼리 추가 완료: [{query_type}] {query_ko}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="평가용 쿼리 CSV를 관리합니다.",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("--auto", help="id_map.csv 기반 known-item 쿼리 자동 생성")

    add_p = sub.add_parser("--add", help="쿼리 수동 추가")
    add_p.add_argument("query_ko", help="한국어 쿼리")
    add_p.add_argument(
        "type",
        choices=["known_item", "broad"],
        default="broad",
        nargs="?",
    )

    args = parser.parse_args()

    if args.cmd == "--auto":
        generate_known_item_queries()
    elif args.cmd == "--add":
        add_query(args.query_ko, args.type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
