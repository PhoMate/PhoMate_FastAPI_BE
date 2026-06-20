"""
prepare_coco_queries.py — MS COCO 캡션 기반 평가 쿼리·정답 자동 생성

captions_val2017.json 에서 이미지별 캡션을 추출하고, id_map.csv 와 매핑하여
queries.csv 와 qrels.txt 를 한 번에 생성합니다.

사전 조건:
    - prepare_corpus.py 실행 완료 (id_map.csv 존재)
    - COCO annotations 다운로드 완료 (captions_val2017.json)

사용법:
    cd PhoMate_FastAPI_BE
    python -m evaluation.prepare_coco_queries \\
        --captions /path/to/annotations/captions_val2017.json

    # VLM_API_KEY 설정 시 한국어 번역 포함
    VLM_API_KEY=sk-... python -m evaluation.prepare_coco_queries \\
        --captions /path/to/annotations/captions_val2017.json --translate

출력:
    evaluation/data/queries.csv  — (query_id, query_ko, query_en, type)
    evaluation/data/qrels.txt    — known-item 정답 (relevance=3)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import evaluation.config as cfg
from evaluation.trec_io import write_qrels

QUERY_FIELDNAMES = ["query_id", "query_ko", "query_en", "type"]


# ── Caption selection ─────────────────────────────────────────────────────────

def _pick_caption(captions: list[str]) -> str:
    """이미지의 캡션 후보 중 가장 긴 것을 선택합니다."""
    return max(captions, key=len)


# ── Translation (optional) ────────────────────────────────────────────────────

def _translate_batch(texts: list[str]) -> list[str]:
    """
    영어 캡션 목록을 한국어로 일괄 번역합니다.

    VLM_API_KEY 가 설정되어 있어야 합니다.
    API 호출 비용 절감을 위해 10건씩 묶어서 요청합니다.

    Args:
        texts: 영어 캡션 문자열 목록.

    Returns:
        한국어 번역 문자열 목록 (순서 유지).
    """
    import openai

    client = openai.OpenAI(api_key=cfg.VLM_API_KEY, base_url=cfg.OPENAI_BASE_URL)
    results: list[str] = []
    CHUNK = 10

    for i in tqdm(range(0, len(texts), CHUNK), desc="번역"):
        chunk = texts[i : i + CHUNK]
        numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(chunk))
        resp = client.chat.completions.create(
            model=cfg.VLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "아래 영어 사진 검색 쿼리들을 자연스러운 한국어로 번역하세요. "
                        "번호와 번역문만 출력하고, 설명은 쓰지 마세요.\n"
                        "예시)\n1. 영어 문장\n→\n1. 한국어 번역"
                    ),
                },
                {"role": "user", "content": numbered},
            ],
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        # 줄별로 파싱: "1. ..." 형태에서 텍스트만 추출
        translated = []
        for line in raw.splitlines():
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                translated.append(line.split(". ", 1)[1].strip())
        # 파싱 실패 시 영어 원문 유지
        if len(translated) != len(chunk):
            translated = chunk
        results.extend(translated)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def build_coco_queries(captions_json: Path, translate: bool = False) -> None:
    """
    COCO 캡션 JSON 과 id_map.csv 를 매핑하여 queries.csv 와 qrels.txt 를 생성합니다.

    id_map.csv 에 등록된 이미지만 대상으로 합니다 (코퍼스 범위 내로 제한).

    Args:
        captions_json: captions_val2017.json 경로.
        translate:     True 이면 영어 캡션을 한국어로 번역하여 query_ko 에 저장.
                       False 이면 query_ko 에 영어 캡션을 그대로 사용합니다.
    """
    if not cfg.ID_MAP_PATH.exists():
        raise FileNotFoundError(
            f"id_map.csv 없음: {cfg.ID_MAP_PATH}\n"
            "먼저 prepare_corpus.py 를 실행하세요."
        )
    if not captions_json.exists():
        raise FileNotFoundError(f"captions JSON 없음: {captions_json}")

    # ── id_map 로드: doc_id → qdrant_id
    with cfg.ID_MAP_PATH.open(encoding="utf-8") as f:
        id_map = {row["doc_id"]: row["qdrant_id"] for row in csv.DictReader(f)}

    # ── COCO 캡션 로드
    print(f"COCO 캡션 로드 중: {captions_json}")
    with captions_json.open(encoding="utf-8") as f:
        coco = json.load(f)

    # image_id → file_name (확장자 제외 = doc_id)
    image_id_to_doc: dict[int, str] = {
        img["id"]: Path(img["file_name"]).stem
        for img in coco["images"]
    }

    # image_id → 캡션 목록
    captions_by_image: dict[int, list[str]] = {}
    for ann in coco["annotations"]:
        captions_by_image.setdefault(ann["image_id"], []).append(ann["caption"])

    # ── 코퍼스 내 이미지만 필터링
    pairs: list[tuple[str, str]] = []  # (doc_id, english_caption)
    for image_id, doc_id in image_id_to_doc.items():
        if doc_id not in id_map:
            continue
        caps = captions_by_image.get(image_id, [])
        if not caps:
            continue
        pairs.append((doc_id, _pick_caption(caps)))

    pairs.sort(key=lambda x: x[0])
    print(f"매핑된 이미지: {len(pairs)}장 (코퍼스 {len(id_map)}장 중)")

    if not pairs:
        print("매핑 결과가 없습니다. id_map.csv 와 captions JSON 의 이미지가 일치하는지 확인하세요.")
        return

    # ── 번역 (선택)
    english_captions = [cap for _, cap in pairs]
    if translate:
        if not cfg.VLM_API_KEY:
            print("[경고] VLM_API_KEY 가 설정되지 않아 번역을 건너뜁니다. query_ko 에 영어를 사용합니다.")
            korean_captions = english_captions
        else:
            print(f"{len(english_captions)}건 번역 시작 (모델: {cfg.VLM_MODEL})")
            korean_captions = _translate_batch(english_captions)
    else:
        print("번역 건너뜀. query_ko 에 영어 캡션을 사용합니다. (--translate 옵션으로 한국어 번역 가능)")
        korean_captions = english_captions

    # ── queries.csv 작성
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    query_rows: list[dict] = []
    qrels: dict[str, dict[str, int]] = {}

    for query_id, ((doc_id, en_cap), ko_cap) in enumerate(
        zip(pairs, korean_captions), start=1
    ):
        qid = str(query_id)
        query_rows.append(
            {
                "query_id": qid,
                "query_ko": ko_cap,
                "query_en": en_cap,
                "type": "known_item",
            }
        )
        qrels[qid] = {doc_id: 3}

    with cfg.QUERIES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUERY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(query_rows)

    # ── qrels.txt 작성
    write_qrels(qrels, cfg.QRELS_PATH)

    print(f"\n완료:")
    print(f"  queries.csv : {len(query_rows)}건 → {cfg.QUERIES_PATH}")
    print(f"  qrels.txt   : {len(qrels)}건 → {cfg.QRELS_PATH}")
    if not translate:
        print("\n※ A 방식(한국어 쿼리) 평가를 위해 --translate 옵션으로 한국어 번역을 추가할 수 있습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MS COCO 캡션 기반 평가 쿼리와 정답을 자동 생성합니다.",
    )
    parser.add_argument(
        "--captions",
        required=True,
        type=Path,
        help="captions_val2017.json 경로",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="영어 캡션을 한국어로 번역 (VLM_API_KEY 필요)",
    )
    args = parser.parse_args()
    build_coco_queries(args.captions, translate=args.translate)


if __name__ == "__main__":
    main()
