"""
prepare_concepts.py — VLM concept 추출 → photo_concepts.csv + concept_vocab.csv

각 이미지에서 mood/place/object/style 개념을 VLM으로 추출하고,
정규화된 concept_id로 변환한 뒤 CSV로 저장합니다.
(Qdrant 저장은 하지 않음 — scoring.py에서 CSV를 직접 읽어 IDF 가중 재랭킹)

사전 조건:
    - prepare_corpus.py 실행 완료 (id_map.csv 존재)
    - .env 에 VLM_API_KEY, VLM_MODEL 설정

사용법:
    cd PhoMate_FastAPI_BE
    python -m evaluation.prepare_concepts

출력:
    evaluation/data/concept_vocab.csv  — (concept_id, label)
    evaluation/data/photo_concepts.csv — (doc_id, concept_id, confidence)

TODO:
    - VLM concept 추출 (현재 NotImplementedError)
    - concept 정규화 로직
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import evaluation.config as cfg


# ── Concept categories ────────────────────────────────────────────────────────
# VLM에 요청할 카테고리. 추후 확장 가능.
CONCEPT_CATEGORIES = ["mood", "place", "object", "style"]


# ── VLM concept extraction (TODO) ─────────────────────────────────────────────

def extract_concepts(image_path: Path) -> list[dict]:
    """
    Extracts visual concepts from an image using a VLM.

    TODO: implement OpenAI vision API call.

    Args:
        image_path: path to local image file.

    Returns:
        List of dicts with keys 'category', 'label', 'confidence' (0.0–1.0).
        Example: [{"category": "mood", "label": "평화로운", "confidence": 0.9}, ...]

    Raises:
        NotImplementedError: until VLM_API_KEY is configured.
    """
    # import base64, json, openai
    # client = openai.OpenAI(api_key=cfg.VLM_API_KEY, base_url=cfg.OPENAI_BASE_URL)
    # with open(image_path, "rb") as f:
    #     b64 = base64.b64encode(f.read()).decode()
    # ext = image_path.suffix.lstrip(".").lower()
    # mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    # system_prompt = (
    #     "You are a visual concept extractor. "
    #     "Given an image, return a JSON array of concepts in these categories: "
    #     f"{', '.join(CONCEPT_CATEGORIES)}. "
    #     "Each item: {{\"category\": str, \"label\": str (Korean OK), \"confidence\": float 0-1}}. "
    #     "Return only valid JSON, no explanation."
    # )
    # resp = client.chat.completions.create(
    #     model=cfg.VLM_MODEL,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": [
    #             {"type": "image_url",
    #              "image_url": {"url": f"data:{mime};base64,{b64}"}},
    #         ]},
    #     ],
    #     max_tokens=256,
    # )
    # return json.loads(resp.choices[0].message.content)
    raise NotImplementedError(
        "VLM concept 추출 미구현. VLM_API_KEY 설정 후 extract_concepts 코드를 주석 해제하세요."
    )


# ── Concept normalisation ─────────────────────────────────────────────────────

def normalise_concept(category: str, label: str) -> str:
    """
    Converts a (category, label) pair into a stable concept_id string.

    concept_id format: "<category>:<normalised_label>"
    Normalisation: lower-case, strip whitespace, replace spaces with '_'.

    Args:
        category: one of CONCEPT_CATEGORIES.
        label:    raw label from VLM (may be Korean or English).

    Returns:
        concept_id string (e.g. "mood:평화로운", "place:카페").
    """
    normalised = label.strip().lower().replace(" ", "_")
    return f"{category}:{normalised}"


# ── Main extraction logic ──────────────────────────────────────────────────────

def build_concept_index() -> None:
    """
    Reads id_map.csv, extracts concepts via VLM, and writes:
      - concept_vocab.csv  : (concept_id, category, label)
      - photo_concepts.csv : (doc_id, concept_id, confidence)

    Raises:
        FileNotFoundError: if id_map.csv does not exist.
        NotImplementedError: until VLM integration is implemented.
    """
    if not cfg.ID_MAP_PATH.exists():
        raise FileNotFoundError(
            f"id_map.csv 없음: {cfg.ID_MAP_PATH}\n"
            "먼저 prepare_corpus.py를 실행하세요."
        )

    with cfg.ID_MAP_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"concept 추출 대상: {len(rows)}장")

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    vocab: dict[str, dict] = {}          # concept_id → {category, label}
    photo_concepts: list[dict] = []

    for row in tqdm(rows, desc="concept 추출"):
        doc_id = row["doc_id"]
        filepath = Path(row["filepath"])

        concepts_raw = extract_concepts(filepath)  # TODO

        for item in concepts_raw:
            category   = item["category"]
            label      = item["label"]
            confidence = float(item.get("confidence", 1.0))
            concept_id = normalise_concept(category, label)

            if concept_id not in vocab:
                vocab[concept_id] = {"concept_id": concept_id,
                                     "category": category,
                                     "label": label}
            photo_concepts.append(
                {"doc_id": doc_id, "concept_id": concept_id, "confidence": confidence}
            )

    # ── write vocab
    with cfg.CONCEPT_VOCAB_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_id", "category", "label"])
        writer.writeheader()
        writer.writerows(sorted(vocab.values(), key=lambda r: r["concept_id"]))

    # ── write photo concepts
    with cfg.PHOTO_CONCEPTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "concept_id", "confidence"])
        writer.writeheader()
        writer.writerows(photo_concepts)

    print(f"\n완료: vocab {len(vocab)}개 concept, photo_concepts {len(photo_concepts)}행")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM으로 이미지별 시각 개념을 추출하여 CSV로 저장합니다.",
    )
    parser.parse_args()
    build_concept_index()


if __name__ == "__main__":
    main()
