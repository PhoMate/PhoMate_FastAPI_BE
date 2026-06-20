"""
prepare_captions.py — VLM 캡션 생성 → 텍스트 임베딩 → eval_caption 인덱싱

id_map.csv 의 각 이미지에 대해 VLM으로 영어 캡션을 생성하고,
텍스트 임베더로 벡터화한 뒤 eval_caption 컬렉션에 upsert합니다.

사전 조건:
    - prepare_corpus.py 실행 완료 (id_map.csv 존재)
    - .env 에 VLM_API_KEY, VLM_MODEL, TEXT_EMBED_MODEL 설정

사용법:
    cd PhoMate_FastAPI_BE
    python -m evaluation.prepare_captions

TODO:
    - VLM 캡션 생성 (현재 NotImplementedError)
    - 텍스트 임베딩 (현재 NotImplementedError)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

import evaluation.config as cfg


# ── VLM caption generation (TODO) ────────────────────────────────────────────

def generate_caption(image_path: Path) -> str:
    """
    Generates an English visual caption for a local image using a VLM.

    TODO: implement OpenAI vision API call.

    Args:
        image_path: path to local image file.

    Returns:
        English caption string describing the visual content.

    Raises:
        NotImplementedError: until VLM_API_KEY is configured.
    """
    # import base64, openai
    # client = openai.OpenAI(api_key=cfg.VLM_API_KEY, base_url=cfg.OPENAI_BASE_URL)
    # with open(image_path, "rb") as f:
    #     b64 = base64.b64encode(f.read()).decode()
    # ext = image_path.suffix.lstrip(".").lower()
    # mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    # resp = client.chat.completions.create(
    #     model=cfg.VLM_MODEL,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image_url",
    #                  "image_url": {"url": f"data:{mime};base64,{b64}"}},
    #                 {"type": "text",
    #                  "text": (
    #                      "Describe this photo in 1-2 English sentences focusing on "
    #                      "visual elements: scene, objects, mood, lighting, and composition. "
    #                      "Be concise and factual."
    #                  )},
    #             ],
    #         }
    #     ],
    #     max_tokens=128,
    # )
    # return resp.choices[0].message.content.strip()
    raise NotImplementedError(
        "VLM 캡션 생성 미구현. VLM_API_KEY 설정 후 generate_caption 코드를 주석 해제하세요."
    )


# ── Text embedding (TODO) ─────────────────────────────────────────────────────

def embed_caption(caption: str) -> list[float]:
    """
    Embeds a caption string using the configured text embedding model.

    TODO: implement OpenAI embeddings API call.

    Args:
        caption: English caption string.

    Returns:
        Embedding vector as a list of floats.

    Raises:
        NotImplementedError: until VLM_API_KEY and TEXT_EMBED_MODEL are set.
    """
    # import openai
    # client = openai.OpenAI(api_key=cfg.VLM_API_KEY, base_url=cfg.OPENAI_BASE_URL)
    # resp = client.embeddings.create(model=cfg.TEXT_EMBED_MODEL, input=caption)
    # return resp.data[0].embedding
    raise NotImplementedError(
        "텍스트 임베딩 미구현. VLM_API_KEY/TEXT_EMBED_MODEL 설정 후 embed_caption 코드를 주석 해제하세요."
    )


# ── Collection bootstrap ───────────────────────────────────────────────────────

def _ensure_caption_collection(client: QdrantClient, vector_size: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if cfg.EVAL_CAPTION_COLLECTION not in existing:
        client.create_collection(
            collection_name=cfg.EVAL_CAPTION_COLLECTION,
            vectors_config=qm.VectorParams(
                size=vector_size,
                distance=qm.Distance.COSINE,
            ),
        )
        print(f"컬렉션 생성: {cfg.EVAL_CAPTION_COLLECTION}")
    else:
        print(f"컬렉션 기존 사용: {cfg.EVAL_CAPTION_COLLECTION}")


# ── Main indexing logic ────────────────────────────────────────────────────────

def build_caption_index() -> None:
    """
    Reads id_map.csv, generates captions, embeds them, and upserts to
    the eval_caption Qdrant collection.

    Raises:
        FileNotFoundError: if id_map.csv does not exist.
        NotImplementedError: until VLM and text-embedding are implemented.
    """
    if not cfg.ID_MAP_PATH.exists():
        raise FileNotFoundError(
            f"id_map.csv 없음: {cfg.ID_MAP_PATH}\n"
            "먼저 prepare_corpus.py를 실행하세요."
        )

    with cfg.ID_MAP_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"캡션 생성 대상: {len(rows)}장")

    client = QdrantClient(url=cfg.QDRANT_URL)
    vector_size: int | None = None
    batch: list[qm.PointStruct] = []
    BATCH_SIZE = 16

    def _flush(batch: list[qm.PointStruct]) -> None:
        if batch:
            client.upsert(
                collection_name=cfg.EVAL_CAPTION_COLLECTION,
                points=batch,
                wait=True,
            )

    for row in tqdm(rows, desc="캡션·임베딩"):
        qdrant_id = int(row["qdrant_id"])
        doc_id = row["doc_id"]
        filepath = Path(row["filepath"])

        caption = generate_caption(filepath)  # TODO
        vec = embed_caption(caption)          # TODO

        if vector_size is None:
            vector_size = len(vec)
            _ensure_caption_collection(client, vector_size)

        batch.append(
            qm.PointStruct(
                id=qdrant_id,
                vector=vec,
                payload={"doc_id": doc_id, "caption": caption},
            )
        )

        if len(batch) >= BATCH_SIZE:
            _flush(batch)
            batch.clear()

    _flush(batch)
    count = client.count(collection_name=cfg.EVAL_CAPTION_COLLECTION).count
    print(f"\n완료: eval_caption 포인트 수 {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM 캡션 임베딩을 eval_caption 컬렉션에 인덱싱합니다.",
    )
    parser.parse_args()
    build_caption_index()


if __name__ == "__main__":
    main()
