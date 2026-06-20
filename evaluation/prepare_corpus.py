"""
prepare_corpus.py — SigLIP 이미지 임베딩 → eval_siglip2 인덱싱

로컬 이미지 데이터셋을 Qdrant eval 전용 컬렉션에 인덱싱합니다.
prod 컬렉션(post_vectors_siglip2)은 절대 건드리지 않습니다.

사용법:
    cd PhoMate_FastAPI_BE
    python -m evaluation.prepare_corpus --dataset-dir /path/to/images

완료 후 evaluation/data/id_map.csv 가 생성됩니다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

# Allow importing embedding_worker from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm

import evaluation.config as cfg


# ── Local image embedding helper ─────────────────────────────────────────────
# embedding_worker/app/embedder.py 수정 금지이므로 PIL 로딩만 이쪽에서 처리하고
# 모델 forward는 embedder 내부 processor/model을 직접 호출합니다.

def _embed_local_image(embedder, path: Path) -> list[float]:
    """
    Embeds a local image file using the shared Embedder instance.

    Replicates embed_image_url() but reads from disk instead of HTTP.
    Accesses embedder.processor / embedder.model / embedder.device which
    are set during Embedder.__init__() — no modification to embedder.py needed.

    Args:
        embedder: an initialised Embedder instance.
        path:     path to a local image file.

    Returns:
        L2-normalised SigLIP image embedding as a list of floats.
    """
    image = Image.open(path).convert("RGB")
    inputs = embedder.processor(images=image, return_tensors="pt")
    inputs = {k: v.to(embedder.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embedder.model.get_image_features(**inputs)
    # outputs may be a tensor or a dataclass; normalise the same way embedder does
    if hasattr(outputs, "image_embeds"):
        feats = outputs.image_embeds
    else:
        feats = outputs
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().tolist()


# ── Qdrant collection bootstrap ──────────────────────────────────────────────

def _ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    """
    Creates the Qdrant collection if it does not exist.
    Uses cosine distance to match the SigLIP embedding space.
    """
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=vector_size,
                distance=qm.Distance.COSINE,
            ),
        )
        print(f"컬렉션 생성: {name} (dim={vector_size})")
    else:
        print(f"컬렉션 기존 사용: {name}")


# ── Corpus indexing ───────────────────────────────────────────────────────────

def index_dataset(dataset_dir: Path, limit: int | None = None) -> None:
    """
    Scans dataset_dir for image files, embeds each with SigLIP, and upserts
    them into the eval_siglip2 Qdrant collection.

    Also writes evaluation/data/id_map.csv with columns:
        qdrant_id (int), doc_id (str), filepath (str)

    Args:
        dataset_dir: directory containing image files (searched recursively).
        limit:       if set, processes at most this many images (for dry-runs).
    """
    # ── collect image paths
    image_paths: list[Path] = sorted(
        p for p in dataset_dir.rglob("*") if p.suffix.lower() in cfg.IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {dataset_dir}")

    if limit is not None:
        image_paths = image_paths[:limit]

    print(f"이미지 {len(image_paths)}장 처리 시작 (데이터셋: {dataset_dir})")

    # ── lazy embedder init (SigLIP model load)
    from embedding_worker.app.embedder import Embedder
    embedder = Embedder()

    # ── probe vector size
    probe_vec = _embed_local_image(embedder, image_paths[0])
    vector_size = len(probe_vec)

    # ── Qdrant client & collection
    client = QdrantClient(url=cfg.QDRANT_URL)
    _ensure_collection(client, cfg.EVAL_SIGLIP_COLLECTION, vector_size)

    # ── index loop
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    id_map_rows: list[dict] = []
    batch: list[qm.PointStruct] = []
    BATCH_SIZE = 32

    def _flush(batch: list[qm.PointStruct]) -> None:
        if batch:
            client.upsert(
                collection_name=cfg.EVAL_SIGLIP_COLLECTION,
                points=batch,
                wait=True,
            )

    for qdrant_id, img_path in enumerate(tqdm(image_paths, desc="임베딩"), start=1):
        rel_path = img_path.relative_to(dataset_dir)
        doc_id = str(rel_path).replace("/", "__").replace("\\", "__")
        doc_id = doc_id.rsplit(".", 1)[0]  # strip extension

        try:
            vec = _embed_local_image(embedder, img_path)
        except Exception as exc:
            print(f"  [SKIP] {img_path.name}: {exc}")
            continue

        batch.append(
            qm.PointStruct(
                id=qdrant_id,
                vector=vec,
                payload={"doc_id": doc_id, "filepath": str(img_path)},
            )
        )
        id_map_rows.append(
            {"qdrant_id": qdrant_id, "doc_id": doc_id, "filepath": str(img_path)}
        )

        if len(batch) >= BATCH_SIZE:
            _flush(batch)
            batch.clear()

    _flush(batch)

    # ── write id_map.csv
    with cfg.ID_MAP_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qdrant_id", "doc_id", "filepath"])
        writer.writeheader()
        writer.writerows(id_map_rows)

    # ── verify
    count = client.count(collection_name=cfg.EVAL_SIGLIP_COLLECTION).count
    print(f"\n완료: id_map.csv {len(id_map_rows)}행, Qdrant 포인트 수 {count}")
    if count < len(id_map_rows):
        print(f"  [경고] Qdrant 포인트({count}) < id_map 행 수({len(id_map_rows)}): 일부 upsert 실패 가능")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="로컬 이미지 데이터셋을 eval_siglip2 컬렉션에 인덱싱합니다.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="이미지 파일이 있는 최상위 디렉토리",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 최대 이미지 수 (생략 시 전체)",
    )
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        print(f"오류: {args.dataset_dir} 는 디렉토리가 아닙니다.", file=sys.stderr)
        sys.exit(1)

    index_dataset(args.dataset_dir, limit=args.limit)


if __name__ == "__main__":
    main()
