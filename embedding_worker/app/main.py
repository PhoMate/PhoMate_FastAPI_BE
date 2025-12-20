from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException

from app.embedder import Embedder
from app.qdrant_store import QdrantStore
from app.models import (
    PostEmbeddingJob,
    TextSearchRequest,
    ImageSearchRequest,
    HybridSearchRequest,
    SearchHit,
    SearchResponse,
)

app = FastAPI(title="PhoMate Embedding Worker")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "post_vectors")

embedder: Embedder | None = None
store: QdrantStore | None = None


@app.on_event("startup")
def startup():
    global embedder, store
    embedder = Embedder()
    store = QdrantStore(url=QDRANT_URL, collection=QDRANT_COLLECTION)
    store.ensure_collection(image_dim=512, text_dim=1024)


@app.get("/health")
def health():
    return {"status": "ok", "qdrant": QDRANT_URL, "collection": QDRANT_COLLECTION}


# ===== 저장(임베딩 + upsert) =====
@app.post("/jobs/post-embedding")
def post_embedding(job: PostEmbeddingJob):
    try:
        assert embedder is not None and store is not None

        image_vec = embedder.embed_image_url(str(job.imageUrl))
        text_vec = embedder.embed_text(job.text)

        store.upsert(
            post_id=job.postId,
            member_id=job.memberId,
            created_at_ms=job.createdAtMs,
            image_vec=image_vec,
            text_vec=text_vec,
        )
        return {"status": "OK", "postId": job.postId}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


def _payload_to_hit(p, score: float, source: str | None = None) -> SearchHit:
    payload = p.payload or {}
    post_id = int(payload.get("postId", p.id))
    return SearchHit(
        postId=post_id,
        score=float(score),
        memberId=payload.get("memberId"),
        createdAtMs=payload.get("createdAtMs"),
        source=source,
    )


# ===== 텍스트 검색 =====
@app.post("/search/text", response_model=SearchResponse)
def search_text(req: TextSearchRequest):
    try:
        assert embedder is not None and store is not None
        qvec = embedder.embed_text(req.query)

        points = store.search_text(
            text_vec=qvec,
            top_k=req.topK,
            member_id=req.memberId,
            after_ms=req.createdAfterMs,
            before_ms=req.createdBeforeMs,
        )

        hits = [_payload_to_hit(p, p.score, source="text") for p in points]
        return SearchResponse(hits=hits)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# ===== 이미지 검색 =====
@app.post("/search/image", response_model=SearchResponse)
def search_image(req: ImageSearchRequest):
    try:
        assert embedder is not None and store is not None
        ivec = embedder.embed_image_url(str(req.imageUrl))

        points = store.search_image(
            image_vec=ivec,
            top_k=req.topK,
            member_id=req.memberId,
            after_ms=req.createdAfterMs,
            before_ms=req.createdBeforeMs,
        )

        hits = [_payload_to_hit(p, p.score, source="image") for p in points]
        return SearchResponse(hits=hits)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# ===== 하이브리드(텍스트 + 이미지) 검색 =====
@app.post("/search/hybrid", response_model=SearchResponse)
def search_hybrid(req: HybridSearchRequest):
    try:
        assert embedder is not None and store is not None

        if not req.query and not req.imageUrl:
            raise HTTPException(status_code=400, detail="Either 'query' or 'imageUrl' must be provided.")

        candidate_k = max(req.candidateK, req.topK)

        text_points = []
        image_points = []

        if req.query:
            qvec = embedder.embed_text(req.query)
            text_points = store.search_text(
                text_vec=qvec,
                top_k=candidate_k,
                member_id=req.memberId,
                after_ms=req.createdAfterMs,
                before_ms=req.createdBeforeMs,
            )

        if req.imageUrl:
            ivec = embedder.embed_image_url(str(req.imageUrl))
            image_points = store.search_image(
                image_vec=ivec,
                top_k=candidate_k,
                member_id=req.memberId,
                after_ms=req.createdAfterMs,
                before_ms=req.createdBeforeMs,
            )

        # ---- RRF(Reciprocal Rank Fusion) ----
        # score = w_text/(k + rank_text) + w_img/(k + rank_img)
        RRF_K = 60.0

        # postId -> (payload, score)
        merged: dict[int, dict] = {}

        # text ranks
        for rank, p in enumerate(text_points, start=1):
            hit = _payload_to_hit(p, score=p.score, source=None)
            post_id = hit.postId
            merged.setdefault(post_id, {"hit": hit, "rrf": 0.0, "src": set()})
            merged[post_id]["rrf"] += req.weightText / (RRF_K + rank)
            merged[post_id]["src"].add("text")

        # image ranks
        for rank, p in enumerate(image_points, start=1):
            hit = _payload_to_hit(p, score=p.score, source=None)
            post_id = hit.postId
            merged.setdefault(post_id, {"hit": hit, "rrf": 0.0, "src": set()})
            merged[post_id]["rrf"] += req.weightImage / (RRF_K + rank)
            merged[post_id]["src"].add("image")

        # 최종 정렬: rrf 내림차순
        items = sorted(merged.items(), key=lambda kv: kv[1]["rrf"], reverse=True)[: req.topK]

        final_hits: list[SearchHit] = []
        for post_id, v in items:
            h: SearchHit = v["hit"]
            h.score = float(v["rrf"])
            h.source = "+".join(sorted(v["src"]))
            final_hits.append(h)

        return SearchResponse(hits=final_hits)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
