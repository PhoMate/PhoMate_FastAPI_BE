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
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "post_vectors_siglip2")

embedder: Embedder | None = None
store: QdrantStore | None = None


@app.on_event("startup")
def startup():
    global embedder, store
    embedder = Embedder()
    store = QdrantStore(url=QDRANT_URL, collection=QDRANT_COLLECTION)
    store.ensure_collection(vector_dim=embedder.vector_dim())


@app.get("/health")
def health():
    return {"status": "ok", "qdrant": QDRANT_URL, "collection": QDRANT_COLLECTION}


@app.post("/jobs/post-embedding")
def post_embedding(job: PostEmbeddingJob):
    try:
        assert embedder is not None and store is not None

        vector = embedder.embed_image_url(str(job.imageUrl))

        store.upsert(
            post_id=job.postId,
            member_id=job.memberId,
            created_at_ms=job.createdAtMs,
            vector=vector,
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


@app.post("/search/text", response_model=SearchResponse)
def search_text(req: TextSearchRequest):
    try:
        assert embedder is not None and store is not None

        qvec = embedder.embed_text(req.query)

        points = store.search(
            query_vec=qvec,
            top_k=req.topK,
            member_id=req.memberId,
            after_ms=req.createdAfterMs,
            before_ms=req.createdBeforeMs,
        )

        hits = [_payload_to_hit(p, p.score, source="text") for p in points]
        return SearchResponse(hits=hits)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/search/image", response_model=SearchResponse)
def search_image(req: ImageSearchRequest):
    try:
        assert embedder is not None and store is not None

        ivec = embedder.embed_image_url(str(req.imageUrl))

        points = store.search(
            query_vec=ivec,
            top_k=req.topK,
            member_id=req.memberId,
            after_ms=req.createdAfterMs,
            before_ms=req.createdBeforeMs,
        )

        hits = [_payload_to_hit(p, p.score, source="image") for p in points]
        return SearchResponse(hits=hits)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/search/hybrid", response_model=SearchResponse)
def search_hybrid(req: HybridSearchRequest):
    try:
        assert embedder is not None and store is not None

        if not req.query and not req.imageUrl:
            raise HTTPException(
                status_code=400,
                detail="Either 'query' or 'imageUrl' must be provided.",
            )

        candidate_k = max(req.candidateK, req.topK)

        text_points = []
        image_points = []

        if req.query:
            qvec = embedder.embed_text(req.query)
            text_points = store.search(
                query_vec=qvec,
                top_k=candidate_k,
                member_id=req.memberId,
                after_ms=req.createdAfterMs,
                before_ms=req.createdBeforeMs,
            )

        if req.imageUrl:
            ivec = embedder.embed_image_url(str(req.imageUrl))
            image_points = store.search(
                query_vec=ivec,
                top_k=candidate_k,
                member_id=req.memberId,
                after_ms=req.createdAfterMs,
                before_ms=req.createdBeforeMs,
            )

        rrf_k = 60.0
        merged: dict[int, dict] = {}

        for rank, p in enumerate(text_points, start=1):
            hit = _payload_to_hit(p, score=p.score, source=None)
            post_id = hit.postId
            merged.setdefault(post_id, {"hit": hit, "rrf": 0.0, "src": set()})
            merged[post_id]["rrf"] += req.weightText / (rrf_k + rank)
            merged[post_id]["src"].add("text")

        for rank, p in enumerate(image_points, start=1):
            hit = _payload_to_hit(p, score=p.score, source=None)
            post_id = hit.postId
            merged.setdefault(post_id, {"hit": hit, "rrf": 0.0, "src": set()})
            merged[post_id]["rrf"] += req.weightImage / (rrf_k + rank)
            merged[post_id]["src"].add("image")

        items = sorted(
            merged.items(),
            key=lambda kv: kv[1]["rrf"],
            reverse=True,
        )[: req.topK]

        final_hits: list[SearchHit] = []
        for _, value in items:
            hit: SearchHit = value["hit"]
            hit.score = float(value["rrf"])
            hit.source = "+".join(sorted(value["src"]))
            final_hits.append(hit)

        return SearchResponse(hits=final_hits)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.delete("/vectors/posts/{post_id}")
def delete_post_vector(post_id: int):
    try:
        assert store is not None
        store.delete_by_post_id(post_id)
        return {"status": "OK", "deletedPostId": post_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")