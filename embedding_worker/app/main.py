import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from app.embedder import Embedder
from app.qdrant_store import QdrantStore

app = FastAPI(title="PhoMate Embedding Worker")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "post_vectors")

class PostEmbeddingJob(BaseModel):
    postId: int
    memberId: int
    imageUrl: HttpUrl
    text: str

embedder: Embedder | None = None
store: QdrantStore | None = None

@app.on_event("startup")
def startup():
    global embedder, store
    # 1) 모델 미리 로드 (여기서 다운로드/로딩 다 끝냄)
    embedder = Embedder()

    # 2) Qdrant 준비
    store = QdrantStore(url=QDRANT_URL, collection=QDRANT_COLLECTION)
    store.ensure_collection(image_dim=512, text_dim=1024)

@app.get("/health")
def health():
    return {"status": "ok", "qdrant": QDRANT_URL, "collection": QDRANT_COLLECTION}

@app.post("/jobs/post-embedding")
def post_embedding(job: PostEmbeddingJob):
    try:
        assert embedder is not None and store is not None

        image_vec = embedder.embed_image_url(str(job.imageUrl))
        text_vec = embedder.embed_text(job.text)

        store.upsert(
            post_id=job.postId,
            member_id=job.memberId,
            image_vec=image_vec,
            text_vec=text_vec,
        )

        return {"status": "OK", "postId": job.postId}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
