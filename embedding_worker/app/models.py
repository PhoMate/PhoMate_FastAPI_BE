from __future__ import annotations

from pydantic import BaseModel, HttpUrl, Field


# ===== Embedding Job =====
class PostEmbeddingJob(BaseModel):
    postId: int
    memberId: int
    imageUrl: HttpUrl
    text: str
    createdAtMs: int = Field(..., description="Epoch milliseconds (e.g., System.currentTimeMillis())")


# ===== Search Requests =====
class TextSearchRequest(BaseModel):
    query: str
    topK: int = 20
    memberId: int | None = None
    createdAfterMs: int | None = None
    createdBeforeMs: int | None = None


class ImageSearchRequest(BaseModel):
    imageUrl: HttpUrl
    topK: int = 20
    memberId: int | None = None
    createdAfterMs: int | None = None
    createdBeforeMs: int | None = None


class HybridSearchRequest(BaseModel):
    query: str | None = None
    imageUrl: HttpUrl | None = None

    topK: int = 20

    candidateK: int = 50

    # 결과 결합 가중치 (RRF)
    weightText: float = 1.0
    weightImage: float = 1.0

    memberId: int | None = None
    createdAfterMs: int | None = None
    createdBeforeMs: int | None = None


# ===== Search Responses =====
class SearchHit(BaseModel):
    postId: int
    score: float
    memberId: int | None = None
    createdAtMs: int | None = None
    # 검색 추적
    source: str | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]
