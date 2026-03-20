from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, vector_dim: int) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=vector_dim,
                distance=qm.Distance.COSINE,
            ),
        )

    def upsert(
        self,
        post_id: int,
        member_id: int,
        created_at_ms: int,
        vector: list[float],
    ) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=[
                qm.PointStruct(
                    id=post_id,
                    vector=vector,
                    payload={
                        "postId": post_id,
                        "memberId": member_id,
                        "createdAtMs": created_at_ms,
                    },
                )
            ],
        )

    def _build_filter(
        self,
        member_id: int | None,
        after_ms: int | None,
        before_ms: int | None,
    ) -> qm.Filter | None:
        must: list[qm.FieldCondition] = []

        if member_id is not None:
            must.append(
                qm.FieldCondition(
                    key="memberId",
                    match=qm.MatchValue(value=member_id),
                )
            )

        if after_ms is not None or before_ms is not None:
            must.append(
                qm.FieldCondition(
                    key="createdAtMs",
                    range=qm.Range(
                        gte=after_ms,
                        lte=before_ms,
                    ),
                )
            )

        if not must:
            return None

        return qm.Filter(must=must)

    def search(
        self,
        query_vec: list[float],
        top_k: int,
        member_id: int | None = None,
        after_ms: int | None = None,
        before_ms: int | None = None,
    ):
        flt = self._build_filter(member_id, after_ms, before_ms)

        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=flt,
        )

    def delete_by_post_id(self, post_id: int) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.PointIdsList(points=[post_id]),
        )