from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, image_dim: int, text_dim: int):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "image": qm.VectorParams(size=image_dim, distance=qm.Distance.COSINE),
                "text": qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE),
            },
        )

    def upsert(self, post_id: int, member_id: int, image_vec: list[float], text_vec: list[float]):
        self.client.upsert(
            collection_name=self.collection,
            points=[
                qm.PointStruct(
                    id=post_id,
                    vector={"image": image_vec, "text": text_vec},
                    payload={"postId": post_id, "memberId": member_id},
                )
            ],
        )
