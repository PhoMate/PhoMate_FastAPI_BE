from __future__ import annotations

import requests
from PIL import Image

import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class Embedder:
    """
    SigLIP 2 공통 임베딩.
    """

    def __init__(self):
        self.device = "cpu"

        self.model_name = "google/siglip2-base-patch16-224"

        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    def embed_image_url(self, image_url: str) -> list[float]:
        r = requests.get(image_url, stream=True, timeout=20)
        r.raise_for_status()

        img = Image.open(r.raw).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)

        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats[0].cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        query = (text or "").strip()
        if not query:
            raise ValueError("text query is empty")

        inputs = self.tokenizer(
            [query],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats[0].cpu().tolist()

    def vector_dim(self) -> int:
        return len(self.embed_text("test"))