from __future__ import annotations

import requests
from PIL import Image

import torch
from transformers import AutoModel, AutoProcessor


class Embedder:
    """
    SigLIP 2 공통 임베딩.
    - image -> shared vector
    - text  -> shared vector
    """

    def __init__(self):
        self.device = "cpu"
        self.model_name = "google/siglip2-base-patch16-224"

        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()

        self._vector_dim: int | None = None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def _extract_tensor(self, outputs) -> torch.Tensor:
        """
        transformers 출력이 Tensor가 아닐 수 있으므로
        pooled_output / text_embeds / image_embeds / last_hidden_state 순으로 안전하게 꺼낸다.
        """
        if isinstance(outputs, torch.Tensor):
            return outputs

        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            return outputs.text_embeds

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            # [batch, seq, hidden] -> CLS/token 0 사용
            return outputs.last_hidden_state[:, 0, :]

        raise TypeError(f"Unsupported model output type: {type(outputs)}")

    def embed_image_url(self, image_url: str) -> list[float]:
        response = requests.get(image_url, stream=True, timeout=20)
        response.raise_for_status()

        image = Image.open(response.raw).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # 1순위: 전용 feature method
            if hasattr(self.model, "get_image_features"):
                outputs = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)

        feats = self._extract_tensor(outputs)
        feats = self._normalize(feats)
        return feats[0].cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        query = (text or "").strip()
        if not query:
            raise ValueError("text query is empty")

        inputs = self.processor(text=[query], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # 1순위: 전용 feature method
            if hasattr(self.model, "get_text_features"):
                outputs = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)

        feats = self._extract_tensor(outputs)
        feats = self._normalize(feats)
        return feats[0].cpu().tolist()

    def vector_dim(self) -> int:
        if self._vector_dim is None:
            self._vector_dim = len(self.embed_text("test"))
        return self._vector_dim