import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    CPU 기준 MVP.
    이미지: openai/clip-vit-base-patch32 -> 512 dim
    텍스트: BAAI/bge-m3 -> 보통 1024 dim
    """
    def __init__(self):
        # device 설정 (추후 GPU면 "cuda")
        self.device = "cpu"

        # CLIP
        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model.eval()

        # bge-m3
        self.text_model_name = "BAAI/bge-m3"
        self.text_model = SentenceTransformer(self.text_model_name, device=self.device)

    def embed_image_url(self, image_url: str) -> list[float]:
        r = requests.get(image_url, stream=True, timeout=20)
        r.raise_for_status()
        img = Image.open(r.raw).convert("RGB")

        inputs = self.clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = self.clip_model.get_image_features(**inputs)

        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats[0].cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        vec = self.text_model.encode(text, normalize_embeddings=True)
        return vec.tolist()
