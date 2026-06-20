"""
Centralised configuration for the evaluation harness.

All values are read from environment variables so that secrets are never
hard-coded.  Create a .env file in PhoMate_FastAPI_BE/ and run scripts with:

    python -m evaluation.<script>

or load the env manually before calling any module:

    export QDRANT_URL=http://localhost:6333
    export VLM_API_KEY=sk-...
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (PhoMate_FastAPI_BE/)
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Evaluation-only collections — never touch prod (post_vectors_siglip2)
EVAL_SIGLIP_COLLECTION  = os.getenv("EVAL_SIGLIP_COLLECTION",  "eval_siglip2")
EVAL_CAPTION_COLLECTION = os.getenv("EVAL_CAPTION_COLLECTION", "eval_caption")

# ── Search hyper-parameters ───────────────────────────────────────────────────
TOP_K               = int(os.getenv("EVAL_TOP_K", "10"))
RERANK_CANDIDATE_K  = int(os.getenv("EVAL_RERANK_CANDIDATE_K", "50"))

# ── VLM / text-embedding (B, B+C, A') ────────────────────────────────────────
VLM_API_KEY      = os.getenv("VLM_API_KEY", "")
VLM_MODEL        = os.getenv("VLM_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "text-embedding-3-small")

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR      = Path(os.getenv("EVAL_DATA_DIR", str(Path(__file__).parent / "data")))
ID_MAP_PATH   = DATA_DIR / "id_map.csv"
QUERIES_PATH  = DATA_DIR / "queries.csv"
QRELS_PATH    = DATA_DIR / "qrels.txt"
RUNS_DIR      = DATA_DIR / "runs"

# Photo concept vocabulary (populated by prepare_concepts.py)
CONCEPT_VOCAB_PATH    = DATA_DIR / "concept_vocab.csv"
PHOTO_CONCEPTS_PATH   = DATA_DIR / "photo_concepts.csv"

# ── Supported image extensions ─────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
