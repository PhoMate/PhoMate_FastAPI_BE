"""
Search method registry for the evaluation harness.

Each method implements the common search(query) -> list[doc_id] interface.
The registry allows run_search.py to iterate all registered methods without
knowing their internals.

Usage:
    from evaluation.methods import get_method, list_methods

    method = get_method("A")
    doc_ids = method.search("카페에서 커피 마시는 사람")
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

# Allow importing embedding_worker from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

import evaluation.config as cfg
from evaluation.scoring import rerank_by_concept_overlap, compute_idf

if TYPE_CHECKING:
    pass


# ── Common interface ──────────────────────────────────────────────────────────

class SearchMethod(ABC):
    """Base class for all retrieval methods."""

    @abstractmethod
    def search(self, query: str, top_k: int = cfg.TOP_K) -> list[str]:
        """
        Searches for relevant photos given a Korean query.

        Args:
            query:  Korean natural-language query string.
            top_k:  maximum number of results to return.

        Returns:
            List of doc_ids in descending relevance order, length ≤ top_k.
        """
        ...


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, SearchMethod] = {}


def register(name: str, method: SearchMethod) -> None:
    """Adds a method to the global registry under the given name."""
    _REGISTRY[name] = method


def get_method(name: str) -> SearchMethod:
    """
    Retrieves a registered method by name.

    Raises:
        KeyError: if name is not in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_methods() -> list[str]:
    """Returns all registered method names in sorted order."""
    return sorted(_REGISTRY)


# ── Shared helpers ────────────────────────────────────────────────────────────

_embedder = None  # lazy singleton — SigLIP model is expensive to load


def _get_embedder():
    """Returns the shared Embedder instance, loading it on first call."""
    global _embedder
    if _embedder is None:
        from embedding_worker.app.embedder import Embedder
        _embedder = Embedder()
    return _embedder


def _qdrant_search(collection: str, vector: list[float], top_k: int) -> list[str]:
    """
    Runs a cosine-similarity vector search against an eval Qdrant collection.

    Args:
        collection: name of the Qdrant collection to query.
        vector:     query embedding (must match the collection's vector size).
        top_k:      maximum number of hits to retrieve.

    Returns:
        List of doc_ids in descending score order.
    """
    client = QdrantClient(url=cfg.QDRANT_URL)
    hits = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return [str(h.payload.get("doc_id", h.id)) for h in hits]


# ── Method A — SigLIP, Korean query as-is ─────────────────────────────────────

class MethodA(SearchMethod):
    """
    SigLIP image embeddings searched with the raw Korean query.
    Fully implemented; requires eval_siglip2 to be indexed first.
    """

    def search(self, query: str, top_k: int = cfg.TOP_K) -> list[str]:
        vec = _get_embedder().embed_text(query)
        return _qdrant_search(cfg.EVAL_SIGLIP_COLLECTION, vec, top_k)


# ── Method A' — SigLIP, English-translated query ──────────────────────────────

class MethodAPrime(SearchMethod):
    """
    SigLIP image embeddings searched with an English query.

    run_search.py passes query_en when available (COCO 영어 캡션 등).
    query_en 이 없는 경우에는 _translate_ko_to_en 이 필요하지만 현재 TODO.
    """

    def search(self, query: str, top_k: int = cfg.TOP_K) -> list[str]:
        # query 는 run_search.py 가 language 선택 후 전달한 텍스트
        # (query_en 있으면 영어, 없으면 Korean — 후자는 VLM_API_KEY 필요)
        vec = _get_embedder().embed_text(query)
        return _qdrant_search(cfg.EVAL_SIGLIP_COLLECTION, vec, top_k)


def _translate_ko_to_en(query: str) -> str:
    """
    Translates a Korean query to English via the OpenAI chat API.

    TODO: uncomment the implementation below and set VLM_API_KEY.

    Args:
        query: Korean query string.

    Returns:
        English translation suitable for SigLIP text embedding.

    Raises:
        NotImplementedError: until VLM_API_KEY is configured and the
            implementation is uncommented.
    """
    # import openai
    # client = openai.OpenAI(api_key=cfg.VLM_API_KEY, base_url=cfg.OPENAI_BASE_URL)
    # resp = client.chat.completions.create(
    #     model=cfg.VLM_MODEL,
    #     messages=[
    #         {"role": "system",
    #          "content": (
    #              "Translate the following Korean photo-search query into a short "
    #              "English visual description of what might be seen in the photo. "
    #              "Return only the translated text, no explanation."
    #          )},
    #         {"role": "user", "content": query},
    #     ],
    #     max_tokens=64,
    # )
    # return resp.choices[0].message.content.strip()
    raise NotImplementedError(
        "A' translation not implemented. Set VLM_API_KEY and uncomment _translate_ko_to_en."
    )


# ── Method B — Caption text embeddings ───────────────────────────────────────

class MethodB(SearchMethod):
    """
    Text-to-text search using VLM-generated captions embedded with a text model.
    Requires eval_caption to be indexed first (prepare_captions.py).
    VLM and text-embedder configuration is TODO.
    """

    def search(self, query: str, top_k: int = cfg.TOP_K) -> list[str]:
        # TODO: embed query with text embedder and search eval_caption
        # query_vec = _embed_text_openai(query)
        # return _qdrant_search(cfg.EVAL_CAPTION_COLLECTION, query_vec, top_k)
        raise NotImplementedError(
            "Method B not implemented. Set VLM_API_KEY and TEXT_EMBED_MODEL, "
            "then uncomment _embed_text_openai and run prepare_captions.py."
        )


def _embed_text_openai(text: str) -> list[float]:
    """
    Embeds text using the OpenAI embeddings API.

    TODO: uncomment and set VLM_API_KEY / TEXT_EMBED_MODEL.

    Args:
        text: input string to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    # import openai
    # client = openai.OpenAI(api_key=cfg.VLM_API_KEY, base_url=cfg.OPENAI_BASE_URL)
    # resp = client.embeddings.create(model=cfg.TEXT_EMBED_MODEL, input=text)
    # return resp.data[0].embedding
    raise NotImplementedError("_embed_text_openai: set VLM_API_KEY and uncomment.")


# ── Method B+C — B retrieval + concept overlap re-ranking ─────────────────────

class MethodBC(SearchMethod):
    """
    Two-stage retrieval: B caption search for candidates, then IDF-weighted
    concept-overlap re-ranking (C).
    Requires prepare_captions.py and prepare_concepts.py to be run first.
    Concept extraction on the query side is TODO.
    """

    def search(self, query: str, top_k: int = cfg.TOP_K) -> list[str]:
        # Stage 1 — B candidate retrieval
        candidates = MethodB().search(query, top_k=cfg.RERANK_CANDIDATE_K)

        # TODO: extract query concepts
        # query_concepts = _extract_query_concepts(query)
        raise NotImplementedError(
            "B+C not implemented. Implement _extract_query_concepts and load "
            "photo_concepts from prepare_concepts.py output."
        )

        # Stage 2 — concept re-ranking (skeleton; uncomment after TODO above)
        # import pandas as pd
        # df = pd.read_csv(cfg.PHOTO_CONCEPTS_PATH)
        # photo_concept_map = (
        #     df.groupby("doc_id")["concept_id"].apply(list).to_dict()
        # )
        # all_concepts = {doc_id: concepts for doc_id, concepts in photo_concept_map.items()}
        # idf = compute_idf(all_concepts)
        # return rerank_by_concept_overlap(
        #     candidates, query_concepts, photo_concept_map, idf, top_k
        # )


def _extract_query_concepts(query: str) -> list[str]:
    """
    Extracts visual concepts from a Korean query string.

    TODO: implement via VLM prompt or keyword mapping.

    Args:
        query: Korean query string.

    Returns:
        List of normalised concept IDs matching the concept vocabulary.
    """
    # Possible implementation:
    # - prompt VLM to list mood/place/object/style concepts visible in the described scene
    # - normalise against concept_vocab.csv
    raise NotImplementedError("_extract_query_concepts: implement concept extraction for queries.")


# ── Register all methods ──────────────────────────────────────────────────────

register("A",       MethodA())
register("A_prime", MethodAPrime())
register("B",       MethodB())
register("BC",      MethodBC())
