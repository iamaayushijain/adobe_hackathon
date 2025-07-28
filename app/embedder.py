"""
app/embedder.py
---------------
Singleton wrapper around an offline SentenceTransformer model (all-MiniLM-L6-v2 ~80 MB).
Loads lazily on first call to avoid start-up penalty.
"""

from pathlib import Path
from functools import lru_cache
from typing import List
import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the transformer model once."""
    from sentence_transformers import SentenceTransformer  # local import to keep import time low

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts → ndarray shape (n, dim)."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True)


def embed(text: str) -> np.ndarray:
    """Embed a single string → 1-D vector."""
    return embed_texts([text])[0] 