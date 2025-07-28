"""
app/ranker.py
-------------
Ranks sections (heading + text) by relevance to a task/persona vector using cosine similarity.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from app.embedder import embed_texts

def rank_sections(sections: List[Dict[str, str]], task_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
    """
    Args:
        sections: list of dicts with at least `text` key.
        task_vector: numpy array of shape (dim,)
        top_k: return this many top matches
    Returns:
        list of (section_dict, score) sorted by score desc
    """
    if not sections:
        return []

    texts = [s["text"] for s in sections]
    vecs = embed_texts(texts)
    scores = cosine_similarity(vecs, task_vector.reshape(1, -1)).flatten()

    ranked = sorted(zip(sections, scores), key=lambda x: x[1], reverse=True)[: top_k]
    return ranked 