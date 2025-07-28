"""
app/subsection_selector.py
-------------------------
Selects the most relevant sentences/paragraphs inside a larger section text.
"""

import re
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.embedder import embed_texts


def _split_into_sentences(text: str) -> List[str]:
    # naive sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def select_subsections(section_text: str, task_vector: np.ndarray, top_k: int = 3) -> str:
    sentences = _split_into_sentences(section_text)
    if not sentences:
        return section_text

    vecs = embed_texts(sentences)
    scores = cosine_similarity(vecs, task_vector.reshape(1, -1)).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    top_sentences = [sentences[i] for i in top_idx]
    return " " .join(top_sentences) 