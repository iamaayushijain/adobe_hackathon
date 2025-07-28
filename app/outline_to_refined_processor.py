"""
app/outline_to_refined_processor.py
----------------------------------
Generates ranked sections and refined paragraph snippets (subsection_analysis)
from the outline-only JSON produced in Challenge-1B.

Algorithm:
1. Embed persona+task once (MiniLM).
2. For each document outline entry:
   – Rank headings by cosine similarity text→task.
   – Take top-K.
3. For each top heading: take its page’s raw_text and select the most
   relevant N sentences using sentence similarity.
Output matches Adobe sample schema.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.embedder import embed, embed_texts
from app.subsection_selector import select_subsections

logger = logging.getLogger(__name__)


class OutlineToRefinedProcessor:
    TOP_SECTIONS = 5
    TOP_SENTENCES = 3

    def generate_refined_output(self, outline_json_path: str | Path, out_path: str | Path) -> Dict[str, Any]:
        outline_json_path = Path(outline_json_path)
        data = json.loads(outline_json_path.read_text())

        persona = data.get("persona", "")
        task    = data.get("job_to_be_done", "")
        task_vec = embed(f"{persona} {task}")

        extracted_sections: List[Dict[str, Any]] = []
        subsection_analysis: List[Dict[str, Any]] = []

        for doc_entry in data.get("outlines", []):
            docname = doc_entry.get("document")
            outline = doc_entry.get("outline", [])
            raw_pages = {p["page"]: p["text"] for p in doc_entry.get("raw_text", [])}
            if not outline:
                continue

            # Rank headings
            headings_text = [h["text"] for h in outline]
            vecs = embed_texts(headings_text)
            scores = cosine_similarity(vecs, task_vec.reshape(1, -1)).flatten()
            top_idx = scores.argsort()[-self.TOP_SECTIONS:][::-1]

            for rank, idx in enumerate(top_idx, 1):
                h = outline[idx]
                extracted_sections.append({
                    "document": docname,
                    "section_title": h["text"],
                    "importance_rank": rank,
                    "page_number": h["page"]
                })

                page_text = raw_pages.get(h["page"], "")
                refined = select_subsections(page_text, task_vec, self.TOP_SENTENCES) if page_text else ""
                subsection_analysis.append({
                    "document": docname,
                    "refined_text": refined if refined else page_text[:400],
                    "page_number": h["page"]
                })

        final = {
            "metadata": {
                "input_documents": data.get("input_documents", []),
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": data.get("processing_timestamp")
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis,
        }

        out_path = Path(out_path)
        out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False))
        logger.info(f"Refined output saved to {out_path}")
        return final 