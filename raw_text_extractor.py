"""
raw_text_extractor.py
--------------------
Extracts lossless raw text from PDF files using pdfminer.six. This serves as a
fallback to guarantee we capture *all* textual content even if layout parsing
fails or misses data.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import pdfplumber
from itertools import groupby


class RawTextExtractor:
    """Extract all text strings from a PDF preserving page numbers."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _extract_with_pdfplumber(pdf_path: Path) -> List[Dict[str, Any]]:
        """Secondary extraction using pdfplumber to catch text pdfminer sometimes misses (tiny fonts, rotated)."""
        results: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(str(pdf_path)) as doc:
                for page_idx, page in enumerate(doc.pages):
                    # Extract raw text including duplicate chars; keep flow.
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                    results.append({"page": page_idx+1, "text": page_text})
        except Exception as e:
            logging.getLogger(__name__).warning(f"[RawTextExtractor] pdfplumber fallback failed: {e}")
        return results

    def extract(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Return a list with one entry per page containing raw text.

        Each list item is a dict: {"page": page_number, "text": "..."}
        """
        results: List[Dict[str, Any]] = []
        self.logger.info(f"[RawTextExtractor] Extracting text from {pdf_path.name}")

        try:
            for page_index, page_layout in enumerate(extract_pages(str(pdf_path))):
                page_text_parts: List[str] = []
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        page_text_parts.append(element.get_text())
                    elif isinstance(element, LTChar):
                        # Rare case where text containers not detected; capture chars
                        page_text_parts.append(element.get_text())
                page_text = "".join(page_text_parts)
                results.append({"page": page_index + 1, "text": page_text})
            self.logger.info(f"[RawTextExtractor] Extracted text for {len(results)} pages")

            # Fallback pass with pdfplumber for any pages whose text is empty
            missing_pages = [r["page"] for r in results if not r["text"].strip()]
            if missing_pages:
                plumber_pages = _extract_with_pdfplumber(pdf_path)
                page_map = {p["page"]: p["text"] for p in plumber_pages}
                for r in results:
                    if not r["text"].strip() and r["page"] in page_map:
                        r["text"] = page_map[r["page"]]
            # Final sanity: if *still* empty, concatenate char text via pdfplumber char boxes
            empties = [r for r in results if not r["text"].strip()]
            if empties:
                with pdfplumber.open(str(pdf_path)) as doc:
                    for r in empties:
                        page = doc.pages[r["page"]-1]
                        r["text"] = "".join(ch["text"] for ch in page.chars)

            return results
        except Exception as e:
            self.logger.error(f"[RawTextExtractor] Failed to extract text: {e}")
            raise 