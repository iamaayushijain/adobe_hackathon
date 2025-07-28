"""
ocr_utils.py
-----------
Utility helpers for performing OCR on scanned PDF pages. Uses pdf2image to
rasterize pages and pytesseract for text recognition.

NOTE: OCR is only triggered for pages where no text is detected by the primary
parsers. This keeps performance high on digital PDFs.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pdfplumber
import io



def _pdfplumber_page_to_image(pdf_path: Path, page_num: int, dpi: int) -> Optional[Image.Image]:
    """Render a single PDF page to a PIL.Image via pdfplumber's to_image (needs Wand)."""
    try:
        with pdfplumber.open(str(pdf_path)) as doc:
            page = doc.pages[page_num - 1]
            return page.to_image(resolution=dpi).original
    except Exception:
        return None


class OCRProcessor:
    """Perform OCR on specified page numbers of a PDF and return text."""

    def __init__(self, dpi: int = 300):
        self.logger = logging.getLogger(__name__)
        self.dpi = dpi

    def ocr_pages(self, pdf_path: Path, pages: List[int]) -> List[Dict[str, Any]]:
        """Run OCR on the given pages list (1-indexed) and return results."""
        if not pages:
            return []

        self.logger.info(f"[OCR] Converting {len(pages)} page(s) of {pdf_path.name} to images…")
        # pdf2image expects 1-indexed page numbers.
        try:
            images: List[Image.Image] = convert_from_path(
                str(pdf_path), dpi=self.dpi, first_page=min(pages), last_page=max(pages), fmt="png"
            )
        except Exception as e:
            self.logger.warning(f"[OCR] pdf2image failed: {e}. Falling back to pdfplumber rendering.")
            images = []
            for p in pages:
                img = _pdfplumber_page_to_image(pdf_path, p, self.dpi)
                if img is not None:
                    images.append(img)
                else:
                    self.logger.error(f"[OCR] Could not render page {p} for OCR")

        # Mapping page number to OCR text
        results: List[Dict[str, Any]] = []
        for idx, image in enumerate(images):
            page_number = pages[idx] if idx < len(pages) else pages[0] + idx
            text = pytesseract.image_to_string(image, lang="eng")
            results.append({"page": page_number, "text": text})
        self.logger.info(f"[OCR] OCR complete for {len(results)} pages")
        return results 