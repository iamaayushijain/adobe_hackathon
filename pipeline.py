"""
pipeline.py
-----------
End-to-end document processing pipeline that:
1. Extracts hierarchical outline via `PDFOutlineParser`
2. Extracts full raw text via `RawTextExtractor`
3. Detects pages with no text and applies OCR fallback via `OCRProcessor`
4. Merges everything into a single rich JSON output structure ready for RAG.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from parser import PDFOutlineParser
from raw_text_extractor import RawTextExtractor
from ocr_utils import OCRProcessor
from table_extractor import TableExtractor


class DocumentPipeline:
    """High-level façade for processing a PDF through all extractors."""

    def __init__(self, ocr_enabled: bool = True):
        self.logger = logging.getLogger(__name__)
        self.outline_parser = PDFOutlineParser()
        self.raw_extractor = RawTextExtractor()
        self.ocr_processor = OCRProcessor()
        self.table_extractor = TableExtractor()
        self.ocr_enabled = ocr_enabled

    def process(self, pdf_path: Path) -> Dict[str, Any]:
        """Run the pipeline and return rich JSON output."""
        # 1. Outline extraction (structure & hierarchy)
        outline_data = self.outline_parser.extract_outline(pdf_path)

        # 2. Raw text extraction for completeness
        raw_text_pages = self.raw_extractor.extract(pdf_path)

        # 3. Table extraction
        tables = self.table_extractor.extract_tables(pdf_path)

        # 4. Identify missing pages text and OCR
        pages_missing_text: List[int] = [p["page"] for p in raw_text_pages if not p["text"].strip()]

        ocr_results: List[Dict[str, Any]] = []
        if self.ocr_enabled and pages_missing_text:
            self.logger.info(f"OCR needed for {len(pages_missing_text)} pages without text")
            ocr_results = self.ocr_processor.ocr_pages(pdf_path, pages_missing_text)
            # Merge OCR text back into raw_text_pages
            page_to_text = {r["page"]: r["text"] for r in ocr_results}
            for page_obj in raw_text_pages:
                if page_obj["page"] in page_to_text:
                    page_obj["text"] = page_to_text[page_obj["page"]]

        # Assemble final data structure
        return {
            "title": outline_data.get("title", "Untitled Document"),
            "outline": outline_data.get("outline", []),
            "raw_text": raw_text_pages,
            "tables": tables,
            "ocr": ocr_results,
        } 