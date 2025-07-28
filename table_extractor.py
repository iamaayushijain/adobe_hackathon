"""
table_extractor.py
------------------
Extract table structures from PDFs using pdfplumber's built-in table finder.
Returns each table as a list-of-lists (rows) preserving cell text order.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
import io

class TableExtractor:
    """Extract tables page-by-page."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(str(pdf_path)) as doc:
                for page_idx, page in enumerate(doc.pages):
                    page_tables = page.find_tables(table_settings={"vertical_strategy":"lines","horizontal_strategy":"lines"})
                    for t_idx, table in enumerate(page_tables):
                        data = table.extract()
                        tables.append({"page": page_idx+1, "index": t_idx, "data": data})
        except Exception as e:
            self.logger.error(f"[TableExtractor] Failed on {pdf_path.name}: {e}")
        self.logger.info(f"[TableExtractor] Found {len(tables)} tables")
        return tables 