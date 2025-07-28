"""
Multi-Parser PDF Text Extractor
Combines multiple PDF parsing libraries to ensure maximum text extraction
from complex, weird, and difficult PDFs.

Libraries used:
- pdfplumber (layout-aware)
- PyMuPDF (fitz) - fast and accurate
- pdfminer.six - detailed text extraction
- camelot - table extraction

Installation:
pip install pdfplumber PyMuPDF pdfminer.six camelot-py[cv]
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import difflib
import os 
import json
import io


# Core PDF libraries
import pdfplumber
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.layout import LAParams

# Table extraction libraries
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.warning("Camelot not available - table extraction will be limited")

# OCR libraries for scanned PDFs
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR libraries not available - scanned PDF extraction will be limited")


@dataclass
class ExtractedText:
    """Container for extracted text with metadata."""
    text: str
    source: str
    page_num: int
    confidence: float = 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None
    font_info: Optional[Dict] = None


@dataclass
class ParsingResult:
    """Result from a single parser."""
    parser_name: str
    texts: List[ExtractedText]
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0


class PDFOutlineParser:
    """
    Advanced PDF text extractor using multiple parsing libraries.
    Combines results from different parsers to ensure maximum text recovery.
    """

    def __init__(self, enable_ocr: bool = True, enable_tables: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.enable_tables = enable_tables
        
        # Parser configurations - only 4 parsers
        self.parsers = {
            'pdfplumber': self._extract_with_pdfplumber,
            'pymupdf': self._extract_with_pymupdf,
            'pdfminer': self._extract_with_pdfminer,
        }
        
        if self.enable_tables and CAMELOT_AVAILABLE:
            self.parsers['camelot'] = self._extract_with_camelot
        
        if self.enable_ocr:
            self.parsers['ocr'] = self._extract_with_ocr

    def save_to_json(self, data, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


    def extract_outline(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract structured outline with headings using multi-parser approach.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with title, outline (H1,H2,H3), raw text, and tables
        """
        self.logger.info(f"Starting multi-parser extraction for: {pdf_path.name}")
        
        results = {}
        all_texts = []
        
        # Run all parsers
        for parser_name, parser_func in self.parsers.items():
            self.logger.info(f"Running parser: {parser_name}")
            try:
                import time
                start_time = time.time()
                result = parser_func(pdf_path)
                execution_time = time.time() - start_time
                
                result.execution_time = execution_time
                results[parser_name] = result
                
                if result.success:
                    all_texts.extend(result.texts)
                    self.logger.info(f"{parser_name}: extracted {len(result.texts)} text blocks in {execution_time:.2f}s")
                else:
                    self.logger.warning(f"{parser_name}: failed - {result.error}")
                    
            except Exception as e:
                self.logger.error(f"Parser {parser_name} crashed: {str(e)}")
                results[parser_name] = ParsingResult(
                    parser_name=parser_name,
                    texts=[],
                    success=False,
                    error=str(e)
                )
        
        # Merge and deduplicate results
        merged_text = self._merge_extracted_texts(all_texts)
        
        # 🔥 NEW: Extract structured outline and headings
        structured_data = self._create_structured_outline(pdf_path, all_texts)
        
        # 🔥 NEW: Extract tables
        tables = self._extract_tables(pdf_path)
        
        # Generate final output in expected format
        final_result = {
            'title': structured_data.get('title', 'Untitled Document'),
            'outline': structured_data.get('outline', []),
            'raw_text': [{'page': i+1, 'text': text} for i, text in enumerate(merged_text.split('\n\n'))],
            'tables': tables,
            'parser_results': results,
            'statistics': self._generate_statistics(results),
            'quality_score': self._calculate_quality_score(results)
        }
        
        self.logger.info(f"Extracted {len(final_result['outline'])} headings and {len(tables)} tables")
        return final_result

    def _extract_with_pdfplumber(self, pdf_path: Path) -> ParsingResult:
        """Extract text using pdfplumber - excellent for layout preservation."""
        texts = []
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout information
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        texts.append(ExtractedText(
                            text=page_text,
                            source='pdfplumber',
                            page_num=page_num,
                            confidence=0.9
                        ))
                    
                    # Also extract words with detailed positioning
                    words = page.extract_words(
                        use_text_flow=True,
                        keep_blank_chars=True,
                        extra_attrs=['fontname', 'size']
                    )
                    
                    if words:
                        # Group words into lines
                        lines = self._group_words_into_lines(words)
                        for line_text, bbox, font_info in lines:
                            texts.append(ExtractedText(
                                text=line_text,
                                source='pdfplumber_detailed',
                                page_num=page_num,
                                confidence=0.95,
                                bbox=bbox,
                                font_info=font_info
                            ))
            
            return ParsingResult('pdfplumber', texts, True)
            
        except Exception as e:
            return ParsingResult('pdfplumber', [], False, str(e))

    def _extract_with_pymupdf(self, pdf_path: Path) -> ParsingResult:
        """Extract text using PyMuPDF - fast and handles complex layouts well."""
        texts = []
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract plain text
                page_text = page.get_text()
                if page_text.strip():
                    texts.append(ExtractedText(
                        text=page_text,
                        source='pymupdf',
                        page_num=page_num + 1,
                        confidence=0.9
                    ))
                
                # Extract text with detailed formatting
                text_dict = page.get_text("dict")
                blocks = text_dict.get("blocks", [])
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            line_bbox = None
                            font_info = {}
                            
                            for span in line.get("spans", []):
                                line_text += span.get("text", "")
                                if not line_bbox:
                                    line_bbox = span.get("bbox")
                                font_info = {
                                    'font': span.get("font", ""),
                                    'size': span.get("size", 0),
                                    'flags': span.get("flags", 0)
                                }
                            
                            if line_text.strip():
                                texts.append(ExtractedText(
                                    text=line_text,
                                    source='pymupdf_detailed',
                                    page_num=page_num + 1,
                                    confidence=0.95,
                                    bbox=line_bbox,
                                    font_info=font_info
                                ))
            
            doc.close()
            return ParsingResult('pymupdf', texts, True)
            
        except Exception as e:
            return ParsingResult('pymupdf', [], False, str(e))

    def _extract_with_pdfminer(self, pdf_path: Path) -> ParsingResult:
        """Extract text using pdfminer - excellent for character-level accuracy."""
        texts = []
        try:
            # Standard extraction
            laparams = LAParams(
                detect_vertical=True,
                word_margin=0.1,
                char_margin=2.0,
                line_margin=0.5,
                boxes_flow=0.5
            )
            
            text = pdfminer_extract(str(pdf_path), laparams=laparams)
            if text.strip():
                # Split by pages (approximate)
                pages = text.split('\f')  # Form feed character often separates pages
                for page_num, page_text in enumerate(pages, 1):
                    if page_text.strip():
                        texts.append(ExtractedText(
                            text=page_text,
                            source='pdfminer',
                            page_num=page_num,
                            confidence=0.85
                        ))
            
            return ParsingResult('pdfminer', texts, True)
            
        except Exception as e:
            return ParsingResult('pdfminer', [], False, str(e))

    def _extract_with_camelot(self, pdf_path: Path) -> ParsingResult:
        """Extract tables using camelot."""
        texts = []
        if not CAMELOT_AVAILABLE:
            return ParsingResult('camelot', [], False, "Camelot not available")
        
        try:
            # Extract tables from all pages
            tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')
            
            for i, table in enumerate(tables):
                table_text = table.df.to_string(index=False)
                if table_text.strip():
                    texts.append(ExtractedText(
                        text=f"Table {i+1}:\n{table_text}",
                        source='camelot',
                        page_num=table.page,
                        confidence=0.8
                    ))
            
            # Try stream flavor as backup
            if not texts:
                tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='stream')
                for i, table in enumerate(tables):
                    table_text = table.df.to_string(index=False)
                    if table_text.strip():
                        texts.append(ExtractedText(
                            text=f"Table {i+1}:\n{table_text}",
                            source='camelot_stream',
                            page_num=table.page,
                            confidence=0.7
                        ))
            
            return ParsingResult('camelot', texts, True)
            
        except Exception as e:
            return ParsingResult('camelot', [], False, str(e))

    def _extract_with_ocr(self, pdf_path: Path) -> ParsingResult:
        """Extract text using OCR for scanned PDFs."""
        texts = []
        if not OCR_AVAILABLE:
            return ParsingResult('ocr', [], False, "OCR libraries not available")
        
        try:
            import fitz  # Use PyMuPDF for image extraction
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Check if page contains images (likely scanned)
                image_list = page.get_images()
                if image_list:
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                    img_data = pix.tobytes("png")
                    
                    # OCR the image
                    img = Image.open(io.BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(img, config='--psm 6')
                    
                    if ocr_text.strip():
                        texts.append(ExtractedText(
                            text=ocr_text,
                            source='ocr',
                            page_num=page_num + 1,
                            confidence=0.6  # OCR is less reliable
                        ))
            
            doc.close()
            return ParsingResult('ocr', texts, True)
            
        except Exception as e:
            return ParsingResult('ocr', [], False, str(e))

    def _group_words_into_lines(self, words: List[Dict]) -> List[Tuple[str, Tuple, Dict]]:
        """Group words into lines based on y-coordinate."""
        if not words:
            return []
        
        # Group by y-coordinate (with some tolerance)
        lines = defaultdict(list)
        for word in words:
            y = round(word.get('top', 0), 1)
            lines[y].append(word)
        
        result = []
        for y, line_words in lines.items():
            # Sort words by x-coordinate
            line_words.sort(key=lambda w: w.get('x0', 0))
            
            # Combine text
            line_text = ' '.join(w.get('text', '') for w in line_words)
            
            # Calculate bounding box
            if line_words:
                bbox = (
                    min(w.get('x0', 0) for w in line_words),
                    min(w.get('top', 0) for w in line_words),
                    max(w.get('x1', 0) for w in line_words),
                    max(w.get('bottom', 0) for w in line_words)
                )
                
                # Get font info from first word
                font_info = {
                    'fontname': line_words[0].get('fontname', ''),
                    'size': line_words[0].get('size', 0)
                }
                
                result.append((line_text, bbox, font_info))
        
        return result

    def _merge_extracted_texts(self, all_texts: List[ExtractedText]) -> str:
        """
        Intelligently merge text from all parsers to create the most complete version.
        """
        if not all_texts:
            return ""
        
        # Group texts by page
        pages = defaultdict(list)
        for text in all_texts:
            pages[text.page_num].append(text)
        
        merged_pages = []
        
        for page_num in sorted(pages.keys()):
            page_texts = pages[page_num]
            
            # Find the best text for this page
            best_text = self._find_best_page_text(page_texts)
            merged_pages.append(best_text)
        
        return '\n\n--- Page Break ---\n\n'.join(merged_pages)

    def _find_best_page_text(self, page_texts: List[ExtractedText]) -> str:
        """Find the best text representation for a single page."""
        if not page_texts:
            return ""
        
        if len(page_texts) == 1:
            return page_texts[0].text
        
        # Score each text based on length, confidence, and source reliability
        scored_texts = []
        for text in page_texts:
            score = (
                len(text.text) * 0.4 +  # Length factor
                text.confidence * 1000 +  # Confidence factor
                self._get_source_reliability(text.source) * 500  # Source reliability
            )
            scored_texts.append((score, text))
        
        # Sort by score
        scored_texts.sort(key=lambda x: x[0], reverse=True)
        
        # Take the highest scoring text as base
        base_text = scored_texts[0][1].text
        
        # Try to merge missing content from other sources
        for score, text in scored_texts[1:]:
            base_text = self._merge_text_intelligently(base_text, text.text)
        
        return base_text

    def _get_source_reliability(self, source: str) -> float:
        """Get reliability score for different sources."""
        reliability_scores = {
            'pdfplumber_detailed': 1.0,
            'pymupdf_detailed': 0.95,
            'pdfplumber': 0.9,
            'pymupdf': 0.85,
            'pdfminer': 0.8,
            'camelot': 0.8,
            'camelot_stream': 0.75,
            'ocr': 0.5
        }
        return reliability_scores.get(source, 0.5)

    def _merge_text_intelligently(self, base_text: str, additional_text: str) -> str:
        """Merge two text strings, adding missing content from additional_text."""
        # Simple approach: if additional text has significantly more content,
        # and it contains most of the base text, use the additional text
        
        if len(additional_text) > len(base_text) * 1.5:
            # Check if base text is mostly contained in additional text
            base_words = set(base_text.lower().split())
            additional_words = set(additional_text.lower().split())
            
            overlap = len(base_words & additional_words) / len(base_words) if base_words else 0
            
            if overlap > 0.7:  # 70% overlap
                return additional_text
        
        return base_text

    def _organize_by_page(self, all_texts: List[ExtractedText]) -> Dict[int, List[str]]:
        """Organize extracted text by page number."""
        pages = defaultdict(list)
        for text in all_texts:
            pages[text.page_num].append(f"[{text.source}] {text.text}")
        return dict(pages)

    def _generate_statistics(self, results: Dict[str, ParsingResult]) -> Dict[str, Any]:
        """Generate statistics about the parsing results."""
        stats = {
            'total_parsers': len(results),
            'successful_parsers': sum(1 for r in results.values() if r.success),
            'failed_parsers': sum(1 for r in results.values() if not r.success),
            'total_text_blocks': sum(len(r.texts) for r in results.values()),
            'parser_performance': {}
        }
        
        for parser_name, result in results.items():
            stats['parser_performance'][parser_name] = {
                'success': result.success,
                'text_blocks': len(result.texts),
                'execution_time': result.execution_time,
                'error': result.error
            }
        
        return stats

    def _calculate_quality_score(self, results: Dict[str, ParsingResult]) -> float:
        """Calculate overall extraction quality score."""
        if not results:
            return 0.0
        
        successful_parsers = [r for r in results.values() if r.success]
        if not successful_parsers:
            return 0.0
        
        # Base score on number of successful parsers and text volume
        base_score = len(successful_parsers) / len(results)
        
        # Bonus for high-reliability parsers
        high_reliability_parsers = ['pdfplumber', 'pymupdf', 'pdfminer']
        bonus = sum(0.1 for name in high_reliability_parsers 
                   if name in results and results[name].success)
        
        return min(1.0, base_score + bonus)

    def _create_structured_outline(self, pdf_path: Path, all_texts: List[ExtractedText]) -> Dict[str, Any]:
        """Create structured outline with title and H1/H2/H3 headings."""
        # Get font-enriched text blocks from PyMuPDF
        font_blocks = []
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["text"].strip():
                                    font_blocks.append({
                                        'text': span["text"].strip(),
                                        'font': span["font"],
                                        'size': span["size"],
                                        'flags': span["flags"],
                                        'bbox': span["bbox"],
                                        'page': page_num + 1,
                                        'is_bold': bool(span["flags"] & 2**4),
                                        'is_italic': bool(span["flags"] & 2**1)
                                    })
            doc.close()
        except Exception as e:
            self.logger.error(f"Failed to extract font information: {e}")
            return {'title': 'Untitled Document', 'outline': []}
        
        if not font_blocks:
            return {'title': 'Untitled Document', 'outline': []}
        
        # Analyze fonts to determine body text size
        font_stats = self._analyze_fonts(font_blocks)
        body_font_size = font_stats.get('body_font_size', 12.0)
        
        # Detect title
        title = self._detect_title_from_blocks(font_blocks, font_stats)
        
        # Detect headings
        headings = self._detect_headings_from_blocks(font_blocks, font_stats)
        
        return {
            'title': title,
            'outline': headings
        }
    
    def _analyze_fonts(self, font_blocks: List[Dict]) -> Dict[str, Any]:
        """Analyze font patterns to determine body text characteristics."""
        if not font_blocks:
            return {'body_font_size': 12.0}
        
        # Collect font sizes
        font_sizes = [block['size'] for block in font_blocks if block['size'] > 0]
        
        if not font_sizes:
            return {'body_font_size': 12.0}
        
        # Find most common font size (likely body text)
        from collections import Counter
        size_counts = Counter(font_sizes)
        body_font_size = size_counts.most_common(1)[0][0]
        
        return {
            'body_font_size': body_font_size,
            'font_sizes': font_sizes,
            'size_distribution': dict(size_counts),
            'max_size': max(font_sizes),
            'min_size': min(font_sizes)
        }
    
    def _detect_title_from_blocks(self, font_blocks: List[Dict], font_stats: Dict) -> str:
        """Detect document title from font blocks."""
        # Focus on first page
        first_page_blocks = [b for b in font_blocks if b['page'] == 1]
        
        if not first_page_blocks:
            return "Untitled Document"
        
        body_font_size = font_stats['body_font_size']
        max_font_size = font_stats.get('max_size', body_font_size)
        
        # Find title candidates (large font on first page, upper portion)
        title_candidates = []
        for block in first_page_blocks:
            if (block['size'] >= max(max_font_size * 0.9, body_font_size * 1.3) and
                len(block['text']) > 3 and
                block['bbox'][1] < 300):  # Upper portion of page
                
                score = self._calculate_title_score(block, body_font_size)
                title_candidates.append((block, score))
        
        if title_candidates:
            title_candidates.sort(key=lambda x: (-x[1], x[0]['bbox'][1]))
            return title_candidates[0][0]['text']
        
        return "Untitled Document"
    
    def _calculate_title_score(self, block: Dict, body_font_size: float) -> float:
        """Calculate title likelihood score."""
        score = 0.0
        
        # Font size factor
        size_ratio = block['size'] / body_font_size
        score += min(0.5, (size_ratio - 1.0) * 0.25)
        
        # Position (prefer top)
        if block['bbox'][1] < 100:
            score += 0.3
        elif block['bbox'][1] < 200:
            score += 0.2
        
        # Style factors
        if block['is_bold']:
            score += 0.2
        
        # Length factor
        text_length = len(block['text'])
        if 10 <= text_length <= 100:
            score += 0.2
        
        return score
    
    def _detect_headings_from_blocks(self, font_blocks: List[Dict], font_stats: Dict) -> List[Dict[str, Any]]:
        """Detect and classify headings from font blocks."""
        body_font_size = font_stats['body_font_size']
        heading_threshold = body_font_size * 1.15  # 15% larger than body
        
        heading_candidates = []
        
        for block in font_blocks:
            if (block['size'] >= heading_threshold and
                len(block['text'].strip()) >= 3 and
                len(block['text'].strip()) <= 200 and
                not self._is_page_number(block['text'])):
                
                confidence = self._calculate_heading_confidence(block, body_font_size)
                if confidence >= 0.4:  # Lower threshold for better recall
                    heading_candidates.append({
                        'text': block['text'],
                        'page': block['page'],
                        'font_size': block['size'],
                        'is_bold': block['is_bold'],
                        'confidence': confidence,
                        'y_position': block['bbox'][1]
                    })
        
        # Assign heading levels based on font size
        return self._assign_heading_levels(heading_candidates)
    
    def _calculate_heading_confidence(self, block: Dict, body_font_size: float) -> float:
        """Calculate heading confidence score."""
        score = 0.0
        
        # Font size factor
        size_ratio = block['size'] / body_font_size
        if size_ratio > 1.5:
            score += 0.4
        elif size_ratio > 1.2:
            score += 0.3
        
        # Bold bonus
        if block['is_bold']:
            score += 0.3
        
        # Position (left-aligned more likely)
        if block['bbox'][0] < 100:
            score += 0.2
        
        # Pattern matching
        text = block['text'].lower()
        if any(pattern in text for pattern in ['chapter', 'section', 'introduction', 'conclusion']):
            score += 0.3
        
        # Numbered headings
        if re.match(r'^\d+(\.\d+)*\.?\s', block['text']):
            score += 0.2
        
        return min(1.0, score)
    
    def _assign_heading_levels(self, candidates: List[Dict]) -> List[Dict[str, Any]]:
        """Assign H1, H2, H3 levels to heading candidates."""
        if not candidates:
            return []
        
        # Sort by confidence and font size
        candidates.sort(key=lambda x: (-x['font_size'], -x['confidence']))
        
        # Group by font size
        size_groups = {}
        for candidate in candidates:
            size_key = round(candidate['font_size'], 1)
            if size_key not in size_groups:
                size_groups[size_key] = []
            size_groups[size_key].append(candidate)
        
        # Assign levels
        outline = []
        level_map = {0: "H1", 1: "H2", 2: "H3"}
        
        for level_idx, (size, group) in enumerate(sorted(size_groups.items(), reverse=True)):
            if level_idx >= 3:  # Only H1, H2, H3
                break
            
            level = level_map[level_idx]
            
            # Sort by page and position
            group.sort(key=lambda x: (x['page'], x['y_position']))
            
            for heading in group:
                outline.append({
                    'level': level,
                    'text': heading['text'],
                    'page': heading['page']
                })
        
        # Final sort by document order
        outline.sort(key=lambda x: (x['page'], 
            next((h['y_position'] for h in candidates 
                  if h['text'] == x['text'] and h['page'] == x['page']), 0)))
        
        return outline
    
    def _is_page_number(self, text: str) -> bool:
        """Check if text is likely a page number."""
        text = text.strip()
        if text.isdigit() and 1 <= int(text) <= 999:
            return True
        patterns = [r'^page\s+\d+$', r'^-\s*\d+\s*-$', r'^\d+\s*$']
        return any(re.match(pattern, text.lower()) for pattern in patterns)
    
    def _extract_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(str(pdf_path)) as doc:
                for page_idx, page in enumerate(doc.pages):
                    page_tables = page.find_tables()
                    for t_idx, table in enumerate(page_tables):
                        data = table.extract()
                        if data:
                            tables.append({
                                "page": page_idx + 1,
                                "index": t_idx,
                                "data": data
                            })
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
        
        return tables


# Usage example and testing
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 2:
        print("Usage: python multi_parser_extractor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    extractor = PDFOutlineParser(enable_ocr=True, enable_tables=True)
    result = extractor.extract_outline(pdf_path)

    save_to_json(
    result,
    f"output/{Path(pdf_path).stem}.json"  # or f"output/{Path(file_path).stem}.json"
)

    
    print(f"\n=== EXTRACTION RESULTS ===")
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"Total Characters: {len(result['merged_text'])}")
    print(f"\n=== PARSER STATISTICS ===")
    
    for parser, stats in result['statistics']['parser_performance'].items():
        status = "✓" if stats['success'] else "✗"
        print(f"{status} {parser}: {stats['text_blocks']} blocks, {stats['execution_time']:.2f}s")
    
    print(f"\n=== MERGED TEXT (first 1000 chars) ===")
    print(result['merged_text'][:1000])
    print("..." if len(result['merged_text']) > 1000 else "")