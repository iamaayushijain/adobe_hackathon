"""
Utility Functions for PDF Outline Extraction
Adobe Hackathon "Connecting the Dots" Challenge - Round 1A

Provides font analysis, heading detection, and text processing utilities.
"""

import re
import statistics
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class FontInfo:
    """Font information container."""
    name: str
    size: float
    is_bold: bool
    is_italic: bool
    usage_count: int


class FontAnalyzer:
    """
    Analyzes font patterns in PDF text blocks to determine body text and heading characteristics.
    """
    
    def __init__(self):
        self.min_font_frequency = 5  # Minimum occurrences to consider a font as body text
        
    def analyze_fonts(self, text_blocks: List[Any]) -> Dict[str, Any]:
        """
        Analyze font patterns in text blocks to determine document structure.
        
        Args:
            text_blocks: List of TextBlock objects
            
        Returns:
            Dictionary containing font analysis results
        """
        if not text_blocks:
            return {
                'body_font_size': 12.0,
                'common_fonts': [],
                'font_size_distribution': {},
                'heading_font_threshold': 14.0
            }
        
        # Collect font statistics
        font_sizes = []
        font_names = []
        bold_sizes = []
        
        for block in text_blocks:
            font_sizes.append(block.font_size)
            font_names.append(block.font_name)
            
            if block.is_bold:
                bold_sizes.append(block.font_size)
        
        # Analyze font size distribution
        size_counter = Counter(font_sizes)
        
        # Determine body text font size (most common size)
        most_common_sizes = size_counter.most_common(5)
        body_font_size = most_common_sizes[0][0] if most_common_sizes else 12.0
        
        # Filter out very small text (footnotes, page numbers) when determining body size
        filtered_sizes = [size for size in font_sizes if size >= 8.0]
        if filtered_sizes:
            body_font_size = statistics.mode(filtered_sizes)
        
        # Determine heading threshold
        sorted_sizes = sorted(size_counter.keys(), reverse=True)
        heading_threshold = body_font_size * 1.2  # 20% larger than body text
        
        # Find sizes that could be headings
        potential_heading_sizes = [size for size in sorted_sizes if size > heading_threshold]
        
        # Analyze font names
        name_counter = Counter(font_names)
        common_fonts = [name for name, count in name_counter.most_common(3)]
        
        return {
            'body_font_size': body_font_size,
            'common_fonts': common_fonts,
            'font_size_distribution': dict(size_counter),
            'heading_font_threshold': heading_threshold,
            'potential_heading_sizes': potential_heading_sizes,
            'bold_font_sizes': Counter(bold_sizes),
            'size_statistics': {
                'min': min(font_sizes),
                'max': max(font_sizes),
                'median': statistics.median(font_sizes),
                'mean': statistics.mean(font_sizes)
            }
        }
    
    def get_font_category(self, font_size: float, font_stats: Dict) -> str:
        """
        Categorize a font size as body, heading, or other.
        
        Args:
            font_size: Font size to categorize
            font_stats: Font analysis statistics
            
        Returns:
            Category string: 'body', 'heading', 'small', or 'large'
        """
        body_size = font_stats.get('body_font_size', 12.0)
        threshold = font_stats.get('heading_font_threshold', body_size * 1.2)
        
        if font_size < body_size * 0.8:
            return 'small'  # Footnotes, captions
        elif font_size >= threshold:
            return 'heading'
        elif abs(font_size - body_size) < 1.0:
            return 'body'
        else:
            return 'large'


class HeadingDetector:
    """
    Advanced heading detection using multiple heuristics and pattern matching.
    """
    
    def __init__(self):
        # Common heading patterns
        self.heading_patterns = [
            r'^(chapter|section|part|appendix)\s+\d+',  # Chapter 1, Section 2
            r'^\d+(\.\d+)*\.?\s+',  # 1., 1.1., 1.1.1.
            r'^[A-Z]+\.\s+',  # A., B., C.
            r'^[IVX]+\.\s+',  # I., II., III. (Roman numerals)
            r'^(introduction|conclusion|summary|abstract|references)',
            r'^table\s+of\s+contents',
            r'^bibliography|^index$',
        ]
        
        # Patterns that are NOT headings
        self.false_positive_patterns = [
            r'^\d+$',  # Just page numbers
            r'^page\s+\d+',  # "Page 1"
            r'^figure\s+\d+',  # Figure captions
            r'^table\s+\d+',  # Table captions
            r'^equation\s+\d+',  # Equation numbers
            r'^\(\d+\)$',  # Numbered lists in parentheses
            r'^[a-z]+@[a-z]+\.',  # Email addresses
            r'^https?://',  # URLs
            r'^\d{1,2}/\d{1,2}/\d{2,4}',  # Dates
        ]
    
    def is_likely_heading(self, text: str, font_size: float, is_bold: bool, 
                         position_x: float, font_stats: Dict) -> Tuple[bool, float]:
        """
        Determine if a text block is likely a heading.
        
        Args:
            text: Text content
            font_size: Font size of the text
            is_bold: Whether text is bold
            position_x: X position on page
            font_stats: Font analysis statistics
            
        Returns:
            Tuple of (is_heading, confidence_score)
        """
        text = text.strip()
        confidence = 0.0
        
        # Quick rejection for obviously non-headings
        if len(text) < 2 or len(text) > 300:
            return False, 0.0
        
        # Check false positive patterns
        for pattern in self.false_positive_patterns:
            if re.match(pattern, text.lower()):
                return False, 0.0
        
        # Font size scoring
        body_font_size = font_stats.get('body_font_size', 12.0)
        size_ratio = font_size / body_font_size
        
        if size_ratio > 1.5:
            confidence += 0.4
        elif size_ratio > 1.2:
            confidence += 0.3
        elif size_ratio > 1.1:
            confidence += 0.2
        
        # Bold text bonus
        if is_bold:
            confidence += 0.3
        
        # Position scoring (left-aligned text more likely to be headings)
        if position_x < 50:  # Very left-aligned
            confidence += 0.2
        elif position_x < 100:  # Moderately left-aligned
            confidence += 0.1
        
        # Pattern matching for known heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text.lower()):
                confidence += 0.4
                break
        
        # Text formatting heuristics
        if text.isupper() and len(text) > 5:  # ALL CAPS headings
            confidence += 0.2
        elif text.istitle():  # Title Case
            confidence += 0.1
        
        # Length-based scoring
        text_length = len(text)
        if 5 <= text_length <= 80:  # Good heading length
            confidence += 0.1
        elif text_length > 150:  # Too long for a heading
            confidence -= 0.2
        
        # Word count heuristics
        word_count = len(text.split())
        if 1 <= word_count <= 15:  # Reasonable number of words
            confidence += 0.1
        elif word_count > 25:  # Too many words
            confidence -= 0.1
        
        # Check for common non-heading indicators
        lower_text = text.lower()
        non_heading_indicators = [
            'http', 'www', '.com', '.org', '.edu',  # URLs
            'fig.', 'figure', 'table', 'eq.', 'equation',  # References
            'see page', 'continued on', 'page',  # Page references
        ]
        
        for indicator in non_heading_indicators:
            if indicator in lower_text:
                confidence -= 0.3
                break
        
        is_heading = confidence >= 0.5
        return is_heading, min(1.0, max(0.0, confidence))
    
    def extract_numbering(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Extract numbering scheme from heading text.
        
        Args:
            text: Heading text
            
        Returns:
            Tuple of (numbering, clean_text) or None
        """
        text = text.strip()
        
        # Decimal numbering (1., 1.1., 1.1.1.)
        match = re.match(r'^(\d+(?:\.\d+)*\.?)\s*(.+)', text)
        if match:
            return match.group(1), match.group(2).strip()
        
        # Roman numerals (I., II., III.)
        match = re.match(r'^([IVX]+\.?)\s*(.+)', text.upper())
        if match:
            return match.group(1), match.group(2).strip()
        
        # Letters (A., B., C.)
        match = re.match(r'^([A-Z]\.?)\s*(.+)', text.upper())
        if match:
            return match.group(1), match.group(2).strip()
        
        # Chapter/Section patterns
        match = re.match(r'^(chapter|section|part)\s+(\d+)\s*(.+)', text.lower())
        if match:
            return f"{match.group(1).title()} {match.group(2)}", match.group(3).strip()
        
        return None


class TextBlockProcessor:
    """
    Processes and cleans text blocks for better heading detection.
    """
    
    def __init__(self):
        pass
    
    def clean_title_text(self, text: str) -> str:
        """
        Clean and normalize title text.
        
        Args:
            text: Raw title text
            
        Returns:
            Cleaned title text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common title prefixes/suffixes
        prefixes_to_remove = [
            r'^title:\s*',
            r'^subject:\s*',
            r'^document:\s*',
        ]
        
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        
        # Limit title length
        if len(text) > 200:
            # Try to find a good breaking point
            words = text.split()
            if len(words) > 20:
                text = ' '.join(words[:20]) + '...'
        
        return text.strip()
    
    def clean_heading_text(self, text: str) -> str:
        """
        Clean and normalize heading text.
        
        Args:
            text: Raw heading text
            
        Returns:
            Cleaned heading text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove trailing dots that aren't part of numbering
        if not re.match(r'^\d+(\.\d+)*\.$', text):  # Don't remove from "1.1."
            text = text.rstrip('.')
        
        # Remove page references at the end
        text = re.sub(r'\s+\d+$', '', text)  # Remove trailing page numbers
        text = re.sub(r'\s+\.{2,}\s*\d+$', '', text)  # Remove ".... 25"
        
        return text.strip()
    
    def merge_text_blocks(self, blocks: List[Any], max_gap: float = 5.0) -> List[Any]:
        """
        Merge adjacent text blocks that likely belong together.
        
        Args:
            blocks: List of text blocks
            max_gap: Maximum gap between blocks to merge
            
        Returns:
            List of merged text blocks
        """
        if not blocks:
            return []
        
        # Sort blocks by page and position
        sorted_blocks = sorted(blocks, key=lambda x: (x.page_num, x.y0, x.x0))
        merged = []
        
        for block in sorted_blocks:
            if (merged and 
                merged[-1].page_num == block.page_num and
                abs(merged[-1].y0 - block.y0) <= max_gap and
                merged[-1].font_size == block.font_size and
                merged[-1].is_bold == block.is_bold):
                
                # Merge with previous block
                merged[-1].text += " " + block.text
                merged[-1].x1 = max(merged[-1].x1, block.x1)
            else:
                merged.append(block)
        
        return merged
    
    def calculate_line_spacing(self, blocks: List[Any]) -> Dict[str, float]:
        """
        Calculate typical line spacing in the document.
        
        Args:
            blocks: List of text blocks
            
        Returns:
            Dictionary with spacing statistics
        """
        if len(blocks) < 2:
            return {'mean': 15.0, 'median': 15.0, 'std': 0.0}
        
        # Group blocks by page
        pages = defaultdict(list)
        for block in blocks:
            pages[block.page_num].append(block)
        
        all_gaps = []
        
        for page_blocks in pages.values():
            # Sort by y position
            page_blocks.sort(key=lambda x: x.y0)
            
            # Calculate gaps between consecutive blocks
            for i in range(len(page_blocks) - 1):
                gap = page_blocks[i + 1].y0 - page_blocks[i].y1
                if 0 <= gap <= 100:  # Reasonable line spacing
                    all_gaps.append(gap)
        
        if not all_gaps:
            return {'mean': 15.0, 'median': 15.0, 'std': 0.0}
        
        return {
            'mean': statistics.mean(all_gaps),
            'median': statistics.median(all_gaps),
            'std': statistics.stdev(all_gaps) if len(all_gaps) > 1 else 0.0,
            'gaps': all_gaps
        } 