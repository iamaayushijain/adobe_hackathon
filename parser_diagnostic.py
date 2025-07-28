#!/usr/bin/env python3
"""
parser_diagnostic.py
-------------------
Diagnostic script to test each PDF parser separately and show what content
each one retrieves. Helps identify which parser is best for different types of content.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pdfplumber(pdf_path: Path) -> Dict[str, Any]:
    """Test pdfplumber extraction"""
    try:
        import pdfplumber
        results = {
            "parser": "pdfplumber",
            "text_pages": [],
            "tables": [],
            "chars": [],
            "words": [],
            "font_info": []
        }
        
        with pdfplumber.open(str(pdf_path)) as doc:
            for page_num, page in enumerate(doc.pages):
                # Extract text
                text = page.extract_text() or ""
                results["text_pages"].append({
                    "page": page_num + 1,
                    "text": text[:200] + "..." if len(text) > 200 else text
                })
                
                # Extract tables
                tables = page.find_tables()
                for i, table in enumerate(tables):
                    data = table.extract()
                    results["tables"].append({
                        "page": page_num + 1,
                        "table_index": i,
                        "rows": len(data) if data else 0,
                        "cols": len(data[0]) if data and data[0] else 0,
                        "sample": data[:2] if data else []
                    })
                
                # Extract characters with font info (first 10 chars per page)
                chars = page.chars[:10]
                for char in chars:
                    results["chars"].append({
                        "page": page_num + 1,
                        "text": char.get("text", ""),
                        "fontname": char.get("fontname", ""),
                        "size": char.get("size", 0),
                        "x0": char.get("x0", 0),
                        "y0": char.get("top", 0)
                    })
                
                # Extract words with formatting (first 10 words per page)
                words = page.extract_words()[:10]
                for word in words:
                    results["words"].append({
                        "page": page_num + 1,
                        "text": word.get("text", ""),
                        "fontname": word.get("fontname", ""),
                        "size": word.get("size", 0),
                        "x0": word.get("x0", 0),
                        "top": word.get("top", 0)
                    })
                
                # Font analysis
                font_sizes = [char.get("size", 0) for char in page.chars]
                font_names = [char.get("fontname", "") for char in page.chars]
                if font_sizes:
                    results["font_info"].append({
                        "page": page_num + 1,
                        "unique_fonts": list(set(font_names)),
                        "font_size_range": [min(font_sizes), max(font_sizes)],
                        "avg_font_size": sum(font_sizes) / len(font_sizes)
                    })
        
        return results
    except Exception as e:
        logger.error(f"pdfplumber failed: {e}")
        return {"parser": "pdfplumber", "error": str(e)}

def test_pymupdf(pdf_path: Path) -> Dict[str, Any]:
    """Test PyMuPDF (fitz) extraction"""
    try:
        import fitz
        results = {
            "parser": "pymupdf",
            "text_pages": [],
            "text_blocks": [],
            "font_analysis": []
        }
        
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract plain text
            text = page.get_text()
            results["text_pages"].append({
                "page": page_num + 1,
                "text": text[:200] + "..." if len(text) > 200 else text
            })
            
            # Extract text blocks with formatting
            blocks = page.get_text("dict")
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                results["text_blocks"].append({
                                    "page": page_num + 1,
                                    "text": span["text"],
                                    "font": span["font"],
                                    "size": span["size"],
                                    "flags": span["flags"],
                                    "bbox": span["bbox"],
                                    "is_bold": bool(span["flags"] & 2**4),
                                    "is_italic": bool(span["flags"] & 2**1)
                                })
            
            # Font analysis
            font_info = {}
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_key = f"{span['font']}-{span['size']}"
                            if font_key not in font_info:
                                font_info[font_key] = {
                                    "font": span["font"],
                                    "size": span["size"],
                                    "count": 0,
                                    "sample_text": span["text"][:50]
                                }
                            font_info[font_key]["count"] += 1
            
            results["font_analysis"].append({
                "page": page_num + 1,
                "fonts": list(font_info.values())[:10]  # Top 10 fonts
            })
        
        doc.close()
        return results
    except Exception as e:
        logger.error(f"PyMuPDF failed: {e}")
        return {"parser": "pymupdf", "error": str(e)}

def test_pdfminer(pdf_path: Path) -> Dict[str, Any]:
    """Test pdfminer.six extraction"""
    try:
        from pdfminer.high_level import extract_text, extract_pages
        from pdfminer.layout import LTTextContainer, LTChar
        
        results = {
            "parser": "pdfminer",
            "full_text": "",
            "text_containers": [],
            "chars": []
        }
        
        # Extract full text
        full_text = extract_text(str(pdf_path))
        results["full_text"] = full_text[:500] + "..." if len(full_text) > 500 else full_text
        
        # Extract detailed layout
        char_count = 0
        for page_num, page_layout in enumerate(extract_pages(str(pdf_path))):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if text:
                        results["text_containers"].append({
                            "page": page_num + 1,
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "bbox": element.bbox
                        })
                elif isinstance(element, LTChar) and char_count < 50:  # Limit to first 50 chars
                    results["chars"].append({
                        "page": page_num + 1,
                        "text": element.get_text(),
                        "fontname": element.fontname,
                        "size": element.height,
                        "bbox": element.bbox
                    })
                    char_count += 1
        
        return results
    except Exception as e:
        logger.error(f"pdfminer failed: {e}")
        return {"parser": "pdfminer", "error": str(e)}

def test_camelot(pdf_path: Path) -> Dict[str, Any]:
    """Test Camelot table extraction"""
    try:
        import camelot
        results = {
            "parser": "camelot",
            "tables": []
        }
        
        # Extract tables
        tables = camelot.read_pdf(str(pdf_path), pages='all')
        for i, table in enumerate(tables):
            results["tables"].append({
                "table_index": i,
                "page": table.page,
                "shape": table.shape,
                "accuracy": table.accuracy,
                "sample_data": table.df.head(3).to_dict() if hasattr(table, 'df') else None
            })
        
        return results
    except Exception as e:
        logger.error(f"Camelot failed: {e}")
        return {"parser": "camelot", "error": str(e)}

def run_diagnostic(pdf_path: Path) -> Dict[str, Any]:
    """Run all parsers and return diagnostic results"""
    logger.info(f"Running diagnostic on: {pdf_path}")
    
    results = {
        "pdf_file": str(pdf_path),
        "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
        "parsers": {}
    }
    
    # Test each parser
    parsers = [
        ("pdfplumber", test_pdfplumber),
        ("pymupdf", test_pymupdf),
        ("pdfminer", test_pdfminer),
        ("camelot", test_camelot)
    ]
    
    for parser_name, parser_func in parsers:
        logger.info(f"Testing {parser_name}...")
        start_time = time.time()
        
        try:
            parser_results = parser_func(pdf_path)
            parser_results["processing_time"] = time.time() - start_time
            results["parsers"][parser_name] = parser_results
        except Exception as e:
            logger.error(f"{parser_name} failed completely: {e}")
            results["parsers"][parser_name] = {
                "parser": parser_name,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    return results

def main():
    """Main diagnostic function"""
    if len(sys.argv) != 2:
        print("Usage: python parser_diagnostic.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: {pdf_path} does not exist")
        sys.exit(1)
    
    # Run diagnostic
    results = run_diagnostic(pdf_path)
    
    # Save results
    output_path = Path(f"diagnostic_{pdf_path.stem}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDiagnostic complete! Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PARSER DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for parser_name, parser_data in results["parsers"].items():
        print(f"\n{parser_name.upper()}:")
        print(f"  Status: {'✓ Success' if 'error' not in parser_data else '✗ Failed'}")
        if 'error' not in parser_data:
            print(f"  Processing time: {parser_data.get('processing_time', 0):.2f}s")
            
            # Show what was extracted
            if parser_name == "pdfplumber":
                print(f"  Text pages: {len(parser_data.get('text_pages', []))}")
                print(f"  Tables found: {len(parser_data.get('tables', []))}")
                print(f"  Font variations: {len(parser_data.get('font_info', []))}")
            elif parser_name == "pymupdf":
                print(f"  Text pages: {len(parser_data.get('text_pages', []))}")
                print(f"  Text blocks with formatting: {len(parser_data.get('text_blocks', []))}")
            elif parser_name == "pdfminer":
                print(f"  Full text length: {len(parser_data.get('full_text', ''))}")
                print(f"  Text containers: {len(parser_data.get('text_containers', []))}")
            elif parser_name == "camelot":
                print(f"  Tables found: {len(parser_data.get('tables', []))}")
        else:
            print(f"  Error: {parser_data['error']}")

if __name__ == "__main__":
    main() 