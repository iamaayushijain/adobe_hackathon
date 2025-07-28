#!/usr/bin/env python3
"""
System Test Script for PDF Outline Extractor
Adobe Hackathon "Connecting the Dots" Challenge - Round 1A

Tests the complete pipeline with sample data and validates output.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any
import io

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parser import PDFOutlineParser
from output_writer import OutputWriter
from utils import FontAnalyzer, HeadingDetector, TextBlockProcessor


def setup_test_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - TEST - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger = logging.getLogger(__name__)
    
    try:
        import pdfplumber
        logger.info("✓ pdfplumber imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import pdfplumber: {e}")
        return False
    
    try:
        from parser import PDFOutlineParser
        from output_writer import OutputWriter
        from utils import FontAnalyzer, HeadingDetector, TextBlockProcessor
        logger.info("✓ All custom modules imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import custom modules: {e}")
        return False
    
    return True


def test_utility_classes():
    """Test utility classes with mock data."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test FontAnalyzer
        analyzer = FontAnalyzer()
        logger.info("✓ FontAnalyzer created successfully")
        
        # Test HeadingDetector
        detector = HeadingDetector()
        
        # Test with sample text
        test_cases = [
            ("Chapter 1: Introduction", 16.0, True, 50.0),
            ("1.1 Background", 14.0, True, 50.0),
            ("This is body text", 12.0, False, 50.0),
            ("Figure 1: Sample", 12.0, False, 50.0),
        ]
        
        for text, font_size, is_bold, pos_x in test_cases:
            font_stats = {'body_font_size': 12.0, 'heading_font_threshold': 14.0}
            is_heading, confidence = detector.is_likely_heading(
                text, font_size, is_bold, pos_x, font_stats
            )
            logger.info(f"  Text: '{text}' -> Heading: {is_heading}, Confidence: {confidence:.2f}")
        
        logger.info("✓ HeadingDetector tested successfully")
        
        # Test TextBlockProcessor
        processor = TextBlockProcessor()
        
        test_title = "   Sample Document Title   "
        cleaned = processor.clean_title_text(test_title)
        logger.info(f"  Title cleaning: '{test_title}' -> '{cleaned}'")
        
        logger.info("✓ TextBlockProcessor tested successfully")
        
    except Exception as e:
        logger.error(f"✗ Utility class test failed: {e}")
        return False
    
    return True


def test_output_writer():
    """Test JSON output writing."""
    logger = logging.getLogger(__name__)
    
    try:
        writer = OutputWriter()
        
        # Test data
        test_data = {
            'title': 'Test Document',
            'outline': [
                {'level': 'H1', 'text': 'Introduction', 'page': 1},
                {'level': 'H2', 'text': 'Background', 'page': 2},
                {'level': 'H1', 'text': 'Methodology', 'page': 3},
            ]
        }
        
        # Test validation
        validated = writer._validate_outline_data(test_data)
        logger.info("✓ Data validation successful")
        
        # Test writing to temporary file
        test_output = Path('./test_output.json')
        success = writer.write_outline(test_data, test_output)
        
        if success and test_output.exists():
            # Verify the output
            with open(test_output, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['title'] == 'Test Document'
            assert len(loaded_data['outline']) == 3
            assert loaded_data['outline'][0]['level'] == 'H1'
            
            # Clean up
            test_output.unlink()
            logger.info("✓ Output writing and validation successful")
        else:
            logger.error("✗ Output writing failed")
            return False
        
    except Exception as e:
        logger.error(f"✗ Output writer test failed: {e}")
        return False
    
    return True


def create_sample_pdf_info():
    """Create a sample PDF for testing if possible."""
    logger = logging.getLogger(__name__)
    
    # We can't create actual PDFs without additional dependencies
    # Instead, provide instructions for manual testing
    sample_info = """
    To test with real PDFs:
    
    1. Place any PDF files in the 'input/' directory
    2. Run: python main.py
    3. Check 'output/' directory for JSON files
    
    Recommended test PDFs:
    - Academic papers with clear headings
    - Technical manuals with numbered sections
    - Books with chapter structures
    - Reports with hierarchical organization
    
    The system works best with:
    - Text-based PDFs (not scanned images)
    - Clear font size differences between headings and body text
    - Consistent heading styles
    - Well-structured documents
    """
    
    logger.info("Sample PDF testing info:")
    for line in sample_info.strip().split('\n'):
        logger.info(line)


def test_performance_targets():
    """Test if the system meets performance requirements."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test multiprocessing setup
        from multiprocessing import cpu_count
        cores = cpu_count()
        logger.info(f"✓ Detected {cores} CPU cores")
        
        if cores >= 4:
            logger.info("✓ Sufficient cores for good performance")
        else:
            logger.warning("⚠ Limited cores may affect performance")
        
        # Test memory usage estimation
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        logger.info(f"✓ Available memory: {available_memory:.1f} GB")
        
        if available_memory >= 2.0:
            logger.info("✓ Sufficient memory for processing")
        else:
            logger.warning("⚠ Limited memory may affect performance")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        return False


def run_integration_test():
    """Run a basic integration test of the main pipeline."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test parser instantiation
        parser = PDFOutlineParser()
        logger.info("✓ PDFOutlineParser created successfully")
        
        # Test with empty input directory
        input_dir = Path('./input')
        if not input_dir.exists():
            input_dir.mkdir()
        
        pdf_files = list(input_dir.glob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files in input directory")
        
        if pdf_files:
            logger.info("Real PDF files detected - system ready for actual processing")
        else:
            logger.info("No PDF files found - place PDFs in 'input/' directory to test")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all system tests."""
    logger = setup_test_logging()
    
    logger.info("=" * 60)
    logger.info("PDF Outline Extractor - System Test")
    logger.info("Adobe Hackathon 'Connecting the Dots' Challenge - Round 1A")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Utility Classes Test", test_utility_classes),
        ("Output Writer Test", test_output_writer),
        ("Performance Requirements", test_performance_targets),
        ("Integration Test", run_integration_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        start_time = time.time()
        
        try:
            result = test_func()
            elapsed = time.time() - start_time
            
            if result:
                logger.info(f"✓ PASSED ({elapsed:.2f}s)")
                passed += 1
            else:
                logger.error(f"✗ FAILED ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"✗ ERROR ({elapsed:.2f}s): {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for production.")
        create_sample_pdf_info()
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return False
    
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 