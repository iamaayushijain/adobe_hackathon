#!/usr/bin/env python3
"""
PDF Outline Extractor - Main Entry Point
Adobe Hackathon "Connecting the Dots" Challenge - Round 1A

Processes PDFs from /app/input/ and generates structured JSON outlines in /app/output/
Optimized for performance with multiprocessing support.
"""

import os
import sys
import time
import logging
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import json
import io
from datetime import datetime
from app.embedder import embed
from app.ranker import rank_sections
from app.subsection_selector import select_subsections

from parser import PDFOutlineParser
from output_writer import OutputWriter
from pipeline import DocumentPipeline


def setup_logging() -> logging.Logger:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pdf_outline_extractor.log')
        ]
    )
    return logging.getLogger(__name__)


def process_single_pdf(pdf_path: Path, output_dir: Path) -> Tuple[str, bool, float]:
    """
    Process a single PDF file and generate its outline JSON.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the output JSON
        
    Returns:
        Tuple of (filename, success, processing_time)
    """
    start_time = time.time()
    filename = pdf_path.name
    logger = logging.getLogger(__name__)
    
    try:
        # Parse PDF via pipeline
        pipeline = DocumentPipeline()
        outline_data = pipeline.process(pdf_path)
        
        # Ensure output2 directory exists
        output2_dir = Path("output2")
        output2_dir.mkdir(exist_ok=True)
        
        # Save to output2 directory
        save_to_json(
            outline_data,
            f"output2/{Path(pdf_path).stem}.json"
        )
        
        # Write JSON output to main output directory
        output_filename = pdf_path.stem + '.json'
        output_path = output_dir / output_filename
        
        writer = OutputWriter()
        writer.write_outline(outline_data, output_path)
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully processed {filename} in {processing_time:.2f}s")
        return filename, True, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Failed to process {filename}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return filename, False, processing_time


def save_to_json(data, output_path):
    """Save data to JSON file with proper error handling."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save JSON to {output_path}: {e}")
        raise


def process_pdf_batch(pdf_paths: List[Path], output_dir: Path) -> List[Tuple[str, bool, float]]:
    """
    Process a batch of PDFs using multiprocessing.
    
    Args:
        pdf_paths: List of PDF file paths
        output_dir: Directory to save output files
        
    Returns:
        List of processing results
    """
    # Determine optimal number of processes (max 8 for performance)
    num_processes = min(cpu_count(), 8, len(pdf_paths))
    
    if num_processes == 1:
        # Single process for small batches
        return [process_single_pdf(pdf_path, output_dir) for pdf_path in pdf_paths]
    
    # Multiprocessing for larger batches
    with Pool(processes=num_processes) as pool:
        args = [(pdf_path, output_dir) for pdf_path in pdf_paths]
        results = pool.starmap(process_single_pdf, args)
    
    return results


def validate_directories(input_dir: Path, output_dir: Path) -> bool:
    """
    Validate input and output directories.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        
    Returns:
        True if directories are valid, False otherwise
    """
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return False
    
    if not input_dir.is_dir():
        logging.error(f"Input path is not a directory: {input_dir}")
        return False
    
    # Create output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create output directory {output_dir}: {str(e)}")
        return False
    
    return True


def find_pdf_files(input_dir: Path) -> List[Path]:
    """
    Find all PDF files in the input directory.
    
    Args:
        input_dir: Input directory path
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.pdf':
            pdf_files.append(file_path)
    
    return sorted(pdf_files)


def _process_collection(collection_dir: Path):
    """Handle a single Challenge-1B collection directory."""
    logger = logging.getLogger(__name__)

    challenge_input = collection_dir / "challenge1b_input.json"
    # Handle case-sensitive filesystems (Linux containers) – try common variants
    pdf_dir = None
    for dir_candidate in ("PDFs", "PDFS", "pdfs"):
        candidate_path = collection_dir / dir_candidate
        if candidate_path.exists():
            pdf_dir = candidate_path
            break

    if pdf_dir is None:
        logger.warning(f"No PDF directory found inside {collection_dir}. Expected one of: PDFs/, PDFS/, pdfs/")
        return

    if not challenge_input.exists():
        logger.warning(f"Input JSON not found in {collection_dir.name}; skipping collection")
        return

    logger.info(f"Processing {collection_dir.name}")

    with open(challenge_input, 'r', encoding='utf-8') as f:
        challenge_json = json.load(f)

    docs         = challenge_json.get('documents', [])
    persona_role = challenge_json.get('persona', {}).get('role', 'General User')
    task_text    = challenge_json.get('job_to_be_done', {}).get('task', 'Extract key information')

    try:
        task_vec = embed(f"{persona_role} {task_text}")
    except Exception as e:
        logger.warning(f"Embedding failed: {e}; continuing without vectors")
        task_vec = None

    outlines_output = []

    for doc in docs:
        fname = doc.get('filename', '')
        if not fname:
            continue

        pdf_path = pdf_dir / fname
        if not pdf_path.exists():
            logger.warning(f"PDF missing: {pdf_path}")
            continue

        logger.info(f"   → Parsing {fname}")
        parsed = PDFOutlineParser().extract_outline(pdf_path)

        outline = parsed.get('outline', [])
        if not outline:
            outline = _heuristic_headings(parsed.get('raw_text', []))

        outlines_output.append({
            'document': fname,
            'title': parsed.get('title', fname),
            'outline': outline,
            'raw_text': parsed.get('raw_text', [])
        })

    # write per-collection JSON
    out_file = collection_dir / 'challenge1b_outline_only.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'input_documents': [d.get('filename') for d in docs],
            'persona': persona_role,
            'job_to_be_done': task_text,
            'processing_timestamp': datetime.utcnow().isoformat(),
            'total_documents_processed': len(outlines_output),
            'outlines': outlines_output
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"    Saved outline_only to {out_file}")


def _heuristic_headings(raw_pages):
    """Generate headings if none were detected."""
    headings = []
    for page_obj in raw_pages:
        page = page_obj.get('page', 1)
        for line in page_obj.get('text', '').split('\n'):
            ls = line.strip()
            if 3 < len(ls) < 80 and (ls.isupper() or re.match(r'^\d+\.\s', ls) or re.match(r'^[A-Z][A-Za-z\s]{0,60}$', ls)):
                headings.append({'level': 'H1', 'text': ls, 'page': page})
    return headings


def process_challenge_1b():
    """Iterate over *all* Collection folders inside Challenge_1b."""
    base_dir = Path('Challenge_1b')
    if not base_dir.exists():
        logging.getLogger(__name__).info("Challenge_1b directory not found; skipping B-round processing")
        return

    collections = list(base_dir.glob('Collection*'))
    if not collections:
        logging.getLogger(__name__).info("No collections found in Challenge_1b; skipping B-round processing")
        return

    for collection_dir in collections:
        if collection_dir.is_dir():
            _process_collection(collection_dir)


def generate_refined_output():
    """Generate refined output (extracted_sections + subsection_analysis) for *all* Challenge-1B collections."""
    from app.outline_to_refined_processor import OutlineToRefinedProcessor
 
    logger = logging.getLogger(__name__)
    base_dir = Path('Challenge_1b')
 
    if not base_dir.exists():
        logger.info("Challenge_1b directory not found; skipping refined output generation")
        return
 
    processor = OutlineToRefinedProcessor()
 
    for collection_dir in sorted(base_dir.glob('Collection*')):
        outline_path = collection_dir / 'challenge1b_outline_only.json'
        refined_path = collection_dir / 'challenge1b_refined_output.json'
 
        if not outline_path.exists():
            logger.warning(f"Outline file not found: {outline_path}")
            continue
 
        try:
            refined_output = processor.generate_refined_output(outline_path, refined_path)
            logger.info(f"Refined output generated for {collection_dir.name}: {refined_path}")
            logger.info(f"   Sections: {len(refined_output['extracted_sections'])} | Analyses: {len(refined_output['subsection_analysis'])}")
        except Exception as e:
            logger.error(f"Error refining {collection_dir.name}: {e}")
            import traceback; logger.error(traceback.format_exc())


def main():
    """Main application entry point."""
    logger = setup_logging()
    
    # Define input and output directories - only these will be volumes
    input_dir = Path('/app/input')
    output_dir = Path('/app/output')
    
    logger.info("Starting PDF Outline Extractor")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process Challenge 1B first if it exists (baked into image)
    process_challenge_1b()
    
    # Generate refined output after outline processing
    generate_refined_output()
    
    # Validate directories
    if not validate_directories(input_dir, output_dir):
        logger.error("Directory validation failed")
        sys.exit(1)
    
    # Find PDF files
    pdf_files = find_pdf_files(input_dir)
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        # Don't exit here - Challenge 1B processing might have been successful
        logger.info("Processing complete - no additional PDFs to process")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process PDFs
    start_time = time.time()
    results = process_pdf_batch(pdf_files, output_dir)
    total_time = time.time() - start_time
    
    # Report results
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    avg_time = sum(time for _, _, time in results) / len(results) if results else 0
    
    logger.info("Processing complete!")
    logger.info(f"Successfully processed: {successful}/{len(pdf_files)} files")
    logger.info(f"Failed: {failed} files")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per file: {avg_time:.2f} seconds")
    
    # Performance check
    if avg_time > 10.0:
        logger.warning(f"Average processing time ({avg_time:.2f}s) exceeds 10-second target")
    else:
        logger.info("Performance target met!")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()