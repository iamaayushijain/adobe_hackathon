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
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import json
import io


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
    
    try:
        # Parse PDF via pipeline
        pipeline = DocumentPipeline()
        outline_data = pipeline.process(pdf_path)
        save_to_json(
        outline_data,
        f"output2/{Path(pdf_path).stem}.json") # or f"output/{Path(file_path).stem}.json")
        
        # Write JSON output
        output_filename = pdf_path.stem + '.json'
        output_path = output_dir / output_filename
        
        writer = OutputWriter()
        writer.write_outline(outline_data, output_path)
        
        processing_time = time.time() - start_time
        return filename, True, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"Failed to process {filename}: {str(e)}")
        return filename, False, processing_time

def save_to_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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


def main():
    """Main application entry point."""
    logger = setup_logging()
    
    # Define input and output directories
    input_dir = Path('/app/input')
    output_dir = Path('/app/output')
    
    # For development, fall back to local directories if /app doesn't exist
    if not input_dir.exists():
        input_dir = Path('./input')
        output_dir = Path('./output')
        logger.warning("Using local input/output directories for development")
    
    logger.info("Starting PDF Outline Extractor")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Validate directories
    if not validate_directories(input_dir, output_dir):
        logger.error("Directory validation failed")
        sys.exit(1)
    
    # Find PDF files
    pdf_files = find_pdf_files(input_dir)
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process PDFs
    start_time = time.time()
    results = process_pdf_batch(pdf_files, output_dir)
    print(results)
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