#!/usr/bin/env python3
"""
main2.py – minimal Round-1A runner
----------------------------------
• Scans ./input/ (or /app/input in Docker)
• Uses existing DocumentPipeline to parse each PDF
• Writes pretty-printed JSON into ./output/ (or /app/output)
• Does **not** run Challenge-1B logic so it is isolated from the larger pipeline.
"""

import logging
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

from pipeline import DocumentPipeline
from output_writer import OutputWriter

###############################################################################
# Logging setup
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("main2.log")
    ]
)
LOGGER = logging.getLogger(__name__)

###############################################################################
# Helpers
###############################################################################

def _save_output(data: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, indent=2, ensure_ascii=False)


def _process_single(pdf_path: Path, out_dir: Path) -> Tuple[str, bool, float]:
    start = time.time()
    try:
        parsed = DocumentPipeline().process(pdf_path)
        out_file = out_dir / f"{pdf_path.stem}.json"
        OutputWriter().write_outline(parsed, out_file)
        elapsed = time.time() - start
        LOGGER.info(f"✔ Parsed {pdf_path.name} in {elapsed:.2f}s → {out_file}")
        return pdf_path.name, True, elapsed
    except Exception as exc:
        elapsed = time.time() - start
        LOGGER.error(f"✗ Failed {pdf_path.name}: {exc}")
        return pdf_path.name, False, elapsed


def _find_pdfs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf"])

###############################################################################
# Main
###############################################################################

def main():
    # Resolve directories (Docker vs local)
    in_dir  = Path("/app/input") if Path("/app/input").exists() else Path("input")
    out_dir = Path("/app/output") if Path("/app/input").exists() else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Input : {in_dir.resolve()}")
    LOGGER.info(f"Output: {out_dir.resolve()}")

    pdfs = _find_pdfs(in_dir)
    if not pdfs:
        LOGGER.warning("No PDF files found – nothing to do.")
        return

    LOGGER.info(f"Found {len(pdfs)} PDF files")

    # Run in parallel (max 8 procs)
    procs = min(cpu_count(), 8, len(pdfs))
    if procs == 1:
        results = [_process_single(p, out_dir) for p in pdfs]
    else:
        with Pool(processes=procs) as pool:
            results = pool.starmap(_process_single, [(p, out_dir) for p in pdfs])

    ok = sum(1 for _, success, _ in results if success)
    fail = len(results) - ok
    LOGGER.info(f"Finished. Success: {ok} | Failed: {fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main() 