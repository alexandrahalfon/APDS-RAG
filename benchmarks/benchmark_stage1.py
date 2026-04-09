"""Benchmark Stage 1: Sequential vs parallel PDF ingestion.

Compares sequential, multiprocessing (Pool), and threading (ThreadPoolExecutor)
approaches. PDF reading is I/O-bound, so threading can outperform multiprocessing
by avoiding fork overhead while the GIL is released during I/O syscalls
(Lecture 09).
"""

import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E702
import torch  # noqa: E402, F401 — must load before pdfplumber (macOS segfault)

import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.doc_processing_local import process_pdf_complete_local
from optimized.stage1_ingestion.parallel_ingestion import parallel_ingest, process_single_pdf
from benchmarks.profiler import PipelineProfiler


def _sequential_ingest(pdf_paths: list) -> list:
    """Ingest PDFs sequentially (baseline).

    Args:
        pdf_paths: List of PDF file paths.

    Returns:
        Flat list of chunk dicts.
    """
    # List comprehension + enumerate replaces nested loop with manual counter
    return [
        {**c, 'id': i}
        for i, c in enumerate(
            chunk for path in pdf_paths for chunk in process_single_pdf(path)
        )
    ]


def _threaded_ingest(pdf_paths: List[str], num_workers: int) -> List[Dict]:
    """Ingest PDFs using ThreadPoolExecutor.

    Threading can be competitive with multiprocessing for I/O-bound work
    (PDF file reads) because the GIL is released during I/O syscalls.
    No fork overhead, no pickling cost, shared memory space.

    Args:
        pdf_paths: List of PDF file paths.
        num_workers: Number of threads.

    Returns:
        Flat list of chunk dicts with sequential IDs.
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_pdf, pdf_paths))

    return [
        {**chunk, 'id': i}
        for i, chunk in enumerate(chain.from_iterable(results))
    ]


def run_stage1_benchmark(pdf_folder: str) -> None:
    """Compare sequential, multiprocessing, and threading ingestion.

    Args:
        pdf_folder: Path to directory containing PDF files.
    """
    pdf_paths = sorted(str(p) for p in Path(pdf_folder).glob('*.pdf'))
    if not pdf_paths:
        print(f"⚠ No PDFs found in {pdf_folder}")
        return

    print(f"\n=== Stage 1 Benchmark: Ingestion ({len(pdf_paths)} PDFs) ===\n")
    profiler = PipelineProfiler()

    # Sequential baseline
    profiler.profile_stage("sequential", _sequential_ingest, pdf_paths)

    # Multiprocessing with varying workers
    for workers in [2, 4, 8]:
        if workers > len(pdf_paths):
            continue
        profiler.profile_stage(
            f"multiprocessing_{workers}w",
            parallel_ingest,
            pdf_paths,
            workers,
        )

    # Threading with varying workers — compare against multiprocessing
    # PDF parsing is partially I/O-bound (file reads) and partially
    # CPU-bound (text extraction). Threading avoids fork/pickle overhead
    # but is limited by the GIL for CPU-bound sections.
    for workers in [2, 4, 8]:
        if workers > len(pdf_paths):
            continue
        profiler.profile_stage(
            f"threading_{workers}w",
            _threaded_ingest,
            pdf_paths,
            workers,
        )

    profiler.save_results("stage1_results.json")
    profiler.print_summary()


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else './data/sample_pdfs'
    run_stage1_benchmark(folder)
