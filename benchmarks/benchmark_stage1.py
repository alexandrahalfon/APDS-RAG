"""Benchmark Stage 1: PDF ingestion methods.

Compares:
  1. pdfplumber sequential   — baseline (Python layout engine)
  2. PyMuPDF sequential      — C library, no parallelism
  3. PyMuPDF + multiprocessing — C library + parallel workers
  4. PyMuPDF + threading     — C library + threads (GIL released during I/O)
"""

import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E702
import torch  # noqa: E402, F401 — must load before pdfplumber (macOS segfault)

import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.doc_processing_local import process_pdf_complete_local
from optimized.stage1_ingestion.pymupdf_ingestion import process_pdf_pymupdf
from optimized.stage1_ingestion.parallel_ingestion import parallel_ingest, process_single_pdf
from benchmarks.profiler import PipelineProfiler


def _chunks_from_doc(doc: dict, source_file: str) -> List[Dict]:
    """Extract chunk dicts from a processed document."""
    return [
        {
            'page': page['page_number'],
            'text': para['text'],
            'word_count': para['word_count'],
            'source_file': source_file,
        }
        for page in doc['pages']
        for para in page['paragraphs']
    ]


def _pdfplumber_sequential(pdf_paths: list) -> list:
    """Ingest PDFs sequentially using pdfplumber (baseline).

    Args:
        pdf_paths: List of PDF file paths.

    Returns:
        Flat list of chunk dicts.
    """
    all_chunks = []
    for path in pdf_paths:
        try:
            doc = process_pdf_complete_local(path)
            all_chunks.extend(_chunks_from_doc(doc, doc['file_name']))
        except Exception as e:
            print(f"  ⚠ Error: {Path(path).name}: {e}")
    return [{**c, 'id': i} for i, c in enumerate(all_chunks)]


def _pymupdf_sequential(pdf_paths: list) -> list:
    """Ingest PDFs sequentially using PyMuPDF (C library, no parallelism).

    Args:
        pdf_paths: List of PDF file paths.

    Returns:
        Flat list of chunk dicts.
    """
    all_chunks = []
    for path in pdf_paths:
        try:
            doc = process_pdf_pymupdf(path)
            all_chunks.extend(_chunks_from_doc(doc, doc['file_name']))
        except Exception as e:
            print(f"  ⚠ Error: {Path(path).name}: {e}")
    return [{**c, 'id': i} for i, c in enumerate(all_chunks)]


def _threaded_ingest(pdf_paths: List[str], num_workers: int) -> List[Dict]:
    """Ingest PDFs using PyMuPDF + ThreadPoolExecutor.

    Threading can be competitive with multiprocessing for I/O-bound work
    because the GIL is released during I/O syscalls. No fork overhead,
    no pickling cost, shared memory space.

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
    """Compare PDF parsing libraries and parallelization strategies.

    Args:
        pdf_folder: Path to directory containing PDF files.
    """
    pdf_paths = sorted(str(p) for p in Path(pdf_folder).glob('*.pdf'))
    if not pdf_paths:
        print(f"⚠ No PDFs found in {pdf_folder}")
        return

    print(f"\n=== Stage 1 Benchmark: Ingestion ({len(pdf_paths)} PDFs) ===\n")
    profiler = PipelineProfiler()

    # Tier 1: pdfplumber sequential (Python layout engine — baseline)
    profiler.profile_stage("pdfplumber_sequential", _pdfplumber_sequential, pdf_paths)

    # Tier 2: PyMuPDF sequential (C library, no parallelism)
    profiler.profile_stage("pymupdf_sequential", _pymupdf_sequential, pdf_paths)

    # Tier 3: PyMuPDF + multiprocessing
    for workers in [2, 4, 8]:
        if workers > len(pdf_paths):
            continue
        profiler.profile_stage(
            f"pymupdf_mp_{workers}w",
            parallel_ingest,
            pdf_paths,
            workers,
        )

    # Tier 4: PyMuPDF + threading
    for workers in [2, 4, 8]:
        if workers > len(pdf_paths):
            continue
        profiler.profile_stage(
            f"pymupdf_thread_{workers}w",
            _threaded_ingest,
            pdf_paths,
            workers,
        )

    profiler.save_results("stage1_results.json")
    profiler.print_summary()


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else './data/sample_pdfs'
    run_stage1_benchmark(folder)
