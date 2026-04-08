"""Parallel PDF ingestion using multiprocessing."""

import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baseline.doc_processing_local import process_pdf_complete_local


def process_single_pdf(pdf_path: str) -> List[Dict]:
    """Process one PDF and return its chunks.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of chunk dicts with id, page, text, word_count, source_file.
    """
    try:
        doc = process_pdf_complete_local(pdf_path)
    except Exception as e:
        print(f"  ⚠ Error processing {Path(pdf_path).name}: {e}")
        return []

    # List comprehension replaces nested for-loop (less overhead, single expression)
    chunks: List[Dict] = [
        {
            'page': page['page_number'],
            'text': para['text'],
            'word_count': para['word_count'],
            'source_file': doc['file_name'],
        }
        for page in doc['pages']
        for para in page['paragraphs']
    ]

    return chunks


def parallel_ingest(pdf_paths: List[str], num_workers: Optional[int] = None) -> List[Dict]:
    """Ingest PDFs in parallel using multiprocessing.

    Args:
        pdf_paths: List of paths to PDF files.
        num_workers: Number of worker processes. Defaults to cpu_count().

    Returns:
        Flat list of chunk dicts with sequential IDs assigned.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Don't use more workers than files
    num_workers = min(num_workers, len(pdf_paths))

    print(f"Processing {len(pdf_paths)} PDFs with {num_workers} workers")

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_single_pdf, pdf_paths)

    # Flatten nested results and assign sequential IDs in one pass
    # List comprehension + enumerate replaces nested for-loop with manual counter
    all_chunks: List[Dict] = [
        {**chunk, 'id': i}
        for i, chunk in enumerate(
            chunk for chunk_list in results for chunk in chunk_list
        )
    ]

    print(f"✓ {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
    return all_chunks
