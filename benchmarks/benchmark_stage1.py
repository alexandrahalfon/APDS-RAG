"""Benchmark Stage 1: Sequential vs parallel PDF ingestion."""

import sys
import json
import time
from pathlib import Path

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
    all_chunks = []
    chunk_id = 0
    for path in pdf_paths:
        chunks = process_single_pdf(path)
        for c in chunks:
            c['id'] = chunk_id
            all_chunks.append(c)
            chunk_id += 1
    return all_chunks


def run_stage1_benchmark(pdf_folder: str) -> None:
    """Compare sequential vs parallel ingestion with varying worker counts.

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

    # Parallel with varying workers
    for workers in [2, 4, 8]:
        if workers > len(pdf_paths):
            continue
        profiler.profile_stage(
            f"parallel_{workers}w",
            parallel_ingest,
            pdf_paths,
            workers,
        )

    profiler.save_results("stage1_results.json")
    profiler.print_summary()


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else './data/sample_pdfs'
    run_stage1_benchmark(folder)
