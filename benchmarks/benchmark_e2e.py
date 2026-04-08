"""End-to-end benchmark: baseline vs fully optimized pipeline."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.profiler import PipelineProfiler

# Baseline imports
from baseline.doc_processing_local import process_pdf_complete_local
from baseline.embedding_step_local import generate_embeddings_baseline
from baseline.similarity_search import search_similar_chunks

# Optimized imports
from optimized.stage1_ingestion.parallel_ingestion import parallel_ingest, process_single_pdf
from optimized.stage2_embedding.gpu_embedding import generate_embeddings_batched
from optimized.stage3_search.faiss_search import FAISSIndex


def _baseline_pipeline(pdf_paths: list, num_queries: int = 10) -> None:
    """Run the full baseline pipeline.

    Args:
        pdf_paths: List of PDF file paths.
        num_queries: Number of search queries to run.
    """
    # Ingest sequentially — list comprehension with enumerate for ID assignment
    all_chunks = [
        {**c, 'id': i}
        for i, c in enumerate(
            chunk for path in pdf_paths for chunk in process_single_pdf(path)
        )
    ]

    if not all_chunks:
        return

    # Embed sequentially
    all_chunks = generate_embeddings_baseline(all_chunks)

    # Search with NumPy — explicit float32 dtype
    embeddings = np.array([c['embedding'] for c in all_chunks], dtype=np.float32)
    metadata = [{k: v for k, v in c.items() if k != 'embedding'} for c in all_chunks]
    indices = np.random.choice(len(embeddings), size=min(num_queries, len(embeddings)), replace=False)
    for idx in indices:
        search_similar_chunks(embeddings[idx], embeddings, metadata, top_k=10)


def _optimized_pipeline(pdf_paths: list, num_queries: int = 10) -> None:
    """Run the fully optimized pipeline.

    Args:
        pdf_paths: List of PDF file paths.
        num_queries: Number of search queries to run.
    """
    # Parallel ingest
    all_chunks = parallel_ingest(pdf_paths)
    if not all_chunks:
        return

    # Batched embed
    all_chunks = generate_embeddings_batched(all_chunks, batch_size=64, device='cpu')

    # FAISS search
    embeddings = np.array([c['embedding'] for c in all_chunks], dtype=np.float32)
    index = FAISSIndex(embeddings)
    indices = np.random.choice(len(embeddings), size=min(num_queries, len(embeddings)), replace=False)
    for idx in indices:
        index.search(embeddings[idx], top_k=10)


def run_e2e_benchmark(pdf_folder: str, num_queries: int = 10) -> None:
    """Run baseline and optimized pipelines and compare.

    Args:
        pdf_folder: Path to directory containing PDF files.
        num_queries: Number of search queries to benchmark.
    """
    pdf_paths = sorted(str(p) for p in Path(pdf_folder).glob('*.pdf'))
    if not pdf_paths:
        print(f"⚠ No PDFs found in {pdf_folder}")
        return

    print(f"\n=== End-to-End Benchmark ({len(pdf_paths)} PDFs, {num_queries} queries) ===\n")
    profiler = PipelineProfiler()

    profiler.profile_stage("baseline_e2e", _baseline_pipeline, pdf_paths, num_queries)
    profiler.profile_stage("optimized_e2e", _optimized_pipeline, pdf_paths, num_queries)

    # Calculate speedup
    baseline_time = profiler.results['baseline_e2e']['time_seconds']
    optimized_time = profiler.results['optimized_e2e']['time_seconds']
    speedup = baseline_time / optimized_time if optimized_time > 0 else float('inf')

    profiler.results['speedup'] = {
        'time_seconds': round(speedup, 2),
        'peak_memory_mb': 0.0,
    }

    print(f"\n  ➤ Total speedup: {speedup:.2f}x")

    profiler.save_results("e2e_results.json")
    profiler.print_summary()


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else './data/sample_pdfs'
    run_e2e_benchmark(folder)
