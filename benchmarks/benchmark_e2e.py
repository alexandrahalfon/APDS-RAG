"""End-to-end benchmark: baseline vs traditional-Python-optimized vs fully optimized.

Three-tier progression:
  1. Baseline          — sequential ingestion, one-at-a-time embedding, for-loop search
  2. Trad. Python opt  — same ingestion & embedding, but NumPy vectorized search
                         (isolates "better Python" gains from library/parallelism gains)
  3. Fully optimized   — parallel ingestion, batched embedding, FAISS search
"""

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

# Traditional Python optimization imports
from optimized.stage3_search.numpy_vectorized import (
    search_similar_vectorized,
    normalize_embeddings,
)

# Fully optimized imports
from optimized.stage1_ingestion.parallel_ingestion import parallel_ingest, process_single_pdf
from optimized.stage2_embedding.gpu_embedding import generate_embeddings_batched
from optimized.stage3_search.faiss_search import FAISSIndex


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _ingest_sequential(pdf_paths: list) -> list:
    """Ingest PDFs sequentially and return flat chunk list."""
    return [
        {**c, 'id': i}
        for i, c in enumerate(
            chunk for path in pdf_paths for chunk in process_single_pdf(path)
        )
    ]


def _embed_sequential(all_chunks: list) -> list:
    """Generate embeddings one-at-a-time (baseline)."""
    return generate_embeddings_baseline(all_chunks)


def _extract_embeddings(all_chunks: list) -> np.ndarray:
    """Pull embedding arrays out of chunk dicts into a float32 matrix."""
    return np.array([c['embedding'] for c in all_chunks], dtype=np.float32)


# ---------------------------------------------------------------------------
# Tier 1 — Baseline (Python for-loop search)
# ---------------------------------------------------------------------------

def _baseline_pipeline(pdf_paths: list, num_queries: int = 10) -> None:
    """Run the full baseline pipeline.

    Sequential ingestion, one-at-a-time embedding, Python for-loop search.

    Args:
        pdf_paths: List of PDF file paths.
        num_queries: Number of search queries to run.
    """
    all_chunks = _ingest_sequential(pdf_paths)
    if not all_chunks:
        return

    all_chunks = _embed_sequential(all_chunks)

    embeddings = _extract_embeddings(all_chunks)
    metadata = [{k: v for k, v in c.items() if k != 'embedding'} for c in all_chunks]
    indices = np.random.choice(len(embeddings), size=min(num_queries, len(embeddings)), replace=False)
    for idx in indices:
        search_similar_chunks(embeddings[idx], embeddings, metadata, top_k=10)


# ---------------------------------------------------------------------------
# Tier 2 — Traditional Python optimized (NumPy vectorized search)
# ---------------------------------------------------------------------------

def _trad_python_opt_pipeline(pdf_paths: list, num_queries: int = 10) -> None:
    """Same ingestion & embedding as baseline, but NumPy vectorized search.

    This isolates the gains from "writing better Python" (vectorized ops,
    float32, argpartition, pre-normalized embeddings) without any external
    libraries (Numba, FAISS) or parallelism (multiprocessing, GPU batching).

    Args:
        pdf_paths: List of PDF file paths.
        num_queries: Number of search queries to run.
    """
    all_chunks = _ingest_sequential(pdf_paths)
    if not all_chunks:
        return

    all_chunks = _embed_sequential(all_chunks)

    embeddings = _extract_embeddings(all_chunks)
    # Pre-normalize once — avoids redundant per-query norm computation
    embeddings_normed = normalize_embeddings(embeddings)

    indices = np.random.choice(len(embeddings), size=min(num_queries, len(embeddings)), replace=False)
    for idx in indices:
        search_similar_vectorized(embeddings[idx], embeddings_normed, top_k=10)


# ---------------------------------------------------------------------------
# Tier 3 — Fully optimized (parallel + batched + FAISS)
# ---------------------------------------------------------------------------

def _optimized_pipeline(pdf_paths: list, num_queries: int = 10) -> None:
    """Run the fully optimized pipeline.

    Parallel ingestion, batched embedding, FAISS index search.

    Args:
        pdf_paths: List of PDF file paths.
        num_queries: Number of search queries to run.
    """
    all_chunks = parallel_ingest(pdf_paths)
    if not all_chunks:
        return

    all_chunks = generate_embeddings_batched(all_chunks, batch_size=64, device='cpu')

    embeddings = _extract_embeddings(all_chunks)
    index = FAISSIndex(embeddings)
    indices = np.random.choice(len(embeddings), size=min(num_queries, len(embeddings)), replace=False)
    for idx in indices:
        index.search(embeddings[idx], top_k=10)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_e2e_benchmark(pdf_folder: str, num_queries: int = 10) -> None:
    """Run all three pipeline tiers and compare.

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

    # Tier 1 — Baseline
    profiler.profile_stage("baseline_e2e", _baseline_pipeline, pdf_paths, num_queries)

    # Tier 2 — Traditional Python optimization
    profiler.profile_stage("trad_python_opt_e2e", _trad_python_opt_pipeline, pdf_paths, num_queries)

    # Tier 3 — Fully optimized
    profiler.profile_stage("optimized_e2e", _optimized_pipeline, pdf_paths, num_queries)

    # Calculate speedups vs baseline
    baseline_time = profiler.results['baseline_e2e']['time_seconds']

    trad_time = profiler.results['trad_python_opt_e2e']['time_seconds']
    trad_speedup = baseline_time / trad_time if trad_time > 0 else float('inf')

    opt_time = profiler.results['optimized_e2e']['time_seconds']
    opt_speedup = baseline_time / opt_time if opt_time > 0 else float('inf')

    profiler.results['speedup_trad_python'] = {
        'time_seconds': round(trad_speedup, 2),
        'peak_memory_mb': 0.0,
    }
    profiler.results['speedup_fully_optimized'] = {
        'time_seconds': round(opt_speedup, 2),
        'peak_memory_mb': 0.0,
    }

    print(f"\n  ➤ Traditional Python opt speedup: {trad_speedup:.2f}x")
    print(f"  ➤ Fully optimized speedup:        {opt_speedup:.2f}x")

    profiler.save_results("e2e_results.json")
    profiler.print_summary()


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else './data/sample_pdfs'
    run_e2e_benchmark(folder)
