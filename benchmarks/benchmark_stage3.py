"""Benchmark Stage 3: Vector search methods (NumPy vs Numba vs FAISS)."""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.similarity_search import cosine_similarity, search_similar_chunks
from optimized.stage3_search.numba_search import search_similar_numba
from optimized.stage3_search.faiss_search import FAISSIndex
from benchmarks.profiler import PipelineProfiler


def _generate_random_data(n_vectors: int = 1000, dim: int = 384):
    """Generate random embeddings and queries for benchmarking.

    Args:
        n_vectors: Number of document embeddings.
        dim: Embedding dimension (384 for all-MiniLM-L6-v2).

    Returns:
        Tuple of (embeddings, query_embeddings, metadata).
    """
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    queries = rng.standard_normal((100, dim)).astype(np.float32)
    metadata = [{'id': i, 'page': 1, 'text': f'chunk {i}', 'word_count': 10} for i in range(n_vectors)]
    return embeddings, queries, metadata


def _run_numpy_search(embeddings, queries, metadata, top_k=10):
    """Baseline NumPy search."""
    for q in queries:
        search_similar_chunks(q, embeddings, metadata, top_k)


def _run_numba_search(embeddings, queries, top_k=10):
    """Numba JIT parallel search."""
    emb_f64 = embeddings.astype(np.float64)
    for q in queries:
        search_similar_numba(q.astype(np.float64), emb_f64, top_k)


def _run_faiss_search(embeddings, queries, top_k=10):
    """FAISS index search."""
    index = FAISSIndex(embeddings.copy())
    for q in queries:
        index.search(q, top_k)


def run_stage3_benchmark(n_vectors: int = 1000) -> None:
    """Compare search methods over 100 queries.

    Args:
        n_vectors: Number of document embeddings to search over.
    """
    embeddings, queries, metadata = _generate_random_data(n_vectors)

    print(f"\n=== Stage 3 Benchmark: Search ({n_vectors} vectors, {len(queries)} queries) ===\n")
    profiler = PipelineProfiler()

    # Warm up Numba JIT
    print("  Warming up Numba JIT...")
    _dummy_emb = np.random.randn(10, 384).astype(np.float64)
    search_similar_numba(np.random.randn(384).astype(np.float64), _dummy_emb, 5)

    # NumPy baseline
    profiler.profile_stage("numpy_baseline", _run_numpy_search, embeddings, queries, metadata)

    # Numba JIT
    profiler.profile_stage("numba_jit", _run_numba_search, embeddings, queries)

    # FAISS
    profiler.profile_stage("faiss_flat", _run_faiss_search, embeddings, queries)

    profiler.save_results("stage3_results.json")
    profiler.print_summary()


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_stage3_benchmark(n)
