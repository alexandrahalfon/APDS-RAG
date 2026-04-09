"""Scaling benchmarks: measure how speedups change with problem size.

Produces three datasets for publication-ready figures:
  1. Search scaling   — vary vector count (500 → 50k), 4 search tiers
  2. Generation scaling — vary output length (32 → 256 tokens), 2 gen tiers
  3. Corpus scaling    — vary PDF count (5 → 50), full e2e 3-tier pipeline

All results saved to benchmarks/results/scaling_*.json.
"""

import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E702
import torch  # noqa: E402, F401 — must load before pdfplumber (macOS segfault)

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Search scaling imports
# ---------------------------------------------------------------------------
from baseline.similarity_search import search_similar_chunks
from optimized.stage3_search.numpy_vectorized import (
    search_similar_vectorized,
    normalize_embeddings,
)
from optimized.stage3_search.numba_search import search_similar_numba
from optimized.stage3_search.faiss_search import FAISSIndex


# ---------------------------------------------------------------------------
# 1. Search scaling: time vs number of vectors
# ---------------------------------------------------------------------------

def _time_search_method(func, *args, n_runs: int = 1) -> float:
    """Time a function over n_runs and return total seconds."""
    start = time.perf_counter()
    for _ in range(n_runs):
        func(*args)
    return time.perf_counter() - start


def run_search_scaling(
    vector_counts: List[int] = None,
    num_queries: int = 50,
    dim: int = 384,
) -> Dict:
    """Benchmark search methods across different corpus sizes.

    Args:
        vector_counts: List of corpus sizes to test.
        num_queries: Number of queries per run.
        dim: Embedding dimension.

    Returns:
        Dict of results keyed by method name.
    """
    if vector_counts is None:
        vector_counts = [500, 1000, 2000, 5000, 10000, 25000, 50000]

    rng = np.random.default_rng(42)
    results = {
        "vector_counts": vector_counts,
        "numpy_baseline": [],
        "numpy_vectorized": [],
        "numba_jit": [],
        "faiss_flat": [],
    }

    # Warm up Numba
    print("  Warming up Numba JIT...")
    _d = np.random.randn(10, dim).astype(np.float32)
    search_similar_numba(np.random.randn(dim).astype(np.float32), _d, 5)

    print(f"\n=== Search Scaling Benchmark ({num_queries} queries per size) ===\n")

    for n in vector_counts:
        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        queries = rng.standard_normal((num_queries, dim)).astype(np.float32)
        metadata = [{'id': i, 'page': 1, 'text': f'chunk {i}', 'word_count': 10}
                    for i in range(n)]
        embeddings_normed = normalize_embeddings(embeddings)
        faiss_index = FAISSIndex(embeddings.copy())

        print(f"  n={n:>6d}: ", end="", flush=True)

        # Baseline for-loop
        t = _time_search_method(
            lambda: [search_similar_chunks(q, embeddings, metadata, 10) for q in queries]
        )
        results["numpy_baseline"].append(round(t, 4))
        print(f"baseline={t:.2f}s  ", end="", flush=True)

        # NumPy vectorized
        t = _time_search_method(
            lambda: [search_similar_vectorized(q, embeddings_normed, 10) for q in queries]
        )
        results["numpy_vectorized"].append(round(t, 4))
        print(f"vectorized={t:.2f}s  ", end="", flush=True)

        # Numba JIT
        t = _time_search_method(
            lambda: [search_similar_numba(q, embeddings, 10) for q in queries]
        )
        results["numba_jit"].append(round(t, 4))
        print(f"numba={t:.2f}s  ", end="", flush=True)

        # FAISS
        t = _time_search_method(
            lambda: [faiss_index.search(q, 10) for q in queries]
        )
        results["faiss_flat"].append(round(t, 4))
        print(f"faiss={t:.2f}s")

    # Save
    out_path = Path("./benchmarks/results/scaling_search.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved {out_path}")

    return results


# ---------------------------------------------------------------------------
# 2. Generation scaling: time vs output token length
# ---------------------------------------------------------------------------

def run_generation_scaling(
    token_counts: List[int] = None,
    num_queries: int = 3,
) -> Dict:
    """Benchmark generation time vs output length.

    Args:
        token_counts: List of max_new_tokens values to test.
        num_queries: Number of queries per token count.

    Returns:
        Dict of results keyed by method name.
    """
    if token_counts is None:
        token_counts = [32, 64, 128, 256]

    from baseline.generation_step_local import generate_answer_baseline, get_generation_model
    from optimized.stage4_generation.optimized_generation import (
        generate_answer_optimized, _load_model, DEFAULT_MODEL,
    )

    results = {
        "token_counts": token_counts,
        "gen_float32_cpu": [],
        "gen_float16_cpu": [],
    }

    import torch
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        results["gen_float16_gpu"] = []

    # Pre-load models
    print("  Pre-loading generation models...")
    get_generation_model()
    _load_model(DEFAULT_MODEL, "float16", "cpu")
    if has_gpu:
        _load_model(DEFAULT_MODEL, "float16", "cuda")

    # Synthetic context
    context_chunks = [
        {"text": "The study demonstrates significant improvements in processing efficiency "
                 "through the application of vectorized operations and optimized data structures. "
                 "Results show consistent speedups across varied workloads.", "page": 1},
        {"text": "Experimental evaluation was conducted on datasets ranging from small to large scale. "
                 "The methodology ensures reproducibility through fixed random seeds and "
                 "standardized hardware configurations.", "page": 2},
    ]
    query = "What are the main findings of this study?"

    print(f"\n=== Generation Scaling Benchmark ({num_queries} queries per token count) ===\n")

    for max_tokens in token_counts:
        print(f"  max_tokens={max_tokens:>4d}: ", end="", flush=True)

        # float32 CPU
        start = time.perf_counter()
        for _ in range(num_queries):
            generate_answer_baseline(query, context_chunks, max_new_tokens=max_tokens)
        t = time.perf_counter() - start
        results["gen_float32_cpu"].append(round(t, 4))
        print(f"f32={t:.2f}s  ", end="", flush=True)

        # float16 CPU
        start = time.perf_counter()
        for _ in range(num_queries):
            generate_answer_optimized(
                query, context_chunks, max_new_tokens=max_tokens,
                optimization="float16", device="cpu",
            )
        t = time.perf_counter() - start
        results["gen_float16_cpu"].append(round(t, 4))
        print(f"f16_cpu={t:.2f}s  ", end="", flush=True)

        # float16 GPU
        if has_gpu:
            start = time.perf_counter()
            for _ in range(num_queries):
                generate_answer_optimized(
                    query, context_chunks, max_new_tokens=max_tokens,
                    optimization="float16", device="cuda",
                )
            t = time.perf_counter() - start
            results["gen_float16_gpu"].append(round(t, 4))
            print(f"f16_gpu={t:.2f}s", end="")

        print()

    out_path = Path("./benchmarks/results/scaling_generation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved {out_path}")

    return results


# ---------------------------------------------------------------------------
# 3. Corpus scaling: e2e time vs number of PDFs
# ---------------------------------------------------------------------------

def run_corpus_scaling(
    pdf_folder: str = "./data/sample_pdfs",
    corpus_sizes: List[int] = None,
    num_queries: int = 5,
) -> Dict:
    """Benchmark full pipeline at different corpus sizes.

    Uses subsets of the PDF folder to simulate different scales.

    Args:
        pdf_folder: Directory containing PDF files.
        corpus_sizes: List of PDF counts to test.
        num_queries: Number of queries per run.

    Returns:
        Dict of results keyed by pipeline tier.
    """
    if corpus_sizes is None:
        corpus_sizes = [5, 10, 25, 50]

    from baseline.embedding_step_local import generate_embeddings_baseline
    from baseline.similarity_search import search_similar_chunks as baseline_search
    from optimized.stage1_ingestion.parallel_ingestion import parallel_ingest, process_single_pdf
    from optimized.stage2_embedding.gpu_embedding import generate_embeddings_batched
    from optimized.stage3_search.faiss_search import FAISSIndex
    from optimized.stage3_search.numpy_vectorized import (
        search_similar_vectorized as vectorized_search,
        normalize_embeddings,
    )

    all_pdfs = sorted(str(p) for p in Path(pdf_folder).glob("*.pdf"))
    if not all_pdfs:
        print(f"⚠ No PDFs found in {pdf_folder}")
        return {}

    max_available = len(all_pdfs)
    corpus_sizes = [s for s in corpus_sizes if s <= max_available]
    if not corpus_sizes:
        print(f"⚠ Not enough PDFs ({max_available}) for requested sizes")
        return {}

    results = {
        "corpus_sizes": corpus_sizes,
        "baseline": [],
        "trad_python_opt": [],
        "fully_optimized": [],
    }

    print(f"\n=== Corpus Scaling Benchmark ({max_available} PDFs available, {num_queries} queries) ===\n")

    # Pre-load torch/sentence-transformers before pdfplumber runs.
    # On macOS x86_64, loading torch after pdfplumber's native libs segfaults.
    from baseline.embedding_step_local import get_model
    get_model()

    for n_pdfs in corpus_sizes:
        pdf_subset = all_pdfs[:n_pdfs]
        print(f"  {n_pdfs} PDFs:")

        # --- Baseline: sequential ingest + sequential embed + for-loop search ---
        start = time.perf_counter()
        chunks = [
            {**c, 'id': i}
            for i, c in enumerate(
                chunk for path in pdf_subset for chunk in process_single_pdf(path)
            )
        ]
        if chunks:
            chunks = generate_embeddings_baseline(chunks)
            embs = np.array([c['embedding'] for c in chunks], dtype=np.float32)
            meta = [{k: v for k, v in c.items() if k != 'embedding'} for c in chunks]
            idxs = np.random.choice(len(embs), size=min(num_queries, len(embs)), replace=False)
            for idx in idxs:
                baseline_search(embs[idx], embs, meta, top_k=5)
        t_baseline = time.perf_counter() - start
        results["baseline"].append(round(t_baseline, 4))
        print(f"    baseline={t_baseline:.2f}s ({len(chunks)} chunks)")

        # --- Trad Python opt: sequential ingest + sequential embed + vectorized search ---
        start = time.perf_counter()
        chunks = [
            {**c, 'id': i}
            for i, c in enumerate(
                chunk for path in pdf_subset for chunk in process_single_pdf(path)
            )
        ]
        if chunks:
            chunks = generate_embeddings_baseline(chunks)
            embs = np.array([c['embedding'] for c in chunks], dtype=np.float32)
            embs_normed = normalize_embeddings(embs)
            idxs = np.random.choice(len(embs), size=min(num_queries, len(embs)), replace=False)
            for idx in idxs:
                vectorized_search(embs[idx], embs_normed, top_k=5)
        t_trad = time.perf_counter() - start
        results["trad_python_opt"].append(round(t_trad, 4))
        print(f"    trad_opt={t_trad:.2f}s")

        # --- Fully optimized: parallel ingest + batched embed + FAISS ---
        start = time.perf_counter()
        try:
            chunks = parallel_ingest(pdf_subset)
        except Exception as e:
            print(f"    ⚠ parallel_ingest failed ({e}), falling back to sequential")
            chunks = [
                {**c, 'id': i}
                for i, c in enumerate(
                    chunk for path in pdf_subset for chunk in process_single_pdf(path)
                )
            ]
        if chunks:
            import torch as _torch
            _embed_dev = "cuda" if _torch.cuda.is_available() else "cpu"
            chunks = generate_embeddings_batched(chunks, batch_size=64, device=_embed_dev)
            embs = np.array([c['embedding'] for c in chunks], dtype=np.float32)
            index = FAISSIndex(embs)
            idxs = np.random.choice(len(embs), size=min(num_queries, len(embs)), replace=False)
            for idx in idxs:
                index.search(embs[idx], top_k=5)
        t_opt = time.perf_counter() - start
        results["fully_optimized"].append(round(t_opt, 4))

        speedup_trad = t_baseline / t_trad if t_trad > 0 else 0
        speedup_opt = t_baseline / t_opt if t_opt > 0 else 0
        print(f"    optimized={t_opt:.2f}s  (trad={speedup_trad:.1f}x, full={speedup_opt:.1f}x)\n")

    out_path = Path("./benchmarks/results/scaling_corpus.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved {out_path}")

    return results


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def run_all_scaling_benchmarks(pdf_folder: str = "./data/sample_pdfs") -> None:
    """Run all three scaling benchmarks.

    Args:
        pdf_folder: Directory containing PDF files.
    """
    print("=" * 70)
    print("  SCALING BENCHMARKS")
    print("=" * 70)

    # 1. Search scaling (no PDFs needed — synthetic data)
    run_search_scaling()

    # 2. Generation scaling (no PDFs needed — synthetic context)
    run_generation_scaling()

    # 3. Corpus scaling (needs PDFs)
    run_corpus_scaling(pdf_folder)

    print("\n" + "=" * 70)
    print("  All scaling benchmarks complete.")
    print("  Results in: benchmarks/results/scaling_*.json")
    print("  Run `python benchmarks/visualize.py` to generate figures.")
    print("=" * 70)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./data/sample_pdfs"

    # Allow running individual benchmarks
    if len(sys.argv) > 2:
        which = sys.argv[2]
        if which == "search":
            run_search_scaling()
        elif which == "generation":
            run_generation_scaling()
        elif which == "corpus":
            run_corpus_scaling(folder)
        else:
            print(f"Unknown benchmark: {which}. Use: search, generation, corpus")
    else:
        run_all_scaling_benchmarks(folder)
