"""End-to-end benchmark: baseline vs traditional-Python-optimized vs fully optimized.

Three-tier progression (now includes generation — Stage 4):
  1. Baseline          — sequential ingest, one-at-a-time embed, for-loop search,
                         float32 generation
  2. Trad. Python opt  — same ingest & embed, NumPy vectorized search,
                         float16 generation (dtype optimization)
  3. Fully optimized   — parallel ingest, batched embed, FAISS search,
                         float16 generation (+ GPU if available)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmarks._preload  # noqa: F401 — force torch before pdfplumber

import json
import numpy as np

from benchmarks.profiler import PipelineProfiler

# Baseline imports
from baseline.doc_processing_local import process_pdf_complete_local
from baseline.embedding_step_local import generate_embeddings_baseline
from baseline.similarity_search import search_similar_chunks
from baseline.generation_step_local import generate_answer_baseline

# Traditional Python optimization imports
from optimized.stage3_search.numpy_vectorized import (
    search_similar_vectorized,
    normalize_embeddings,
)

# Fully optimized imports
from optimized.stage1_ingestion.parallel_ingestion import parallel_ingest, process_single_pdf
from optimized.stage2_embedding.gpu_embedding import generate_embeddings_batched
from optimized.stage3_search.faiss_search import FAISSIndex
from optimized.stage4_generation.optimized_generation import generate_answer_optimized


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SAMPLE_QUERIES = [
    "What are the main findings?",
    "Summarize the methodology used.",
    "What conclusions were drawn?",
    "What data was used in the study?",
    "What are the limitations discussed?",
    "How does this compare to previous work?",
    "What future work is suggested?",
    "What is the significance of these results?",
    "Describe the experimental setup.",
    "What are the key contributions?",
]


def _get_sample_queries(n: int) -> list:
    """Return n sample query strings for generation benchmarking."""
    return [_SAMPLE_QUERIES[i % len(_SAMPLE_QUERIES)] for i in range(n)]


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
# Tier 1 — Baseline
# ---------------------------------------------------------------------------

def _baseline_pipeline(pdf_paths: list, num_queries: int = 10) -> None:
    """Full baseline pipeline: sequential everything, float32 generation.

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

    sample_queries = _get_sample_queries(len(indices))
    for i, idx in enumerate(indices):
        # Stage 3: search
        results = search_similar_chunks(embeddings[idx], embeddings, metadata, top_k=5)
        # Stage 4: generate (float32 baseline)
        generate_answer_baseline(sample_queries[i], results[:3], max_new_tokens=64)


# ---------------------------------------------------------------------------
# Tier 2 — Traditional Python optimized
# ---------------------------------------------------------------------------

def _trad_python_opt_pipeline(pdf_paths: list, num_queries: int = 10) -> None:
    """Same ingest & embed as baseline; vectorized search + float16 generation.

    Isolates gains from "writing better Python" (vectorized ops, float32
    arrays, argpartition, dtype optimization) without external libraries
    or parallelism.

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
    embeddings_normed = normalize_embeddings(embeddings)

    indices = np.random.choice(len(embeddings), size=min(num_queries, len(embeddings)), replace=False)

    sample_queries = _get_sample_queries(len(indices))
    for i, idx in enumerate(indices):
        # Stage 3: vectorized search
        top_indices, _ = search_similar_vectorized(embeddings[idx], embeddings_normed, top_k=5)
        context_chunks = [metadata[j] for j in top_indices]
        # Stage 4: generate (float16 — dtype optimization)
        generate_answer_optimized(
            sample_queries[i], context_chunks,
            max_new_tokens=64, optimization="float16", device="cpu",
        )


# ---------------------------------------------------------------------------
# Tier 3 — Fully optimized
# ---------------------------------------------------------------------------

def _optimized_pipeline(pdf_paths: list, num_queries: int = 10) -> None:
    """Parallel ingest, batched embed, FAISS search, float16 generation.

    Args:
        pdf_paths: List of PDF file paths.
        num_queries: Number of search queries to run.
    """
    import torch

    all_chunks = parallel_ingest(pdf_paths)
    if not all_chunks:
        return

    all_chunks = generate_embeddings_batched(all_chunks, batch_size=64, device='cpu')

    embeddings = _extract_embeddings(all_chunks)
    metadata = [{k: v for k, v in c.items() if k != 'embedding'} for c in all_chunks]
    index = FAISSIndex(embeddings)

    indices = np.random.choice(len(embeddings), size=min(num_queries, len(embeddings)), replace=False)
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_queries = _get_sample_queries(len(indices))
    for i, idx in enumerate(indices):
        # Stage 3: FAISS search
        top_indices, _ = index.search(embeddings[idx], top_k=5)
        context_chunks = [metadata[j] for j in top_indices if j >= 0]
        # Stage 4: generate (float16, best available device)
        generate_answer_optimized(
            sample_queries[i], context_chunks,
            max_new_tokens=64, optimization="float16", device=gen_device,
        )


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

    # Pre-load torch/sentence-transformers before pdfplumber runs.
    # On macOS x86_64, loading torch's native libs after pdfplumber's
    # cryptography/cffi libs causes a segfault.
    from baseline.embedding_step_local import get_model
    get_model()

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
