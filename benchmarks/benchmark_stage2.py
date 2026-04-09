"""Benchmark Stage 2: Embedding generation methods and batch sizes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmarks._preload  # noqa: F401 — force torch before pdfplumber

import copy
import json

from baseline.embedding_step_local import generate_embeddings_baseline
from optimized.stage2_embedding.gpu_embedding import (
    generate_embeddings_batched,
    generate_embeddings_gpu_amp,
)
from benchmarks.profiler import PipelineProfiler


def _make_dummy_chunks(n: int = 200) -> list:
    """Create dummy chunks for benchmarking.

    Args:
        n: Number of chunks to generate.

    Returns:
        List of chunk dicts.
    """
    # List comprehension instead of for-loop + append
    return [
        {
            'id': i,
            'page': 1,
            'text': f"This is sample text for benchmark chunk number {i}. "
                    f"It contains enough words to simulate a real paragraph "
                    f"from a PDF document that has been processed and chunked.",
            'word_count': 25,
        }
        for i in range(n)
    ]


def run_stage2_benchmark(pdf_folder: str = None, num_chunks: int = 200) -> None:
    """Compare embedding methods across batch sizes.

    Args:
        pdf_folder: Optional path to PDF folder for real chunks.
        num_chunks: Number of dummy chunks if no PDF folder provided.
    """
    # Try to load real chunks from PDFs, fall back to dummy
    # Pre-load torch/sentence-transformers before pdfplumber runs.
    from baseline.embedding_step_local import get_model
    get_model()

    if pdf_folder:
        from optimized.stage1_ingestion.parallel_ingestion import parallel_ingest
        pdf_paths = sorted(str(p) for p in Path(pdf_folder).glob('*.pdf'))
        if pdf_paths:
            base_chunks = parallel_ingest(pdf_paths)
        else:
            base_chunks = _make_dummy_chunks(num_chunks)
    else:
        base_chunks = _make_dummy_chunks(num_chunks)

    print(f"\n=== Stage 2 Benchmark: Embeddings ({len(base_chunks)} chunks) ===\n")
    profiler = PipelineProfiler()

    # CPU sequential baseline
    profiler.profile_stage(
        "cpu_sequential",
        generate_embeddings_baseline,
        copy.deepcopy(base_chunks),
    )

    # CPU batched with varying batch sizes
    for bs in [16, 32, 64, 128]:
        profiler.profile_stage(
            f"cpu_batched_bs{bs}",
            generate_embeddings_batched,
            copy.deepcopy(base_chunks),
            bs,
            'cpu',
        )

    # GPU batched (will fall back to CPU if no CUDA)
    import torch
    if torch.cuda.is_available():
        for bs in [32, 64, 128]:
            profiler.profile_stage(
                f"gpu_batched_bs{bs}",
                generate_embeddings_batched,
                copy.deepcopy(base_chunks),
                bs,
                'cuda',
            )

        # GPU + AMP
        profiler.profile_stage(
            "gpu_amp_bs64",
            generate_embeddings_gpu_amp,
            copy.deepcopy(base_chunks),
            64,
        )

    profiler.save_results("stage2_results.json")
    profiler.print_summary()


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    run_stage2_benchmark(folder)
