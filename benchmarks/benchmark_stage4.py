"""Benchmark Stage 4: Answer generation methods.

Compares:
  1. Baseline    — float32, CPU, torch.no_grad
  2. float16     — CPU (traditional dtype optimization)
  3. float16     — GPU (if available)
  4. 4-bit quant — GPU (if available, requires bitsandbytes)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.generation_step_local import generate_answer_baseline, get_generation_model
from optimized.stage4_generation.optimized_generation import (
    generate_answer_optimized,
    _load_model,
    DEFAULT_MODEL,
)
from benchmarks.profiler import PipelineProfiler


def _make_synthetic_queries(n: int = 5) -> list:
    """Create synthetic query + context pairs for benchmarking.

    Args:
        n: Number of query/context pairs.

    Returns:
        List of (query_str, context_chunks) tuples.
    """
    rng = np.random.default_rng(42)
    queries = []
    for i in range(n):
        query = f"What are the main findings described in section {i + 1}?"
        context_chunks = [
            {
                "text": (
                    f"In section {i+1}, the researchers found that the proposed method "
                    f"achieves a {rng.uniform(5, 30):.1f}% improvement over the baseline. "
                    f"The key innovation involves a novel approach to data processing that "
                    f"reduces computational overhead while maintaining accuracy."
                ),
                "page": i + 1,
            },
            {
                "text": (
                    f"The experimental setup uses standard evaluation metrics including "
                    f"precision, recall, and F1 score. Results are averaged over "
                    f"{rng.integers(3, 10)} independent runs with different random seeds."
                ),
                "page": i + 2,
            },
        ]
        queries.append((query, context_chunks))
    return queries


def _run_baseline_generation(queries: list, max_new_tokens: int = 64) -> None:
    """Baseline float32 generation."""
    for query, chunks in queries:
        generate_answer_baseline(query, chunks, max_new_tokens=max_new_tokens)


def _run_optimized_generation(
    queries: list, optimization: str, device: str, max_new_tokens: int = 64,
) -> None:
    """Optimized generation with specified settings."""
    for query, chunks in queries:
        generate_answer_optimized(
            query, chunks,
            max_new_tokens=max_new_tokens,
            optimization=optimization,
            device=device,
        )


def run_stage4_benchmark(num_queries: int = 5) -> None:
    """Compare generation methods.

    Args:
        num_queries: Number of queries to benchmark.
    """
    queries = _make_synthetic_queries(num_queries)
    max_tokens = 64  # Short answers for faster benchmarking

    print(f"\n=== Stage 4 Benchmark: Generation ({num_queries} queries, {max_tokens} max tokens) ===\n")

    # Pre-load models so loading time doesn't skew generation benchmarks
    print("  Pre-loading models...")
    get_generation_model()
    _load_model(DEFAULT_MODEL, "float16", "cpu")

    import torch
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        _load_model(DEFAULT_MODEL, "float16", "cuda")
        try:
            _load_model(DEFAULT_MODEL, "4bit", "cuda")
            has_4bit = True
        except Exception:
            has_4bit = False
            print("  ⚠ 4-bit model unavailable (bitsandbytes not installed)")
    else:
        has_4bit = False

    print()
    profiler = PipelineProfiler()

    # Tier 1 — Baseline: float32 CPU
    profiler.profile_stage(
        "gen_float32_cpu", _run_baseline_generation, queries, max_tokens,
    )

    # Tier 2 — Traditional Python opt: float16 CPU
    profiler.profile_stage(
        "gen_float16_cpu",
        _run_optimized_generation, queries, "float16", "cpu", max_tokens,
    )

    # Tier 3 — GPU variants
    if has_gpu:
        profiler.profile_stage(
            "gen_float16_gpu",
            _run_optimized_generation, queries, "float16", "cuda", max_tokens,
        )

        if has_4bit:
            profiler.profile_stage(
                "gen_4bit_gpu",
                _run_optimized_generation, queries, "4bit", "cuda", max_tokens,
            )

    profiler.save_results("stage4_results.json")
    profiler.print_summary()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_stage4_benchmark(n)
