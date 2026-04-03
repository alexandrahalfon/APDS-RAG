"""Profiling framework for the RAG pipeline."""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict


class PipelineProfiler:
    """Profile pipeline stages for time and memory usage.

    Args:
        output_dir: Directory where results JSON files are saved.
    """

    def __init__(self, output_dir: str = './benchmarks/results') -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict[str, float]] = {}

    def profile_stage(self, stage_name: str, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run *func* and record its wall-clock time and peak memory.

        Args:
            stage_name: Label for the stage (used as dict key).
            func: Callable to profile.
            *args, **kwargs: Forwarded to *func*.

        Returns:
            Whatever *func* returns.
        """
        from memory_profiler import memory_usage

        # Measure memory and time together
        start_time = time.perf_counter()
        mem_usage, result = memory_usage(
            (func, args, kwargs),
            interval=0.1,
            retval=True,
            max_usage=True,
        )
        elapsed = time.perf_counter() - start_time

        peak_memory = mem_usage if isinstance(mem_usage, (int, float)) else max(mem_usage)

        self.results[stage_name] = {
            'time_seconds': round(elapsed, 4),
            'peak_memory_mb': round(float(peak_memory), 2),
        }

        print(f"  ✓ {stage_name}: {elapsed:.2f}s | {peak_memory:.1f} MB peak")
        return result

    def save_results(self, filename: str) -> None:
        """Save profiling results to a JSON file.

        Args:
            filename: Name of the JSON file (saved inside output_dir).
        """
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to {path}")

    def print_summary(self) -> None:
        """Print a formatted summary table of all profiled stages."""
        if not self.results:
            print("No profiling results recorded yet.")
            return

        print("\n" + "=" * 60)
        print(f"{'Stage':<30} {'Time (s)':>10} {'Memory (MB)':>12}")
        print("-" * 60)

        total_time = 0.0
        for stage, metrics in self.results.items():
            t = metrics['time_seconds']
            m = metrics['peak_memory_mb']
            total_time += t
            print(f"{stage:<30} {t:>10.2f} {m:>12.1f}")

        print("-" * 60)
        print(f"{'TOTAL':<30} {total_time:>10.2f}")
        print("=" * 60 + "\n")
