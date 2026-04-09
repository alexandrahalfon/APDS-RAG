"""Generate cProfile reports for the baseline pipeline.

Produces both a text summary (sorted by cumulative time) and a .prof binary
that can be visualized with snakeviz or gprof2dot. This shows WHERE time is
spent before any optimization — the profiling-driven approach taught in
Lecture 04.

Usage:
    python benchmarks/profile_baseline_cprofile.py [pdf_folder]

Outputs:
    benchmarks/results/baseline_cprofile.txt   — human-readable top-50 functions
    benchmarks/results/baseline_cprofile.prof  — binary for snakeviz / gprof2dot
"""

import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E702
import torch  # noqa: E402, F401 — load before pdfplumber (macOS segfault)

import cProfile
import pstats
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.run_baseline import run_full_baseline


def profile_baseline(pdf_folder: str, output_dir: str = './benchmarks/results') -> None:
    """Run the full baseline pipeline under cProfile.

    Args:
        pdf_folder: Path to directory containing PDF files.
        output_dir: Where to write the profile output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prof_path = out / 'baseline_cprofile.prof'
    txt_path = out / 'baseline_cprofile.txt'

    print("=== Running baseline pipeline under cProfile ===\n")

    profiler = cProfile.Profile()
    profiler.enable()
    run_full_baseline(pdf_folder)
    profiler.disable()

    # Save binary .prof (for snakeviz: `snakeviz benchmarks/results/baseline_cprofile.prof`)
    profiler.dump_stats(str(prof_path))
    print(f"\n✓ Binary profile saved to {prof_path}")
    print(f"  Visualize with: snakeviz {prof_path}")

    # Save human-readable text report
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(50)  # Top 50 functions by cumulative time

    txt_path.write_text(stream.getvalue(), encoding='utf-8')
    print(f"✓ Text profile saved to {txt_path}")

    # Also print a compact summary to stdout
    print("\n=== Top 20 functions by cumulative time ===\n")
    stats_stdout = pstats.Stats(profiler, stream=sys.stdout)
    stats_stdout.strip_dirs()
    stats_stdout.sort_stats('cumulative')
    stats_stdout.print_stats(20)


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else './data/sample_pdfs'
    profile_baseline(folder)
