"""Visualization utilities for benchmark results."""

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _load_results(results_file: str) -> dict:
    """Load JSON results file.

    Args:
        results_file: Path to the JSON file.

    Returns:
        Parsed results dict.
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_stage_comparison(results_file: str, output_path: str) -> None:
    """Create a bar chart comparing implementations for a single stage.

    Args:
        results_file: Path to the stage results JSON.
        output_path: Where to save the figure.
    """
    data = _load_results(results_file)

    names = []
    times = []
    for name, metrics in data.items():
        if name == 'speedup':
            continue
        names.append(name)
        times.append(metrics['time_seconds'])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Blues_d", len(names))
    bars = ax.barh(names, times, color=colors)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(Path(results_file).stem.replace('_', ' ').title())
    ax.bar_label(bars, fmt='%.2f s', padding=5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {output_path}")


def plot_speedup_summary(all_results: Dict[str, str], output_path: str) -> None:
    """Create a summary bar chart of speedups across all stages.

    Args:
        all_results: Dict mapping stage name to results JSON path.
        output_path: Where to save the figure.
    """
    stages = []
    speedups = []

    for stage_name, results_file in all_results.items():
        data = _load_results(results_file)
        times = [m['time_seconds'] for k, m in data.items() if k != 'speedup']
        if len(times) >= 2:
            baseline = times[0]
            best = min(times[1:])
            speedup = baseline / best if best > 0 else 1.0
            stages.append(stage_name)
            speedups.append(speedup)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Greens_d", len(stages))
    bars = ax.bar(stages, speedups, color=colors)
    ax.set_ylabel("Speedup (x)")
    ax.set_title("Optimization Speedup by Stage")
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.bar_label(bars, fmt='%.1fx', padding=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {output_path}")


def create_results_table(all_results: Dict[str, str]) -> pd.DataFrame:
    """Create a formatted DataFrame summarizing all benchmark results.

    Args:
        all_results: Dict mapping stage name to results JSON path.

    Returns:
        DataFrame with columns: Stage, Method, Time (s), Memory (MB).
    """
    rows = []
    for stage_name, results_file in all_results.items():
        data = _load_results(results_file)
        for method, metrics in data.items():
            if method == 'speedup':
                continue
            rows.append({
                'Stage': stage_name,
                'Method': method,
                'Time (s)': metrics['time_seconds'],
                'Memory (MB)': metrics['peak_memory_mb'],
            })

    df = pd.DataFrame(rows)
    return df


def generate_all_visualizations(results_dir: str = './benchmarks/results',
                                 output_dir: str = './benchmarks/visualizations') -> None:
    """Generate all visualizations from available results.

    Args:
        results_dir: Directory containing result JSON files.
        output_dir: Directory to save figures.
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Per-stage charts
    for stage_file in sorted(results_path.glob('stage*_results.json')):
        stage_name = stage_file.stem.replace('_results', '')
        all_results[stage_name] = str(stage_file)
        plot_stage_comparison(str(stage_file), str(output_path / f'{stage_name}_comparison.png'))

    # E2E chart
    e2e_file = results_path / 'e2e_results.json'
    if e2e_file.exists():
        all_results['e2e'] = str(e2e_file)
        plot_stage_comparison(str(e2e_file), str(output_path / 'e2e_comparison.png'))

    # Speedup summary
    if all_results:
        plot_speedup_summary(all_results, str(output_path / 'speedup_summary.png'))

        # Results table
        df = create_results_table(all_results)
        df.to_csv(str(output_path / 'results_table.csv'), index=False)
        print(f"✓ Saved results_table.csv")
        print(f"\n{df.to_string(index=False)}")

    # --- Scaling charts ---
    scaling_search = results_path / 'scaling_search.json'
    if scaling_search.exists():
        plot_search_scaling(str(scaling_search), str(output_path / 'scaling_search.png'))

    scaling_gen = results_path / 'scaling_generation.json'
    if scaling_gen.exists():
        plot_generation_scaling(str(scaling_gen), str(output_path / 'scaling_generation.png'))

    scaling_corpus = results_path / 'scaling_corpus.json'
    if scaling_corpus.exists():
        plot_corpus_scaling(str(scaling_corpus), str(output_path / 'scaling_corpus.png'))


# ---------------------------------------------------------------------------
# Scaling visualizations
# ---------------------------------------------------------------------------

def plot_search_scaling(results_file: str, output_path: str) -> None:
    """Line chart: search time vs number of vectors for each method.

    Args:
        results_file: Path to scaling_search.json.
        output_path: Where to save the figure.
    """
    data = _load_results(results_file)
    x = data["vector_counts"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: absolute time
    method_labels = {
        "numpy_baseline": "Python for-loop",
        "numpy_vectorized": "NumPy vectorized",
        "numba_jit": "Numba JIT",
        "faiss_flat": "FAISS",
    }
    colors = {"numpy_baseline": "#d62728", "numpy_vectorized": "#ff7f0e",
              "numba_jit": "#2ca02c", "faiss_flat": "#1f77b4"}

    for key, label in method_labels.items():
        if key in data:
            ax1.plot(x, data[key], "o-", label=label, color=colors[key], linewidth=2)

    ax1.set_xlabel("Number of vectors")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Search Time vs Corpus Size")
    ax1.legend()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Right panel: speedup vs baseline
    baseline = np.array(data["numpy_baseline"])
    for key, label in method_labels.items():
        if key in data and key != "numpy_baseline":
            speedups = baseline / np.array(data[key])
            ax2.plot(x, speedups, "o-", label=label, color=colors[key], linewidth=2)

    ax2.set_xlabel("Number of vectors")
    ax2.set_ylabel("Speedup vs baseline (x)")
    ax2.set_title("Search Speedup vs Corpus Size")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend()
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {output_path}")


def plot_generation_scaling(results_file: str, output_path: str) -> None:
    """Line chart: generation time vs output token count.

    Args:
        results_file: Path to scaling_generation.json.
        output_path: Where to save the figure.
    """
    data = _load_results(results_file)
    x = data["token_counts"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    method_labels = {
        "gen_float32_cpu": "float32 (CPU)",
        "gen_float16_cpu": "float16 (CPU)",
        "gen_float16_gpu": "float16 (GPU)",
    }
    colors = {"gen_float32_cpu": "#d62728", "gen_float16_cpu": "#ff7f0e",
              "gen_float16_gpu": "#1f77b4"}

    # Left: absolute time
    for key, label in method_labels.items():
        if key in data:
            ax1.plot(x, data[key], "o-", label=label, color=colors[key], linewidth=2)

    ax1.set_xlabel("Max output tokens")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Generation Time vs Output Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: speedup vs float32
    baseline = np.array(data["gen_float32_cpu"])
    for key, label in method_labels.items():
        if key in data and key != "gen_float32_cpu":
            speedups = baseline / np.array(data[key])
            ax2.plot(x, speedups, "o-", label=label, color=colors[key], linewidth=2)

    ax2.set_xlabel("Max output tokens")
    ax2.set_ylabel("Speedup vs float32 (x)")
    ax2.set_title("Generation Speedup vs Output Length")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {output_path}")


def plot_corpus_scaling(results_file: str, output_path: str) -> None:
    """Line chart: e2e pipeline time and speedup vs number of PDFs.

    Args:
        results_file: Path to scaling_corpus.json.
        output_path: Where to save the figure.
    """
    data = _load_results(results_file)
    x = data["corpus_sizes"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    tier_labels = {
        "baseline": "Baseline",
        "trad_python_opt": "Trad. Python Opt",
        "fully_optimized": "Fully Optimized",
    }
    colors = {"baseline": "#d62728", "trad_python_opt": "#ff7f0e",
              "fully_optimized": "#1f77b4"}

    # Left: absolute time
    for key, label in tier_labels.items():
        if key in data:
            ax1.plot(x, data[key], "o-", label=label, color=colors[key], linewidth=2)

    ax1.set_xlabel("Number of PDFs")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("E2E Pipeline Time vs Corpus Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: speedup vs baseline
    baseline = np.array(data["baseline"])
    for key, label in tier_labels.items():
        if key in data and key != "baseline":
            speedups = baseline / np.array(data[key])
            ax2.plot(x, speedups, "o-", label=label, color=colors[key], linewidth=2)

    ax2.set_xlabel("Number of PDFs")
    ax2.set_ylabel("Speedup vs baseline (x)")
    ax2.set_title("E2E Speedup vs Corpus Size")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {output_path}")


if __name__ == '__main__':
    generate_all_visualizations()
