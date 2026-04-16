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
        if name.startswith('speedup'):
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
        times = [m['time_seconds'] for k, m in data.items() if not k.startswith('speedup')]
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

    Handles two structures:
      1. Flat (stage*_results.json, e2e_results.json) — {method: {time_seconds, peak_memory_mb}}
      2. Nested (beir_results*.json) — {dataset: {tiers: {tier: {...}}}}

    Args:
        all_results: Dict mapping logical group name to results JSON path.

    Returns:
        DataFrame with columns: Stage, Method, Time (s), Memory (MB), plus
        optional quality columns (NDCG@10, Recall@10) when BEIR data is present.
    """
    rows = []
    for stage_name, results_file in all_results.items():
        data = _load_results(results_file)

        # BEIR nested structure — dataset → tiers → metrics
        if any(isinstance(v, dict) and "tiers" in v for v in data.values()):
            for dataset, res in data.items():
                if not isinstance(res, dict) or "tiers" not in res:
                    continue
                for tier, m in res["tiers"].items():
                    rows.append({
                        "Stage": f"{stage_name}:{dataset}",
                        "Method": tier,
                        "Time (s)": m.get("total_time_s", 0),
                        "Memory (MB)": m.get("peak_memory_mb", 0),
                        "NDCG@10": m.get("ndcg@10", None),
                        "Recall@10": m.get("recall@10", None),
                        "Speedup": m.get("speedup_vs_baseline", None),
                    })
            continue

        # Flat structure — method → {time_seconds, peak_memory_mb}
        for method, metrics in data.items():
            if method.startswith('speedup') or not isinstance(metrics, dict):
                continue
            rows.append({
                "Stage": stage_name,
                "Method": method,
                "Time (s)": metrics.get("time_seconds", 0),
                "Memory (MB)": metrics.get("peak_memory_mb", 0),
                "NDCG@10": None,
                "Recall@10": None,
                "Speedup": None,
            })

    return pd.DataFrame(rows)


def create_beir_table(
    results_file: str, output_dir: str = None,
) -> pd.DataFrame:
    """Build a BEIR-specific table keyed by (dataset, tier) for the report.

    Writes:
      - beir_table.csv   — raw CSV (for spreadsheet import)
      - beir_table.md    — GitHub-flavored markdown (paste into report)
      - beir_table.tex   — LaTeX tabular (if you use LaTeX)

    Args:
        results_file: Path to a beir_results*.json file.
        output_dir: Where to write the table files (defaults to same dir as
            the results file).

    Returns:
        The DataFrame.
    """
    data = _load_results(results_file)
    rows = []
    for dataset, res in data.items():
        if not isinstance(res, dict) or "tiers" not in res:
            continue
        n_docs = res.get("n_corpus", 0)
        for tier, m in res["tiers"].items():
            rows.append({
                "Dataset": dataset,
                "Corpus size": f"{n_docs:,}",
                "Tier": tier,
                "Index (s)": round(m.get("index_time_s", 0), 2),
                "Search (s)": round(m.get("search_time_s", 0), 2),
                "Total (s)": round(m.get("total_time_s", 0), 2),
                "Speedup": (f"{m['speedup_vs_baseline']:.2f}x"
                            if "speedup_vs_baseline" in m else "—"),
                "NDCG@10": f"{m.get('ndcg@10', 0):.3f}",
                "Recall@10": f"{m.get('recall@10', 0):.3f}",
                "MAP@10": f"{m.get('map@10', 0):.3f}",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"⚠ No BEIR data in {results_file}")
        return df

    out_dir = Path(output_dir) if output_dir else Path(results_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(results_file).stem  # e.g. "beir_results" or "beir_results_gpu"
    df.to_csv(out_dir / f"{stem}_table.csv", index=False)
    with open(out_dir / f"{stem}_table.md", "w") as f:
        f.write(df.to_markdown(index=False))
    with open(out_dir / f"{stem}_table.tex", "w") as f:
        f.write(df.to_latex(index=False, escape=False))

    print(f"✓ Saved {stem}_table.{{csv,md,tex}} in {out_dir}")
    print(f"\n{df.to_string(index=False)}\n")
    return df


def plot_beir_cpu_vs_gpu(
    cpu_file: str, gpu_file: str, output_path: str,
) -> None:
    """Side-by-side CPU vs GPU comparison for the same BEIR datasets + tiers.

    Produces a 2-panel chart:
      Left  — total time per (dataset, tier) — CPU bars next to GPU bars
      Right — speedup of GPU over CPU at each (dataset, tier)

    Args:
        cpu_file: Path to CPU BEIR results JSON.
        gpu_file: Path to GPU BEIR results JSON.
        output_path: Where to save the figure.
    """
    cpu = _load_results(cpu_file)
    gpu = _load_results(gpu_file)

    datasets = [d for d in cpu if d in gpu and "tiers" in cpu[d] and "tiers" in gpu[d]]
    if not datasets:
        print(f"⚠ No overlapping datasets between {cpu_file} and {gpu_file}")
        return

    tier_order = list(cpu[datasets[0]]["tiers"].keys())
    n_tiers = len(tier_order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Left: paired CPU/GPU bars for each (dataset, tier) ---
    x = np.arange(len(datasets))
    # 2*n_tiers bars per dataset group (CPU, GPU alternating), packed in 0.8 width
    bar_w = 0.8 / (2 * n_tiers)
    colors_cpu = {"baseline": "#c0392b", "trad_python_opt": "#e67e22", "optimized": "#2874a6"}
    colors_gpu = {"baseline": "#e74c3c", "trad_python_opt": "#f39c12", "optimized": "#3498db"}

    for i, tier in enumerate(tier_order):
        cpu_times = [cpu[d]["tiers"][tier]["total_time_s"] for d in datasets]
        gpu_times = [gpu[d]["tiers"][tier]["total_time_s"] for d in datasets]
        off_cpu = (2 * i - n_tiers) * bar_w + bar_w / 2
        off_gpu = off_cpu + bar_w
        ax1.bar(x + off_cpu, cpu_times, bar_w,
                label=f"{tier} (CPU)", color=colors_cpu.get(tier, "#888"))
        ax1.bar(x + off_gpu, gpu_times, bar_w,
                label=f"{tier} (GPU)", color=colors_gpu.get(tier, "#444"))

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{d}\n({cpu[d]['n_corpus']:,})" for d in datasets],
        rotation=15, ha="right",
    )
    ax1.set_ylabel("Total time (seconds, log scale)")
    ax1.set_title("CPU vs GPU — Total Pipeline Time")
    ax1.legend(ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Right: GPU speedup (CPU / GPU) per tier per dataset ---
    for tier in tier_order:
        speedups = []
        for d in datasets:
            c = cpu[d]["tiers"][tier]["total_time_s"]
            g = gpu[d]["tiers"][tier]["total_time_s"]
            speedups.append(c / g if g > 0 else 0)
        ax2.plot(datasets, speedups, "o-",
                 label=tier, color=colors_gpu.get(tier, "#444"), linewidth=2)

    ax2.set_xticklabels(datasets, rotation=15, ha="right")
    ax2.set_ylabel("GPU speedup over CPU (x)")
    ax2.set_title("GPU Speedup over CPU by Tier")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {output_path}")


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

    # --- BEIR charts + tables ---
    # Registers any beir_results*.json (beir_results.json, beir_results_gpu.json, etc.)
    beir_files = sorted(results_path.glob('beir_results*.json'))
    for beir_file in beir_files:
        stem = beir_file.stem
        all_results[stem] = str(beir_file)
        plot_beir_results(str(beir_file), str(output_path / f'{stem}.png'))
        create_beir_table(str(beir_file), str(output_path))

    # If both CPU and GPU BEIR runs exist, produce the side-by-side chart
    cpu_beir = results_path / 'beir_results.json'
    gpu_beir = results_path / 'beir_results_gpu.json'
    if cpu_beir.exists() and gpu_beir.exists():
        plot_beir_cpu_vs_gpu(
            str(cpu_beir), str(gpu_beir),
            str(output_path / 'beir_cpu_vs_gpu.png'),
        )

    # Re-generate the combined CSV now that BEIR is included
    if all_results:
        df = create_results_table(all_results)
        df.to_csv(str(output_path / 'results_table.csv'), index=False)
        with open(output_path / 'results_table.md', 'w') as f:
            f.write(df.to_markdown(index=False))
        print(f"✓ Saved combined results_table.{{csv,md}}")


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


def plot_beir_results(results_file: str, output_path: str) -> None:
    """Two-panel chart: speed (left) and retrieval quality (right) per dataset.

    Speed panel groups bars by tier within each dataset (log y-axis so a
    100× speedup is still visible alongside a 2× speedup). Quality panel
    overlays NDCG@10 across tiers — flat lines confirm the optimization
    preserved retrieval quality.

    Args:
        results_file: Path to beir_results.json.
        output_path: Where to save the figure.
    """
    data = _load_results(results_file)
    datasets = [name for name, res in data.items() if "tiers" in res]
    if not datasets:
        print(f"⚠ No usable datasets in {results_file}")
        return

    # Tier order is preserved from the first dataset (baseline → trad → optimized)
    tier_order = list(data[datasets[0]]["tiers"].keys())
    colors = {
        "baseline": "#d62728",
        "trad_python_opt": "#ff7f0e",
        "optimized": "#1f77b4",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    x = np.arange(len(datasets))
    bar_width = 0.8 / len(tier_order)

    # --- Left panel: total time per tier per dataset (log scale) ---
    for i, tier in enumerate(tier_order):
        times = [data[d]["tiers"][tier]["total_time_s"] for d in datasets]
        offset = (i - (len(tier_order) - 1) / 2) * bar_width
        ax1.bar(x + offset, times, bar_width, label=tier, color=colors.get(tier))

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{d}\n({data[d]['n_corpus']:,} docs)" for d in datasets],
        rotation=15, ha="right",
    )
    ax1.set_ylabel("Total time (seconds, log scale)")
    ax1.set_title("Pipeline Speed by Dataset")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Right panel: NDCG@10 per tier (should be ~flat across tiers) ---
    for tier in tier_order:
        ndcgs = [data[d]["tiers"][tier]["ndcg@10"] for d in datasets]
        ax2.plot(datasets, ndcgs, "o-", label=tier, color=colors.get(tier), linewidth=2)

    ax2.set_xticklabels(datasets, rotation=15, ha="right")
    ax2.set_ylabel("NDCG@10")
    ax2.set_title("Retrieval Quality by Dataset (flat = optimization preserved quality)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {output_path}")


if __name__ == '__main__':
    generate_all_visualizations()
