"""Visualization utilities for benchmark results."""

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    generate_all_visualizations()
