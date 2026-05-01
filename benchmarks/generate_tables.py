"""Regenerate benchmarks/tables/ from current results JSON files."""
import json
import os
from pathlib import Path

import pandas as pd

RESULTS = Path("benchmarks/results")
TABLES = Path("benchmarks/tables")
TABLES.mkdir(parents=True, exist_ok=True)

# ── Table 1: Per-Stage Results ──
rows = []
stage_map = {"stage1": "Stage 1", "stage2": "Stage 2", "stage3": "Stage 3", "stage4": "Stage 4"}
for stage_key, stage_label in stage_map.items():
    path = RESULTS / f"{stage_key}_results.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    baseline_time = None
    for method, m in data.items():
        if method.startswith("speedup") or not isinstance(m, dict):
            continue
        t = m["time_seconds"]
        if baseline_time is None:
            baseline_time = t
        speedup = f"{baseline_time / t:.2f}x" if t > 0 else "—"
        rows.append({
            "Stage": stage_label,
            "Method": method,
            "Time (s)": round(t, 2),
            "Memory (MB)": round(m.get("peak_memory_mb", 0), 1),
            "Speedup": speedup,
        })

df1 = pd.DataFrame(rows)
df1.to_csv(TABLES / "table1_per_stage.csv", index=False)
print(f"✓ table1_per_stage.csv ({len(df1)} rows)")

# ── Table 2: E2E Results ──
e2e_path = RESULTS / "e2e_results.json"
if e2e_path.exists():
    data = json.loads(e2e_path.read_text())
    baseline_t = data["baseline_e2e"]["time_seconds"]
    rows = []
    for key, label in [("baseline_e2e", "Baseline"), ("trad_python_opt_e2e", "Trad Python Opt"), ("optimized_e2e", "Optimized")]:
        m = data[key]
        t = m["time_seconds"]
        rows.append({
            "Pipeline Tier": label,
            "Total Time (s)": round(t, 2),
            "Peak Memory (MB)": round(m["peak_memory_mb"], 1),
            "Speedup vs Baseline": f"{baseline_t / t:.2f}x" if t > 0 else "—",
        })
    df2 = pd.DataFrame(rows)
    df2.to_csv(TABLES / "table2_e2e.csv", index=False)
    print(f"✓ table2_e2e.csv")

# ── Table 3: Search Scaling ──
scaling_path = RESULTS / "scaling_search.json"
if scaling_path.exists():
    data = json.loads(scaling_path.read_text())
    rows = []
    for i, n in enumerate(data["vector_counts"]):
        bl = data["numpy_baseline"][i]
        row = {"Vectors": f"{n:,}"}
        for key, label in [("numpy_vectorized", "NumPy vectorized"), ("numba_jit", "Numba JIT"), ("faiss_flat", "FAISS")]:
            t = data[key][i]
            row[f"{label} (s)"] = round(t, 3)
            row[f"{label} speedup"] = f"{bl / t:.1f}x" if t > 0 else "—"
        row["Python for-loop (s)"] = round(bl, 3)
        rows.append(row)
    cols = ["Vectors", "Python for-loop (s)", "NumPy vectorized (s)", "NumPy vectorized speedup",
            "Numba JIT (s)", "Numba JIT speedup", "FAISS (s)", "FAISS speedup"]
    df3 = pd.DataFrame(rows)[cols]
    df3.to_csv(TABLES / "table3_search_scaling.csv", index=False)
    print(f"✓ table3_search_scaling.csv")

# ── Table 4: Generation Scaling ──
gen_path = RESULTS / "scaling_generation.json"
if gen_path.exists():
    data = json.loads(gen_path.read_text())
    rows = []
    for i, tokens in enumerate(data["token_counts"]):
        f32 = data["gen_float32_cpu"][i]
        row = {"Max Tokens": tokens, "f32 CPU (s)": round(f32, 2)}
        for key, label in [("gen_float16_cpu", "f16 CPU"), ("gen_float16_gpu", "f16 GPU")]:
            if key in data:
                t = data[key][i]
                row[f"{label} (s)"] = round(t, 2)
                row[f"{label} speedup"] = f"{f32 / t:.2f}x" if t > 0 else "—"
        rows.append(row)
    df4 = pd.DataFrame(rows)
    df4.to_csv(TABLES / "table4_generation_scaling.csv", index=False)
    print(f"✓ table4_generation_scaling.csv")

# ── Summary Table ──
summary_rows = []
for stage_key in ["stage1", "stage2", "stage3", "stage4"]:
    path = RESULTS / f"{stage_key}_results.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    items = [(k, v["time_seconds"]) for k, v in data.items() if isinstance(v, dict) and "time_seconds" in v and not k.startswith("speedup")]
    if len(items) < 2:
        continue
    bl_name, bl_time = items[0]
    best_name, best_time = min(items[1:], key=lambda x: x[1])
    summary_rows.append({
        "Stage": stage_key,
        "Baseline": bl_name,
        "Baseline Time (s)": round(bl_time, 2),
        "Best Method": best_name,
        "Best Time (s)": round(best_time, 2),
        "Speedup": f"{bl_time / best_time:.1f}x" if best_time > 0 else "—",
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(TABLES / "table_summary.csv", index=False)
print(f"✓ table_summary.csv")

# ── Combined HTML ──
html = """<html><head><style>
body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
table { border-collapse: collapse; margin: 20px 0; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
th { background-color: #f4f4f4; text-align: center; }
td:first-child, th:first-child { text-align: left; }
h2 { margin-top: 40px; }
</style></head><body>
<h1>APDS-RAG Benchmark Results</h1>
"""
html += "<h2>Table 1: Per-Stage Benchmark Results</h2>\n" + df1.to_html(index=False)
if e2e_path.exists():
    html += "<h2>Table 2: End-to-End Pipeline</h2>\n" + df2.to_html(index=False)
if scaling_path.exists():
    html += "<h2>Table 3: Search Scaling</h2>\n" + df3.to_html(index=False)
if gen_path.exists():
    html += "<h2>Table 4: Generation Scaling</h2>\n" + df4.to_html(index=False)
html += "<h2>Summary: Best Speedup per Stage</h2>\n" + df_summary.to_html(index=False)
html += "</body></html>"

(TABLES / "all_tables.html").write_text(html)
print(f"✓ all_tables.html")
print(f"\nAll tables regenerated in {TABLES}/")
