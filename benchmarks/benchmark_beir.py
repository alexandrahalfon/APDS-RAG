"""benchmarks/benchmark_beir.py — Run RAG retrieval benchmarks on BEIR datasets.

Compares the three pipeline tiers across one or more BEIR retrieval datasets,
reporting BOTH performance (index time, search time, peak memory) AND
retrieval quality (NDCG@10, Recall@10, MAP@10, Precision@10).

Why BEIR vs. synthetic PDFs:
  - Standard IR benchmark — numbers are comparable to the literature
  - Naturally spans corpus sizes (3.6k → 8.8M docs depending on dataset)
  - Naturally spans doc length distributions (claims, abstracts, papers)
  - Provides ground-truth relevance judgments (qrels), so we can verify
    that a "performance optimization" didn't silently break retrieval
    quality (this is the unsung benefit — performance tuning regularly
    introduces correctness regressions otherwise hidden by faster timings).

Usage:
    # Smoke test on the smallest dataset (~5 MB, ~5k docs, runs in <1 min)
    python benchmarks/benchmark_beir.py --datasets scifact

    # Standard 4-dataset report run (~150 MB, ~30 min on CPU, ~5 min on GPU)
    python benchmarks/benchmark_beir.py \\
        --datasets scifact nfcorpus fiqa scidocs

    # Skip the slow Python for-loop baseline on large corpora
    python benchmarks/benchmark_beir.py --datasets trec-covid --skip-baseline

    # Cap query count to bound runtime
    python benchmarks/benchmark_beir.py --datasets fiqa --max-queries 100
"""

import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E702
import torch  # noqa: E402, F401 — must load before pdfplumber (macOS segfault)

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.profiler import PipelineProfiler

# Pipeline implementations — same imports as benchmark_e2e.py
from baseline.embedding_step_local import generate_embeddings_baseline, get_model
from baseline.similarity_search import search_similar_chunks
from optimized.stage2_embedding.gpu_embedding import generate_embeddings_batched
from optimized.stage3_search.numpy_vectorized import (
    search_similar_vectorized,
    normalize_embeddings,
)
from optimized.stage3_search.faiss_search import FAISSIndex


# ---------------------------------------------------------------------------
# BEIR dataset registry
# ---------------------------------------------------------------------------

# Canonical dataset URL pattern from beir-cellar/beir (Thakur et al., 2021).
_BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

# Curated subset spanning small → large, plus task variety.
# `split` is the relevance-judgement split BEIR ships for that dataset.
DATASETS = {
    # name        : (approx_corpus_size, approx_disk_mb, split, task_type)
    "scifact":     (5_183,    5,   "test", "fact-checking (science)"),
    "nfcorpus":    (3_633,    5,   "test", "QA (medical)"),
    "arguana":     (8_674,    8,   "test", "argument retrieval"),
    "scidocs":     (25_657,   80,  "test", "citation prediction"),
    "fiqa":        (57_638,   30,  "test", "QA (financial)"),
    "trec-covid":  (171_332,  600, "test", "bio-medical search"),
    "nq":          (2_681_468, 2900, "test", "QA (Natural Questions)"),
    "msmarco":     (8_841_823, 3000, "dev",  "passage retrieval"),
}


def _dataset_url(name: str) -> str:
    """Return the canonical zip URL for a BEIR dataset."""
    return f"{_BEIR_BASE}/{name}.zip"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def download_dataset(name: str, base_dir: str = "./data/beir") -> str:
    """Download and unzip a BEIR dataset (cached — re-runs are no-ops).

    Args:
        name: Dataset key from DATASETS.
        base_dir: Where to cache datasets locally.

    Returns:
        Absolute path to the extracted dataset directory.
    """
    from beir import util  # local import keeps benchmarks/ importable on machines without beir

    out_dir = Path(base_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = out_dir / name
    if dataset_dir.exists() and (dataset_dir / "corpus.jsonl").exists():
        print(f"  ✓ {name} cached at {dataset_dir}")
        return str(dataset_dir)

    n_docs, mb, _, task = DATASETS[name]
    print(f"  Downloading {name} (~{mb} MB, ~{n_docs:,} docs, {task})...")
    return util.download_and_unzip(_dataset_url(name), str(out_dir))


def load_beir_dataset(
    name: str, base_dir: str = "./data/beir",
) -> Tuple[List[Dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load a BEIR dataset and convert the corpus into chunk-list format.

    Args:
        name: Dataset key from DATASETS.
        base_dir: Local cache directory.

    Returns:
        chunks  — list of dicts (id, text, page, word_count, source_file)
        queries — dict {query_id: query_text}
        qrels   — dict {query_id: {doc_id: relevance}}
    """
    from beir.datasets.data_loader import GenericDataLoader

    path = download_dataset(name, base_dir)
    split = DATASETS[name][2]
    corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split=split)

    # Convert {doc_id: {title, text}} → list of chunk dicts (matches the
    # shape produced by Stage 1 ingestion so the rest of the pipeline can
    # consume it unchanged).
    chunks: List[Dict] = []
    for doc_id, doc in corpus.items():
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        combined = f"{title}\n\n{text}".strip() if title else text
        chunks.append({
            "id": doc_id,                        # preserve original doc_id for qrels lookup
            "text": combined,
            "page": 0,
            "word_count": len(combined.split()),
            "source_file": name,
        })
    return chunks, queries, qrels


def _sample_queries(
    queries: Dict[str, str], qrels: Dict, max_queries: int, seed: int = 42,
) -> Tuple[Dict[str, str], Dict]:
    """Cap query count by random sampling, keeping qrels in sync."""
    if not max_queries or len(queries) <= max_queries:
        return queries, qrels
    rng = np.random.default_rng(seed)
    qids = list(queries.keys())
    rng.shuffle(qids)
    keep = set(qids[:max_queries])
    return (
        {qid: queries[qid] for qid in keep},
        {qid: qrels[qid] for qid in keep if qid in qrels},
    )


# ---------------------------------------------------------------------------
# Tier implementations
# ---------------------------------------------------------------------------
# Each tier returns:
#   results      — {query_id: {doc_id: score}}  (BEIR's expected format)
#   index_time   — seconds to embed corpus + build any auxiliary structure
#   search_time  — seconds to embed all queries + run all top-k searches
# ---------------------------------------------------------------------------

def run_baseline_tier(chunks, queries, top_k=10) -> Tuple[Dict, float, float]:
    """Tier 1 — sequential embed + Python for-loop cosine similarity."""
    # Index: embed corpus one chunk at a time
    t0 = time.perf_counter()
    indexed = generate_embeddings_baseline([dict(c) for c in chunks])
    corpus_embs = np.array([c["embedding"] for c in indexed], dtype=np.float32)
    metadata = [{"id": c["id"]} for c in indexed]
    index_time = time.perf_counter() - t0

    # Search: embed each query the same way + linear scan
    t0 = time.perf_counter()
    query_chunks = generate_embeddings_baseline(
        [{"text": q, "id": qid} for qid, q in queries.items()],
    )
    results: Dict[str, Dict[str, float]] = {}
    for q in query_chunks:
        q_emb = np.asarray(q["embedding"], dtype=np.float32)
        top = search_similar_chunks(q_emb, corpus_embs, metadata, top_k=top_k)
        results[q["id"]] = {r["id"]: float(r["similarity_score"]) for r in top}
    search_time = time.perf_counter() - t0

    return results, index_time, search_time


def run_trad_python_tier(chunks, queries, top_k=10) -> Tuple[Dict, float, float]:
    """Tier 2 — sequential embed (same as baseline) + vectorized NumPy search."""
    t0 = time.perf_counter()
    indexed = generate_embeddings_baseline([dict(c) for c in chunks])
    corpus_embs = np.array([c["embedding"] for c in indexed], dtype=np.float32)
    embs_normed = normalize_embeddings(corpus_embs)
    index_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    query_chunks = generate_embeddings_baseline(
        [{"text": q, "id": qid} for qid, q in queries.items()],
    )
    results: Dict[str, Dict[str, float]] = {}
    for q in query_chunks:
        q_emb = np.asarray(q["embedding"], dtype=np.float32)
        top_idx, top_scores = search_similar_vectorized(q_emb, embs_normed, top_k=top_k)
        results[q["id"]] = {
            chunks[int(j)]["id"]: float(s) for j, s in zip(top_idx, top_scores)
        }
    search_time = time.perf_counter() - t0

    return results, index_time, search_time


def run_optimized_tier(
    chunks, queries, top_k=10, batch_size: int = 64,
) -> Tuple[Dict, float, float]:
    """Tier 3 — batched (GPU if available) embed + FAISS search."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    indexed = generate_embeddings_batched(
        [dict(c) for c in chunks], batch_size=batch_size, device=device,
    )
    corpus_embs = np.array([c["embedding"] for c in indexed], dtype=np.float32)
    faiss_index = FAISSIndex(corpus_embs)
    index_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    query_chunks = generate_embeddings_batched(
        [{"text": q, "id": qid} for qid, q in queries.items()],
        batch_size=batch_size, device=device,
    )
    results: Dict[str, Dict[str, float]] = {}
    for q in query_chunks:
        q_emb = np.asarray(q["embedding"], dtype=np.float32)
        top_idx, top_scores = faiss_index.search(q_emb, top_k=top_k)
        results[q["id"]] = {
            chunks[int(j)]["id"]: float(s)
            for j, s in zip(top_idx, top_scores) if j >= 0
        }
    search_time = time.perf_counter() - t0

    return results, index_time, search_time


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def _evaluate_quality(
    qrels: Dict, results: Dict, top_k: int = 10,
) -> Dict[str, float]:
    """Compute NDCG/MAP/Recall/Precision at top_k via BEIR's evaluator."""
    from beir.retrieval.evaluation import EvaluateRetrieval

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [top_k],
    )
    return {
        f"ndcg@{top_k}":      round(float(ndcg[f"NDCG@{top_k}"]), 4),
        f"map@{top_k}":       round(float(_map[f"MAP@{top_k}"]), 4),
        f"recall@{top_k}":    round(float(recall[f"Recall@{top_k}"]), 4),
        f"precision@{top_k}": round(float(precision[f"P@{top_k}"]), 4),
    }


def run_dataset_benchmark(
    name: str,
    max_queries: int = 50,
    skip_baseline: bool = False,
    top_k: int = 10,
    base_dir: str = "./data/beir",
) -> Dict:
    """Run all three tiers on one BEIR dataset; return combined perf+quality."""
    print(f"\n{'='*72}")
    print(f"  BEIR Benchmark: {name}")
    print(f"{'='*72}")

    chunks, queries_full, qrels_full = load_beir_dataset(name, base_dir)
    queries, qrels = _sample_queries(queries_full, qrels_full, max_queries)
    print(f"  {len(chunks):,} corpus chunks | {len(queries)} queries (of {len(queries_full):,})")

    profiler = PipelineProfiler()
    tier_results: Dict[str, Dict] = {}

    def _run_tier(label: str, fn) -> None:
        retrieved, t_idx, t_search = profiler.profile_stage(
            f"{name}__{label}", fn, chunks, queries, top_k,
        )
        quality = _evaluate_quality(qrels, retrieved, top_k=top_k)
        # Replace the wall-clock from profile_stage with a precise breakdown
        # (profile_stage timed the whole tier; we want index/search separately).
        prof = profiler.results[f"{name}__{label}"]
        tier_results[label] = {
            "index_time_s":  round(t_idx,    4),
            "search_time_s": round(t_search, 4),
            "total_time_s":  round(t_idx + t_search, 4),
            "peak_memory_mb": prof["peak_memory_mb"],
            **quality,
        }

    if not skip_baseline:
        _run_tier("baseline", run_baseline_tier)
    else:
        print("  (Skipping baseline tier — large corpus)")

    _run_tier("trad_python_opt", run_trad_python_tier)
    _run_tier("optimized",       run_optimized_tier)

    # Speedup vs baseline (only computable if baseline was run)
    if "baseline" in tier_results:
        b_total = tier_results["baseline"]["total_time_s"]
        for label, m in tier_results.items():
            m["speedup_vs_baseline"] = round(b_total / m["total_time_s"], 2) if m["total_time_s"] > 0 else 0.0

    return {
        "dataset":     name,
        "task":        DATASETS[name][3],
        "n_corpus":    len(chunks),
        "n_queries":   len(queries),
        "top_k":       top_k,
        "device":      "cuda" if torch.cuda.is_available() else "cpu",
        "tiers":       tier_results,
    }


# ---------------------------------------------------------------------------
# Top-level entry point + summary printing
# ---------------------------------------------------------------------------

def _print_summary(all_results: Dict[str, Dict]) -> None:
    """Print a compact human-readable summary table."""
    print("\n" + "=" * 96)
    print(f"  SUMMARY  ({len(all_results)} dataset(s))")
    print("=" * 96)
    print(f"{'Dataset':<14} {'N_corpus':>10} {'Tier':>18} "
          f"{'Total (s)':>10} {'Speedup':>9} {'NDCG@10':>9} {'Recall@10':>10}")
    print("-" * 96)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<14} ERROR: {res['error']}")
            continue
        for label, m in res["tiers"].items():
            speedup = m.get("speedup_vs_baseline", "—")
            speedup_s = f"{speedup}x" if isinstance(speedup, (int, float)) else f"{speedup}"
            print(
                f"{name:<14} {res['n_corpus']:>10,} {label:>18} "
                f"{m['total_time_s']:>10.2f} {speedup_s:>9} "
                f"{m['ndcg@10']:>9.3f} {m['recall@10']:>10.3f}"
            )
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BEIR retrieval benchmarks across the three pipeline tiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available datasets:\n"
            + "\n".join(
                f"  {name:<12} {n:>10,} docs  (~{mb} MB)  {task}"
                for name, (n, mb, _, task) in DATASETS.items()
            )
        ),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["scifact"], choices=list(DATASETS),
        help="BEIR datasets to benchmark (default: scifact only — quick smoke test).",
    )
    parser.add_argument(
        "--max-queries", type=int, default=50,
        help="Cap query count per dataset (default: 50). Use 0 to disable cap.",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the slow Python for-loop baseline (recommended for >50k corpora).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--data-dir", default="./data/beir",
        help="Local cache directory for downloaded BEIR datasets.",
    )
    parser.add_argument(
        "--output", default="./benchmarks/results/beir_results.json",
    )
    args = parser.parse_args()

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Pre-loading embedding model (all-MiniLM-L6-v2)...")
    get_model()

    all_results: Dict[str, Dict] = {}
    for dataset in args.datasets:
        try:
            all_results[dataset] = run_dataset_benchmark(
                dataset,
                max_queries=args.max_queries,
                skip_baseline=args.skip_baseline,
                top_k=args.top_k,
                base_dir=args.data_dir,
            )
        except Exception as e:
            print(f"  ✗ {dataset} failed: {type(e).__name__}: {e}")
            all_results[dataset] = {"error": f"{type(e).__name__}: {e}"}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved {out}")

    _print_summary(all_results)


if __name__ == "__main__":
    main()
