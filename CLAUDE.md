# APDS-RAG: High-Performance RAG Pipeline Optimization

## Project Context

This is a final project for DS-GA 1019 (Advanced Python for Data Science) at NYU. The goal is to systematically optimize an existing RAG (Retrieval-Augmented Generation) pipeline using advanced Python techniques covered in the course.

**Team:** Mac (Alexandra Halfon) and Kund Meghani
**Timeline:** ~2 weeks
**Deliverables:** Optimized code, benchmarks, 4-page report, 5-minute presentation

---

## Current State

This repo was cloned from `pdfRAG` — a working RAG pipeline that uses Mistral APIs. The baseline code is functional but unoptimized and API-dependent.

**Key Problem:** The current implementation uses Mistral APIs for:
- PDF OCR (`mistral-ocr-latest`)
- Embeddings (`mistral-embed`)

This means bottlenecks are network latency, not CPU/GPU — which defeats the purpose of optimization. We need to convert to a fully local pipeline first.

---

## Implementation Phases

Complete these phases IN ORDER. Do not skip ahead.

### PHASE 0: Project Structure Setup

Create the following directory structure:

```
APDS-RAG/
├── baseline/                 # Original code (copy current files here)
├── optimized/
│   ├── stage1_ingestion/
│   ├── stage2_embedding/
│   └── stage3_search/
├── benchmarks/
│   ├── results/
│   └── visualizations/
├── data/
│   └── sample_pdfs/          # Put 20-50 test PDFs here
├── notebooks/
├── report/
│   └── figures/
├── requirements.txt          # Updated with new dependencies
├── CLAUDE.md                 # This file
└── README.md
```

**Tasks:**
1. Create all directories listed above
2. Copy existing `.py` files into `baseline/` directory
3. Keep original files in root as well (for now)

---

### PHASE 1: Update Dependencies

Replace `requirements.txt` with:

```txt
# Original dependencies
fastapi
uvicorn
numpy
nltk
python-multipart
python-dotenv
streamlit

# Local models (replaces Mistral API)
sentence-transformers
torch
torchvision

# Local PDF parsing (replaces Mistral OCR)
pdfplumber
PyMuPDF

# Optimization tools
numba
faiss-cpu

# Profiling
memory-profiler
line-profiler
matplotlib
pandas
seaborn

# Optional: GPU support (uncomment if GPU available)
# cupy-cuda12x
# faiss-gpu
```

**Tasks:**
1. Update `requirements.txt` with the above
2. Do NOT install yet — user will do this manually

---

### PHASE 2: Create Local PDF Parser

Create `baseline/doc_processing_local.py`:

**Requirements:**
- Use `pdfplumber` to extract text from PDFs
- NO API calls
- Function `extract_text_from_pdf(pdf_path: str) -> str` — returns full text with page markers
- Function `process_pdf_complete_local(pdf_path: str) -> dict` — returns structured data matching original format:
  ```python
  {
      'file_name': str,
      'title': str,
      'pages': [
          {
              'page_number': int,
              'text': str,
              'paragraphs': [{'text': str, 'word_count': int}, ...]
          }
      ]
  }
  ```
- Split paragraphs on double newlines
- Keep the same output structure as original `doc_processing.py` so downstream code still works

---

### PHASE 3: Create Local Embedding Generator

Create `baseline/embedding_step_local.py`:

**Requirements:**
- Use `sentence-transformers` with model `all-MiniLM-L6-v2`
- Load model ONCE globally (not per call)
- Function `generate_embeddings_baseline(chunks: List[Dict]) -> List[Dict]`:
  - Process chunks ONE AT A TIME (deliberately slow — this is the baseline)
  - Add 'embedding' key to each chunk dict
  - NO batching, NO GPU — pure sequential CPU
- Function `get_model()` — lazy loads and caches the model

---

### PHASE 4: Create Profiling Framework

Create `benchmarks/profiler.py`:

**Requirements:**
- Class `PipelineProfiler`:
  - `__init__(self, output_dir='./benchmarks/results')` — creates output dir
  - `profile_stage(self, stage_name: str, func, *args, **kwargs)` — times function, measures memory, returns result
  - `save_results(self, filename: str)` — saves to JSON
  - `print_summary(self)` — prints formatted results table
- Use `time.perf_counter()` for timing
- Use `memory_profiler.memory_usage()` for memory measurement
- Store results in dict: `{stage_name: {'time_seconds': float, 'peak_memory_mb': float}}`

---

### PHASE 5: Create Baseline Benchmark Script

Create `benchmarks/run_baseline.py`:

**Requirements:**
- Import from `baseline/` modules
- Function `run_full_baseline(pdf_folder: str, num_queries: int = 10)`:
  1. Profile Stage 1 (ingestion): load all PDFs, chunk them
  2. Profile Stage 2 (embedding): generate embeddings for all chunks
  3. Profile Stage 3 (search): run `num_queries` similarity searches
  4. Save results to `benchmarks/results/baseline_results.json`
  5. Print summary
- Main block that runs with `data/sample_pdfs/` folder

---

### PHASE 6: Implement Stage 1 Optimization (Parallel Ingestion)

Create `optimized/stage1_ingestion/parallel_ingestion.py`:

**Requirements:**
- Function `process_single_pdf(pdf_path: str) -> List[Dict]` — processes one PDF, returns chunks
- Function `parallel_ingest(pdf_paths: List[str], num_workers: int = None) -> List[Dict]`:
  - Use `multiprocessing.Pool`
  - Default `num_workers` to `mp.cpu_count()`
  - Use `pool.map()` to parallelize
  - Flatten results into single list
  - Print progress: `Processing {n} PDFs with {workers} workers`

Create `optimized/stage1_ingestion/numba_text.py`:

**Requirements:**
- Numba JIT functions for text processing (optional, lower priority)
- `@jit(nopython=True)` decorator
- Function `find_sentence_boundaries_numba(char_codes: np.ndarray) -> np.ndarray`
- Function `count_words_numba(text_array: np.ndarray) -> int`

---

### PHASE 7: Implement Stage 2 Optimization (GPU Batching)

Create `optimized/stage2_embedding/gpu_embedding.py`:

**Requirements:**
- Function `generate_embeddings_batched(chunks, batch_size=32, device='cpu') -> List[Dict]`:
  - Use `sentence-transformers` `model.encode()` with batching
  - Process `batch_size` chunks at once
  - Support both 'cpu' and 'cuda' devices
- Function `generate_embeddings_gpu_amp(chunks, batch_size=64) -> List[Dict]`:
  - Requires CUDA
  - Use `torch.cuda.amp.autocast()` for automatic mixed precision
  - Use `torch.utils.data.DataLoader` for batching
  - Move model to GPU with `model.to('cuda')`

---

### PHASE 8: Implement Stage 3 Optimization (Vector Search)

Create `optimized/stage3_search/numba_search.py`:

**Requirements:**
- `@jit(nopython=True)` cosine similarity function
- `@jit(nopython=True, parallel=True)` search function using `prange`
- Function `cosine_similarity_numba(vec1, vec2) -> float`
- Function `search_similar_numba(query_emb, embeddings, top_k=10) -> Tuple[ndarray, ndarray]`
  - Returns (indices, scores)

Create `optimized/stage3_search/faiss_search.py`:

**Requirements:**
- Class `FAISSIndex`:
  - `__init__(self, embeddings: np.ndarray, use_gpu: bool = False)`
  - Normalize embeddings with `faiss.normalize_L2()`
  - Use `IndexFlatIP` for inner product (cosine similarity on normalized vectors)
  - `search(self, query: np.ndarray, top_k: int = 10) -> Tuple[ndarray, ndarray]`

---

### PHASE 9: Create Benchmark Scripts for Each Stage

Create `benchmarks/benchmark_stage1.py`:
- Compare: sequential vs parallel with 2, 4, 8 workers
- Save to `results/stage1_results.json`

Create `benchmarks/benchmark_stage2.py`:
- Compare: CPU sequential vs CPU batched vs GPU batched vs GPU+AMP
- Vary batch sizes: 16, 32, 64, 128
- Save to `results/stage2_results.json`

Create `benchmarks/benchmark_stage3.py`:
- Compare: NumPy baseline vs Numba JIT vs FAISS
- Run 100+ queries for reliable timing
- Save to `results/stage3_results.json`

Create `benchmarks/benchmark_e2e.py`:
- Run full pipeline: baseline vs fully optimized
- Calculate total speedup
- Save to `results/e2e_results.json`

---

### PHASE 10: Create Visualization Scripts

Create `benchmarks/visualize.py`:

**Requirements:**
- Function `plot_stage_comparison(results_file, output_path)` — bar chart comparing implementations
- Function `plot_speedup_summary(all_results, output_path)` — summary chart for report
- Function `create_results_table(all_results) -> pd.DataFrame` — formatted table for report
- Save all figures to `benchmarks/visualizations/`
- Use clean styling (no excessive colors, professional look)

---

### PHASE 11: Update Main Pipeline

Create `pipeline_optimized.py` in root:

**Requirements:**
- Class `OptimizedRAGPipeline`:
  - Uses parallel ingestion
  - Uses GPU batched embeddings (falls back to CPU if no GPU)
  - Uses FAISS for search
  - Same interface as original `RAGPipeline` class
- Function `main()` with argparse:
  - `process` — process documents
  - `query` — answer question
  - `benchmark` — run benchmarks
  - `compare` — compare baseline vs optimized

---

### PHASE 12: Documentation

Update `README.md`:

**Sections:**
1. Project Overview — what this project does
2. Optimization Techniques — table of techniques per stage
3. Results — speedup summary (leave placeholder for actual numbers)
4. Installation — pip install instructions
5. Usage — how to run benchmarks, how to use pipeline
6. Project Structure — directory layout
7. Team — Mac and Kund, DS-GA 1019 Spring 2026

---

## Code Style Guidelines

- Use type hints on all functions
- Include docstrings with Args/Returns
- Use f-strings for formatting
- Print progress messages with ✓ checkmarks for completed steps
- Handle errors gracefully with try/except
- Use `Path` from pathlib, not string concatenation for paths

---

## Testing Approach

After each phase:
1. Run the code to verify it works
2. Check that outputs match expected format
3. Ensure no regressions in existing functionality

Before moving to next phase, confirm:
- [ ] Code runs without errors
- [ ] Output format is correct
- [ ] Integrates with other components

---

## Do NOT

- Do not delete original files until optimized versions are verified
- Do not use Mistral API anywhere in optimized code
- Do not skip profiling — benchmarks are core to the project
- Do not over-engineer — simple, working code beats complex broken code
- Do not implement PySpark — it's a stretch goal, deprioritize

---

## Ask Before Proceeding If

- You're unsure about the expected output format
- A dependency isn't available
- You need to change the interface of existing functions
- Something in these instructions seems wrong or contradictory
