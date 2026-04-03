# APDS-RAG: High-Performance RAG Pipeline Optimization

A systematic optimization of a Retrieval-Augmented Generation (RAG) pipeline for PDF document processing, built as the final project for **DS-GA 1019 — Advanced Python for Data Science** at NYU.

## Project Overview

Starting from a working but API-dependent RAG pipeline ([pdfRAG](https://github.com/alexandrahalfon/pdfRAG)), we replaced all cloud API calls with local models and applied advanced Python optimization techniques across every stage of the pipeline.

## Optimization Techniques

| Stage | Baseline | Optimized | Techniques |
|-------|----------|-----------|------------|
| **1. PDF Ingestion** | Sequential `pdfplumber` | Parallel ingestion | `multiprocessing.Pool`, Numba JIT text processing |
| **2. Embeddings** | Sequential CPU, one-at-a-time | Batched GPU/CPU | `sentence-transformers` batching, CUDA AMP, `DataLoader` |
| **3. Vector Search** | NumPy loop cosine similarity | FAISS index | `faiss.IndexFlatIP`, Numba `@jit(parallel=True)` with `prange` |

## Results

> Benchmark results will be populated after running on the test dataset. Use `python pipeline_optimized.py compare <pdf_folder>` to generate.

## Installation

```bash
# Clone the repository
git clone https://github.com/alexandrahalfon/APDS-RAG.git
cd APDS-RAG

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### Process documents
```bash
python pipeline_optimized.py process ./data/sample_pdfs/ --device cpu --batch-size 64
```

### Query the pipeline
```bash
python pipeline_optimized.py query "What is the main topic of the document?"
```

### Run benchmarks
```bash
# Individual stages
python benchmarks/benchmark_stage1.py ./data/sample_pdfs/
python benchmarks/benchmark_stage2.py ./data/sample_pdfs/
python benchmarks/benchmark_stage3.py

# End-to-end comparison + visualizations
python pipeline_optimized.py compare ./data/sample_pdfs/
```

### Generate visualizations
```bash
python benchmarks/visualize.py
```

## Project Structure

```
APDS-RAG/
├── baseline/                     # Original + local baseline code
│   ├── doc_processing_local.py   # pdfplumber PDF parser
│   ├── embedding_step_local.py   # Sequential sentence-transformers
│   ├── similarity_search.py      # NumPy cosine similarity
│   └── ...                       # Original pdfRAG files
├── optimized/
│   ├── stage1_ingestion/         # Parallel PDF ingestion + Numba text
│   ├── stage2_embedding/         # Batched GPU/CPU embeddings
│   └── stage3_search/            # FAISS + Numba vector search
├── benchmarks/
│   ├── profiler.py               # PipelineProfiler (time + memory)
│   ├── benchmark_stage*.py       # Per-stage benchmarks
│   ├── benchmark_e2e.py          # End-to-end comparison
│   ├── visualize.py              # Chart generation
│   ├── results/                  # JSON benchmark outputs
│   └── visualizations/           # Generated figures
├── data/sample_pdfs/             # Test PDF documents
├── report/figures/               # Report assets
├── pipeline_optimized.py         # Main optimized pipeline CLI
├── requirements.txt              # All dependencies
└── CLAUDE.md                     # Development instructions
```

## Team

**Mac (Alexandra Halfon)** and **Kund Meghani**
DS-GA 1019 — Advanced Python for Data Science, Spring 2026
New York University
