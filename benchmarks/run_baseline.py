"""Run baseline benchmarks for the full RAG pipeline."""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.doc_processing_local import process_pdf_complete_local
from baseline.embedding_step_local import generate_embeddings_baseline, get_model
from baseline.similarity_search import cosine_similarity, search_similar_chunks
from benchmarks.profiler import PipelineProfiler


def _ingest_pdfs(pdf_folder: str) -> list:
    """Load and chunk all PDFs in a folder.

    Args:
        pdf_folder: Path to directory containing PDF files.

    Returns:
        List of chunk dicts ready for embedding.
    """
    from nltk.tokenize import sent_tokenize

    pdf_paths = sorted(Path(pdf_folder).glob('*.pdf'))
    if not pdf_paths:
        print(f"⚠ No PDFs found in {pdf_folder}")
        return []

    print(f"  Ingesting {len(pdf_paths)} PDFs...")
    all_chunks: list = []
    chunk_id = 0

    for pdf_path in pdf_paths:
        try:
            doc = process_pdf_complete_local(str(pdf_path))
        except Exception as e:
            print(f"  ⚠ Skipping {pdf_path.name}: {e}")
            continue

        # Simple chunking: each paragraph becomes a chunk
        for page in doc['pages']:
            for para in page['paragraphs']:
                all_chunks.append({
                    'id': chunk_id,
                    'page': page['page_number'],
                    'text': para['text'],
                    'word_count': para['word_count'],
                    'source_file': doc['file_name'],
                })
                chunk_id += 1

    print(f"  ✓ {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
    return all_chunks


def _run_searches(chunks: list, num_queries: int) -> list:
    """Run similarity searches using random query embeddings.

    Args:
        chunks: List of chunk dicts with 'embedding' key.
        num_queries: How many search queries to run.

    Returns:
        List of search result lists.
    """
    embeddings = np.array([c['embedding'] for c in chunks])
    metadata = [
        {'id': c['id'], 'page': c['page'], 'text': c['text'], 'word_count': c['word_count']}
        for c in chunks
    ]

    # Use actual chunk embeddings as queries (simulate real queries)
    query_indices = np.random.choice(len(embeddings), size=min(num_queries, len(embeddings)), replace=False)
    results = []

    print(f"  Running {len(query_indices)} search queries...")
    for idx in query_indices:
        query_emb = embeddings[idx]
        result = search_similar_chunks(query_emb, embeddings, metadata, top_k=10)
        results.append(result)

    print(f"  ✓ {len(results)} queries completed")
    return results


def run_full_baseline(pdf_folder: str, num_queries: int = 10) -> None:
    """Run and profile the full baseline pipeline.

    Args:
        pdf_folder: Path to directory containing PDF files.
        num_queries: Number of search queries to benchmark.
    """
    profiler = PipelineProfiler()

    print("\n=== Baseline Pipeline Benchmark ===\n")

    # Pre-load the sentence-transformers model BEFORE pdfplumber runs.
    # On macOS x86_64, loading torch's native libs after pdfplumber's
    # cryptography/cffi libs causes a segfault. Loading torch first avoids this.
    get_model()

    # Stage 1: Ingestion
    chunks = profiler.profile_stage("stage1_ingestion", _ingest_pdfs, pdf_folder)
    if not chunks:
        print("No chunks produced — aborting benchmark.")
        return

    # Stage 2: Embedding
    chunks = profiler.profile_stage("stage2_embedding", generate_embeddings_baseline, chunks)

    # Stage 3: Search
    profiler.profile_stage("stage3_search", _run_searches, chunks, num_queries)

    # Save & summarize
    profiler.save_results("baseline_results.json")
    profiler.print_summary()


if __name__ == '__main__':
    pdf_folder = sys.argv[1] if len(sys.argv) > 1 else './data/sample_pdfs'
    run_full_baseline(pdf_folder)
