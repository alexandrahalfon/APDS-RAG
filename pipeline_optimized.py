"""Optimized RAG pipeline: parallel ingestion, batched embeddings, FAISS search, local generation."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402 — must load before pdfplumber to avoid macOS segfault

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from baseline.embedding_step_local import get_model
from optimized.stage1_ingestion.parallel_ingestion import parallel_ingest
from optimized.stage2_embedding.gpu_embedding import generate_embeddings_batched
from optimized.stage3_search.faiss_search import FAISSIndex
from optimized.stage4_generation.optimized_generation import generate_answer_optimized


class OptimizedRAGPipeline:
    """High-performance RAG pipeline with parallel ingestion, batched
    embeddings, and FAISS vector search.

    Args:
        device: 'cpu' or 'cuda' for embedding generation.
        batch_size: Embedding batch size.
        num_workers: Number of multiprocessing workers for ingestion.
    """

    def __init__(
        self,
        device: str = 'cpu',
        batch_size: int = 64,
        num_workers: Optional[int] = None,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[FAISSIndex] = None

    def process(self, pdf_folder: str) -> None:
        """Ingest PDFs and generate embeddings.

        Args:
            pdf_folder: Path to directory containing PDF files.
        """
        # Pre-load torch/sentence-transformers before pdfplumber runs.
        # On macOS x86_64, loading torch after pdfplumber's native libs segfaults.
        get_model()

        pdf_paths = sorted(str(p) for p in Path(pdf_folder).glob('*.pdf'))
        if not pdf_paths:
            print(f"⚠ No PDFs found in {pdf_folder}")
            return

        # Stage 1: Parallel ingestion
        print("\n--- Stage 1: Parallel Ingestion ---")
        self.chunks = parallel_ingest(pdf_paths, self.num_workers)

        # Stage 2: Batched embeddings
        print("\n--- Stage 2: Batched Embeddings ---")
        self.chunks = generate_embeddings_batched(
            self.chunks, batch_size=self.batch_size, device=self.device
        )

        # Stage 3: Build FAISS index
        print("\n--- Stage 3: Building FAISS Index ---")
        self.embeddings = np.array(
            [c['embedding'] for c in self.chunks], dtype=np.float32
        )
        self.index = FAISSIndex(self.embeddings)

        print(f"\n✓ Pipeline ready: {len(self.chunks)} chunks indexed")

    def query(self, question: str, top_k: int = 10, generate: bool = True) -> Dict:
        """Search for relevant chunks and optionally generate an answer.

        Args:
            question: The query string.
            top_k: Number of retrieval results.
            generate: If True, generate an answer using the local LLM.

        Returns:
            Dict with 'answer' (if generate=True) and 'sources' keys.
        """
        if self.index is None:
            print("⚠ Pipeline not initialized — run process() first")
            return {'answer': '', 'sources': []}

        model = get_model()
        query_emb = model.encode(question).astype(np.float32)
        indices, scores = self.index.search(query_emb, top_k)

        sources = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            sources.append({
                'text': chunk['text'],
                'page': chunk['page'],
                'source_file': chunk.get('source_file', ''),
                'similarity_score': float(score),
            })

        result: Dict = {'sources': sources}

        if generate and sources:
            # Stage 4: Generate answer using optimized local LLM
            answer = generate_answer_optimized(
                question, sources[:5],
                max_new_tokens=256,
                optimization="float16",
                device=self.device,
            )
            result['answer'] = answer
        elif generate:
            result['answer'] = "No relevant context found in the documents."

        return result

    def save(self, output_dir: str = './pipeline_data') -> None:
        """Save pipeline state to disk.

        Args:
            output_dir: Directory to save embeddings and metadata.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / 'embeddings.npy', self.embeddings)

        metadata = [{k: v for k, v in c.items() if k != 'embedding'} for c in self.chunks]
        with open(out / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved pipeline state to {output_dir}/")

    def load(self, data_dir: str = './pipeline_data') -> None:
        """Load pipeline state from disk.

        Args:
            data_dir: Directory containing saved embeddings and metadata.
        """
        data_path = Path(data_dir)

        self.embeddings = np.load(data_path / 'embeddings.npy')
        with open(data_path / 'metadata.json', 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        self.index = FAISSIndex(self.embeddings)
        print(f"✓ Loaded {len(self.chunks)} chunks from {data_dir}/")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Optimized RAG Pipeline")
    sub = parser.add_subparsers(dest='command')

    # process
    p_proc = sub.add_parser('process', help='Process PDF documents')
    p_proc.add_argument('pdf_folder', help='Path to PDF folder')
    p_proc.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p_proc.add_argument('--batch-size', type=int, default=64)
    p_proc.add_argument('--workers', type=int, default=None)
    p_proc.add_argument('--save-dir', default='./pipeline_data')

    # query
    p_query = sub.add_parser('query', help='Query the pipeline')
    p_query.add_argument('question', help='Question to ask')
    p_query.add_argument('--data-dir', default='./pipeline_data')
    p_query.add_argument('--top-k', type=int, default=5)

    # benchmark
    p_bench = sub.add_parser('benchmark', help='Run benchmarks')
    p_bench.add_argument('pdf_folder', help='Path to PDF folder')

    # compare
    p_comp = sub.add_parser('compare', help='Compare baseline vs optimized')
    p_comp.add_argument('pdf_folder', help='Path to PDF folder')

    args = parser.parse_args()

    if args.command == 'process':
        pipeline = OptimizedRAGPipeline(
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        pipeline.process(args.pdf_folder)
        pipeline.save(args.save_dir)

    elif args.command == 'query':
        pipeline = OptimizedRAGPipeline(device=getattr(args, 'device', 'cpu'))
        pipeline.load(args.data_dir)
        result = pipeline.query(args.question, top_k=args.top_k)

        if 'answer' in result:
            print(f"\nAnswer:\n{result['answer']}")

        print(f"\nSources ({len(result['sources'])} chunks):")
        for i, r in enumerate(result['sources'], 1):
            print(f"\n  [{i}] (score: {r['similarity_score']:.4f}) {r['source_file']} p.{r['page']}")
            print(f"      {r['text'][:200]}...")

    elif args.command == 'benchmark':
        from benchmarks.benchmark_e2e import run_e2e_benchmark
        run_e2e_benchmark(args.pdf_folder)

    elif args.command == 'compare':
        from benchmarks.benchmark_e2e import run_e2e_benchmark
        run_e2e_benchmark(args.pdf_folder)
        from benchmarks.visualize import generate_all_visualizations
        generate_all_visualizations()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
