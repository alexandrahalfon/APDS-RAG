"""Optimized RAG pipeline using parallel ingestion, batched embeddings, and FAISS search."""

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

    def query(self, question: str, top_k: int = 10) -> List[Dict]:
        """Search for chunks relevant to a question.

        Args:
            question: The query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with text and similarity score.
        """
        if self.index is None:
            print("⚠ Pipeline not initialized — run process() first")
            return []

        model = get_model()
        query_emb = model.encode(question).astype(np.float32)
        indices, scores = self.index.search(query_emb, top_k)

        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                'text': chunk['text'],
                'page': chunk['page'],
                'source_file': chunk.get('source_file', ''),
                'similarity_score': float(score),
            })

        return results

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
        pipeline = OptimizedRAGPipeline()
        pipeline.load(args.data_dir)
        results = pipeline.query(args.question, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r['similarity_score']:.4f}) ---")
            print(f"Source: {r['source_file']} (page {r['page']})")
            print(r['text'][:300])

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
