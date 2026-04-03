"""FAISS-based vector search index."""

from typing import Tuple

import numpy as np
import faiss


class FAISSIndex:
    """Wrapper around a FAISS inner-product index for cosine similarity search.

    Embeddings are L2-normalized so inner product == cosine similarity.

    Args:
        embeddings: 2-D float32 array of shape (n, dim).
        use_gpu: Whether to move the index to GPU (requires faiss-gpu).
    """

    def __init__(self, embeddings: np.ndarray, use_gpu: bool = False) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print(f"✓ FAISS index on GPU ({embeddings.shape[0]} vectors, dim={dim})")
            except Exception:
                print("⚠ GPU not available for FAISS — using CPU")

        self.index.add(embeddings)

        if not use_gpu:
            print(f"✓ FAISS index on CPU ({embeddings.shape[0]} vectors, dim={dim})")

    def search(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the top_k most similar vectors.

        Args:
            query: 1-D or 2-D query vector(s). Will be L2-normalized.
            top_k: Number of results.

        Returns:
            Tuple of (indices, scores) arrays.
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)

        return indices[0], scores[0]
