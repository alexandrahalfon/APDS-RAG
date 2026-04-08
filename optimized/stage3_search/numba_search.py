"""Numba JIT-compiled vector similarity search."""

from typing import Tuple

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def cosine_similarity_numba(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score.
    """
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for i in range(len(vec1)):
        dot += vec1[i] * vec2[i]
        norm1 += vec1[i] * vec1[i]
        norm2 += vec2[i] * vec2[i]

    denom = (norm1 ** 0.5) * (norm2 ** 0.5)
    if denom == 0.0:
        return 0.0
    return dot / denom


@jit(nopython=True, parallel=True)
def search_similar_numba(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the top_k most similar embeddings using parallel Numba.

    Args:
        query_emb: 1-D query embedding.
        embeddings: 2-D array of shape (n, dim).
        top_k: Number of results to return.

    Returns:
        Tuple of (indices, scores) arrays, each of length top_k.
    """
    n = embeddings.shape[0]
    scores = np.empty(n, dtype=np.float32)

    for i in prange(n):
        scores[i] = cosine_similarity_numba(query_emb, embeddings[i])

    # Partial argsort — get top_k indices
    indices = np.argsort(scores)[::-1][:top_k]
    top_scores = np.empty(top_k, dtype=np.float32)
    for i in range(top_k):
        top_scores[i] = scores[indices[i]]

    return indices, top_scores
