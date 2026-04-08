"""NumPy vectorized search — traditional Python optimization before Numba/FAISS.

Demonstrates core "basic Python optimization" techniques:
  1. Replace explicit for-loops with NumPy vectorized operations (C-level speed)
  2. Use float32 instead of float64 (halves memory, faster on modern CPUs)
  3. Use np.argpartition for O(n) partial sort instead of O(n log n) full sort
  4. Pre-compute norms to avoid redundant work across queries
  5. Use matrix multiplication (@) instead of per-row dot products
"""

from typing import Tuple

import numpy as np


def _ensure_float32(arr: np.ndarray) -> np.ndarray:
    """Cast to float32 if needed — avoids copy when already float32.

    Args:
        arr: Input array of any float dtype.

    Returns:
        Array in float32 format.
    """
    if arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def search_similar_vectorized(
    query_emb: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized cosine similarity search using NumPy broadcasting.

    Replaces the baseline Python for-loop with a single matrix multiply
    and vectorized norm computation. This is the "traditional Python
    optimization" step — faster than the baseline but slower than
    Numba JIT or FAISS.

    Optimizations applied:
      - float32 dtype (vs float64 in baseline) → 2× less memory
      - embeddings @ query via BLAS matrix-vector multiply → no Python loop
      - np.argpartition for O(n) top-k selection → avoids full sort

    Args:
        query_emb: 1-D query embedding.
        embeddings: 2-D array of shape (n, dim).
        top_k: Number of results to return.

    Returns:
        Tuple of (indices, scores) arrays, each of length top_k,
        sorted by descending similarity.
    """
    # --- Optimization 1: float32 for memory efficiency ---
    query_emb = _ensure_float32(query_emb)
    embeddings = _ensure_float32(embeddings)

    # --- Optimization 2: vectorized dot product (no Python loop) ---
    # A single BLAS call replaces: for i in range(n): dot(query, emb[i])
    dot_products = embeddings @ query_emb

    # --- Optimization 3: vectorized norms ---
    query_norm = np.linalg.norm(query_emb)
    embedding_norms = np.linalg.norm(embeddings, axis=1)

    # Cosine similarity = dot / (||a|| * ||b||)
    similarities = dot_products / (embedding_norms * query_norm + 1e-8)

    # --- Optimization 4: O(n) partial sort instead of O(n log n) full sort ---
    # np.argpartition finds the top_k elements without sorting everything
    top_k = min(top_k, len(similarities))
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    # Sort only the top_k elements (tiny sort, negligible cost)
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    return top_indices, similarities[top_indices]


def search_similar_vectorized_prenorm(
    query_emb: np.ndarray,
    embeddings_normed: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized search with pre-normalized embeddings.

    When running many queries against the same corpus, normalizing the
    embeddings once and reusing avoids redundant norm computation.

    Optimizations applied (on top of search_similar_vectorized):
      - Pre-normalized embeddings → skip per-query norm calculation
      - Single dot product = cosine similarity for unit vectors

    Args:
        query_emb: 1-D query embedding (will be normalized internally).
        embeddings_normed: 2-D array of L2-normalized embeddings.
        top_k: Number of results to return.

    Returns:
        Tuple of (indices, scores) arrays, each of length top_k.
    """
    query_emb = _ensure_float32(query_emb)

    # Normalize query (one vector — negligible cost)
    query_normed = query_emb / (np.linalg.norm(query_emb) + 1e-8)

    # Dot product on unit vectors = cosine similarity
    similarities = embeddings_normed @ query_normed

    top_k = min(top_k, len(similarities))
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    return top_indices, similarities[top_indices]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows in-place (float32).

    This is the pre-computation step for search_similar_vectorized_prenorm.

    Args:
        embeddings: 2-D array of shape (n, dim).

    Returns:
        L2-normalized copy in float32.
    """
    embeddings = _ensure_float32(embeddings).copy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings /= norms
    return embeddings
