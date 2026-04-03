"""Local embedding generation using sentence-transformers (no API calls).

Uses all-MiniLM-L6-v2 with deliberate one-at-a-time processing as a baseline.
"""

from typing import List, Dict

_model = None

MODEL_NAME = "all-MiniLM-L6-v2"


def get_model():
    """Lazy-load and cache the sentence-transformers model.

    Returns:
        A SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"✓ Model loaded")
    return _model


def generate_embeddings_baseline(chunks: List[Dict]) -> List[Dict]:
    """Generate embeddings one chunk at a time (baseline — deliberately sequential).

    Args:
        chunks: List of chunk dicts, each must have a 'text' key.

    Returns:
        The same list with an 'embedding' key added to each chunk.
    """
    model = get_model()
    total = len(chunks)
    print(f"Generating embeddings for {total} chunks (sequential baseline)...")

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk['text'], show_progress_bar=False)
        chunk['embedding'] = embedding.tolist()

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  ✓ {i + 1}/{total} chunks embedded")

    print(f"✓ Completed all {total} embeddings")
    return chunks
