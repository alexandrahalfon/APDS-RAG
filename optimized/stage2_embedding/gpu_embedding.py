"""Optimized embedding generation with batching and optional GPU/AMP support."""

from typing import List, Dict

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baseline.embedding_step_local import get_model


def generate_embeddings_batched(
    chunks: List[Dict],
    batch_size: int = 32,
    device: str = 'cpu',
) -> List[Dict]:
    """Generate embeddings with batching on the specified device.

    Args:
        chunks: List of chunk dicts, each must have a 'text' key.
        batch_size: Number of chunks to encode at once.
        device: 'cpu' or 'cuda'.

    Returns:
        The same list with an 'embedding' key added to each chunk.
    """
    model = get_model()
    model.to(device)

    texts = [c['text'] for c in chunks]
    total = len(texts)
    print(f"Generating embeddings for {total} chunks (batched, device={device}, bs={batch_size})...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]

        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            device=device,
        )

        # Store as float32 list — avoids float64 default and keeps JSON-serializable
        for j, emb in enumerate(embeddings):
            chunks[start + j]['embedding'] = emb.astype(np.float32).tolist()

        if end % (batch_size * 5) == 0 or end == total:
            print(f"  ✓ {end}/{total} chunks embedded")

    print(f"✓ Completed all {total} embeddings (batched)")
    return chunks


def generate_embeddings_gpu_amp(
    chunks: List[Dict],
    batch_size: int = 64,
) -> List[Dict]:
    """Generate embeddings on GPU with automatic mixed precision.

    Requires CUDA. Falls back to batched CPU if unavailable.

    Args:
        chunks: List of chunk dicts, each must have a 'text' key.
        batch_size: Number of chunks per batch.

    Returns:
        The same list with an 'embedding' key added to each chunk.
    """
    import torch

    if not torch.cuda.is_available():
        print("⚠ CUDA not available — falling back to CPU batched mode")
        return generate_embeddings_batched(chunks, batch_size=batch_size, device='cpu')

    from torch.utils.data import DataLoader, Dataset

    model = get_model()
    model.to('cuda')

    texts = [c['text'] for c in chunks]
    total = len(texts)
    print(f"Generating embeddings for {total} chunks (GPU + AMP, bs={batch_size})...")

    class TextDataset(Dataset):
        def __init__(self, texts: List[str]) -> None:
            self.texts = texts

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> str:
            return self.texts[idx]

    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    idx = 0
    for batch_texts in loader:
        batch_texts = list(batch_texts)
        with torch.cuda.amp.autocast():
            embeddings = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                device='cuda',
            )

        for emb in embeddings:
            chunks[idx]['embedding'] = emb.astype(np.float32).tolist()
            idx += 1

        if idx % (batch_size * 5) == 0 or idx == total:
            print(f"  ✓ {idx}/{total} chunks embedded")

    print(f"✓ Completed all {total} embeddings (GPU + AMP)")
    return chunks
