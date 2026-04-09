"""Local answer generation using a small language model (no API calls).

Baseline implementation: float32 precision, sequential token generation.
This is deliberately unoptimized as the comparison baseline.
"""

from typing import List, Dict, Optional

_model = None
_tokenizer = None

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_generation_model(model_name: str = DEFAULT_MODEL):
    """Lazy-load and cache the language model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Tuple of (model, tokenizer).
    """
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading generation model '{model_name}'...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Deliberately unoptimized
        )
        _model.eval()
        param_count = sum(p.numel() for p in _model.parameters()) / 1e6
        print(f"✓ Generation model loaded (float32, {param_count:.0f}M params)")
    return _model, _tokenizer


def build_rag_prompt(query: str, context_chunks: List[Dict]) -> str:
    """Build a RAG prompt from query and retrieved context chunks.

    Args:
        query: User question.
        context_chunks: List of chunk dicts with 'text' and optionally 'page'.

    Returns:
        Formatted prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        page_info = f" (Page {chunk['page']})" if 'page' in chunk else ""
        context_parts.append(f"[{i}]{page_info}: {chunk['text']}")

    context = "\n\n".join(context_parts)
    return (
        f"Based on the following context from the documents, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def generate_answer_baseline(
    query: str,
    context_chunks: List[Dict],
    max_new_tokens: int = 128,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Generate an answer using baseline (unoptimized) generation.

    Deliberately unoptimized:
      - float32 precision (2x memory vs float16)
      - torch.no_grad (not inference_mode)
      - CPU only

    Args:
        query: User question.
        context_chunks: Retrieved context chunks with 'text' key.
        max_new_tokens: Maximum tokens to generate.
        model_name: HuggingFace model identifier.

    Returns:
        Generated answer string.
    """
    import torch

    model, tokenizer = get_generation_model(model_name)

    prompt = build_rag_prompt(query, context_chunks)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context. Be concise."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer.strip()
