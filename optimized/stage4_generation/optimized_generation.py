"""Optimized answer generation with dtype and quantization options.

Optimization tiers:
  1. float16  — halves memory footprint, faster matmuls on GPU
  2. bfloat16 — same memory savings, better numerical stability
  3. 4-bit    — ~4x memory reduction via bitsandbytes quantization
  4. compiled — torch.compile kernel fusion (PyTorch 2.0+)
"""

from typing import List, Dict, Tuple

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baseline.generation_step_local import build_rag_prompt, DEFAULT_MODEL

_models: dict = {}  # Cache: key -> (model, tokenizer)


def _load_model(
    model_name: str,
    optimization: str,
    device: str,
) -> Tuple:
    """Load model with specified optimization, with caching.

    Args:
        model_name: HuggingFace model identifier.
        optimization: One of 'float16', 'bfloat16', '4bit', 'compiled'.
        device: 'cpu' or 'cuda'.

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch

    cache_key = f"{model_name}_{optimization}_{device}"
    if cache_key in _models:
        return _models[cache_key]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading generation model '{model_name}' ({optimization}, {device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if optimization == "float16":  # noqa: SIM114
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
        )
    elif optimization == "bfloat16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        )
    elif optimization == "4bit":
        try:
            from transformers import BitsAndBytesConfig

            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=config,
            )
        except ImportError:
            print("  ⚠ bitsandbytes not available — falling back to float16")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
            )
    elif optimization == "compiled":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
        )
        try:
            model = torch.compile(model)
            print("  ✓ torch.compile applied")
        except Exception as e:
            print(f"  ⚠ torch.compile failed ({e}), using uncompiled float16")
    else:
        raise ValueError(f"Unknown optimization: {optimization}")

    model.eval()

    use_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    if use_device == "cuda" and optimization != "4bit":
        model = model.to("cuda")

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Generation model loaded ({optimization}, {param_count:.0f}M params, {use_device})")

    _models[cache_key] = (model, tokenizer, use_device)
    return model, tokenizer, use_device


def generate_answer_optimized(
    query: str,
    context_chunks: List[Dict],
    max_new_tokens: int = 128,
    model_name: str = DEFAULT_MODEL,
    optimization: str = "float16",
    device: str = "cpu",
) -> str:
    """Generate answer with specified optimization.

    Args:
        query: User question.
        context_chunks: Retrieved context chunks with 'text' key.
        max_new_tokens: Maximum tokens to generate.
        model_name: HuggingFace model identifier.
        optimization: 'float16', 'bfloat16', '4bit', or 'compiled'.
        device: 'cpu' or 'cuda'.

    Returns:
        Generated answer string.
    """
    import torch

    model, tokenizer, use_device = _load_model(model_name, optimization, device)

    prompt = build_rag_prompt(query, context_chunks)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context. Be concise."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(use_device)

    with torch.inference_mode():  # Slightly faster than torch.no_grad()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer.strip()
