"""Download arXiv CS/ML papers for benchmarking the RAG pipeline."""

import time
from pathlib import Path

import requests

# Mix of well-known ML papers with varied lengths (5-50+ pages)
ARXIV_IDS = [
    # Foundational models
    "2310.06825",   # Mistral 7B
    "2307.09288",   # Llama 2
    "2312.10997",   # Mixtral of Experts
    "2303.08774",   # GPT-4 Technical Report
    "2305.10403",   # StarCoder
    "2302.13971",   # LLaMA
    "2204.02311",   # PaLM
    "2005.14165",   # GPT-3
    "1706.03762",   # Attention Is All You Need
    "1810.04805",   # BERT

    # RAG and retrieval
    "2005.11401",   # RAG (Retrieval-Augmented Generation)
    "2112.09332",   # Retro
    "2208.03299",   # Atlas
    "2301.12652",   # REPLUG
    "2310.11511",   # Self-RAG

    # Efficiency and optimization
    "2106.09685",   # LoRA
    "2305.14314",   # QLoRA
    "2210.17323",   # GPTQ
    "2310.11453",   # Mistral sliding window attention
    "2205.14135",   # FlashAttention

    # Embeddings and search
    "2212.03533",   # E5 embeddings
    "1908.10084",   # Sentence-BERT
    "2104.08663",   # Condenser
    "2201.10005",   # Text and Code Embeddings
    "2310.07554",   # Retrieval meets Long Context LLMs

    # Agents and reasoning
    "2210.03629",   # ReAct
    "2305.10601",   # Tree of Thoughts
    "2201.11903",   # Chain of Thought
    "2303.11366",   # Reflexion
    "2308.12950",   # AgentBench

    # Vision and multimodal
    "2304.08485",   # LLaVA
    "2310.03744",   # LLaVA-1.5
    "2301.12597",   # BLIP-2
    "2303.04671",   # Visual Instruction Tuning
    "2312.11805",   # Gemini

    # Training and alignment
    "2203.02155",   # InstructGPT / RLHF
    "2305.18290",   # Direct Preference Optimization (DPO)
    "2204.05862",   # Training Compute-Optimal LLMs (Chinchilla)
    "2210.11416",   # Scaling Data-Constrained LMs
    "2307.15043",   # Universal and Transferable Attacks on LLMs

    # Data and benchmarks
    "2306.11644",   # Textbooks Are All You Need
    "2305.16264",   # AlpacaEval
    "2311.12022",   # MMLU-Pro
    "2306.05685",   # Decontamination of Benchmarks
    "2304.06767",   # Generative Agents

    # Additional variety (shorter/longer papers)
    "2309.10305",   # Textbooks Are All You Need II
    "2311.10770",   # Orca 2
    "2402.13228",   # Gemma
    "2309.16609",   # Mistral 7B Instruct
    "2401.04088",   # Mixtral of Experts (extended)
]


def download_papers(
    arxiv_ids: list = ARXIV_IDS,
    output_dir: str = "./data/sample_pdfs",
    delay: float = 1.0,
) -> None:
    """Download arXiv papers as PDFs.

    Args:
        arxiv_ids: List of arXiv paper IDs.
        output_dir: Directory to save PDFs.
        delay: Seconds to wait between downloads (be polite to arXiv).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for i, arxiv_id in enumerate(arxiv_ids, 1):
        filename = f"{arxiv_id.replace('/', '_').replace('.', '_')}.pdf"
        output_path = out / filename

        if output_path.exists() and output_path.stat().st_size > 1000:
            print(f"  [{i}/{len(arxiv_ids)}] Skipping {arxiv_id} (already exists)")
            skipped += 1
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"  [{i}/{len(arxiv_ids)}] Downloading {arxiv_id}...", end=" ", flush=True)

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and len(response.content) > 1000:
                output_path.write_bytes(response.content)
                size_kb = len(response.content) / 1024
                print(f"✓ ({size_kb:.0f} KB)")
                downloaded += 1
            else:
                print(f"✗ (status {response.status_code})")
                failed += 1
        except Exception as e:
            print(f"✗ ({e})")
            failed += 1

        # Be polite to arXiv servers
        if i < len(arxiv_ids):
            time.sleep(delay)

    print(f"\n✓ Done: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    print(f"  Total PDFs in {output_dir}: {len(list(out.glob('*.pdf')))}")


def create_subsets(source_dir: str = "./data/sample_pdfs") -> None:
    """Create subset folders for scaling benchmarks.

    Creates sample_pdfs_5, sample_pdfs_20, sample_pdfs_50 with symlinks.

    Args:
        source_dir: Directory containing the full PDF corpus.
    """
    source = Path(source_dir)
    pdfs = sorted(source.glob("*.pdf"))

    for size in [5, 20, 50]:
        subset_dir = source.parent / f"sample_pdfs_{size}"
        subset_dir.mkdir(parents=True, exist_ok=True)

        for pdf in pdfs[:size]:
            link = subset_dir / pdf.name
            if not link.exists():
                link.symlink_to(pdf.resolve())

        actual = len(list(subset_dir.glob("*.pdf")))
        print(f"✓ {subset_dir}: {actual} PDFs")


if __name__ == "__main__":
    print("=== Downloading arXiv papers ===\n")
    download_papers()
    print("\n=== Creating benchmark subsets ===\n")
    create_subsets()
