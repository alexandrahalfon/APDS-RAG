"""Generate synthetic PDF corpus for benchmarking.

Creates PDFs with realistic text content of varied lengths,
suitable for testing the full RAG pipeline without network access.
"""

import random
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Realistic ML/AI paragraphs for corpus generation
PARAGRAPHS = [
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Rather than being explicitly programmed, these systems improve their performance on a specific task through experience. The field has seen tremendous growth in recent years, driven by advances in computing power, the availability of large datasets, and improvements in algorithms.",

    "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes organized in layers, where each connection has an associated weight. During training, these weights are adjusted through backpropagation to minimize a loss function. Deep neural networks, with many hidden layers, have proven particularly effective for complex tasks.",

    "Natural language processing encompasses a range of computational techniques for analyzing and representing naturally occurring text. Key tasks include tokenization, part-of-speech tagging, named entity recognition, sentiment analysis, and machine translation. Recent advances in transformer architectures have significantly improved performance across these tasks.",

    "Retrieval-augmented generation combines the strengths of retrieval-based and generative approaches to natural language processing. By first retrieving relevant documents from a knowledge base and then conditioning the generation on these documents, RAG systems can produce more accurate and factual responses than purely generative models.",

    "Transformer architectures have revolutionized the field of deep learning since their introduction. The self-attention mechanism allows the model to weigh the importance of different parts of the input when producing each part of the output. This has proven especially powerful for sequence-to-sequence tasks, enabling models to capture long-range dependencies effectively.",

    "Transfer learning involves leveraging knowledge gained from one task to improve performance on a related task. In the context of natural language processing, pre-trained language models like BERT and GPT have demonstrated that representations learned on large text corpora can be fine-tuned for a wide variety of downstream tasks with relatively little task-specific data.",

    "Vector databases are specialized systems designed to store, index, and query high-dimensional vector embeddings efficiently. They support similarity search operations that are fundamental to many modern AI applications, including recommendation systems, image search, and retrieval-augmented generation pipelines.",

    "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models. Variants include stochastic gradient descent, which updates parameters using a single training example, and mini-batch gradient descent, which uses a small subset of training examples. Adam and AdaGrad are popular adaptive learning rate methods.",

    "Convolutional neural networks are specialized for processing grid-like data such as images. They use convolutional layers that apply filters to local regions of the input, enabling the network to learn spatial hierarchies of features. Pooling layers reduce dimensionality, and fully connected layers at the end perform classification.",

    "Recurrent neural networks are designed for sequential data processing. They maintain a hidden state that captures information from previous time steps. Long Short-Term Memory networks and Gated Recurrent Units address the vanishing gradient problem that affects standard RNNs, enabling them to learn long-range dependencies in sequences.",

    "Attention mechanisms allow neural networks to focus on relevant parts of the input when producing each element of the output. Scaled dot-product attention computes a weighted sum of values, where the weights are determined by the compatibility of queries and keys. Multi-head attention applies this mechanism in parallel across different representation subspaces.",

    "Embedding models convert discrete objects like words, sentences, or documents into continuous vector representations. These dense representations capture semantic relationships, enabling mathematical operations like similarity computation. Sentence transformers extend this concept to produce fixed-size representations for variable-length text inputs.",

    "Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space. It is widely used in information retrieval and natural language processing to measure the similarity between document representations. Values range from negative one to one, with one indicating identical direction.",

    "Parallel computing involves performing multiple computations simultaneously. In Python, multiprocessing allows true parallelism by creating separate processes, each with its own memory space. This is particularly useful for CPU-bound tasks like document processing, where the Global Interpreter Lock prevents effective threading.",

    "FAISS is a library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors. It provides implementations of several indexing structures, including flat indices for exact search and approximate nearest neighbor methods for large-scale applications. GPU acceleration is available for many operations.",

    "Batch processing in machine learning refers to processing multiple input samples simultaneously rather than one at a time. This approach leverages hardware parallelism in CPUs and GPUs, significantly improving throughput. The batch size is a hyperparameter that affects both training dynamics and computational efficiency.",

    "Mixed precision training uses both 16-bit and 32-bit floating-point types during model training. By performing most operations in half precision, it reduces memory usage and increases computational speed, particularly on modern GPUs with tensor cores. Automatic mixed precision simplifies the implementation by automatically managing the precision of each operation.",

    "Document chunking is the process of splitting large documents into smaller, semantically meaningful segments. Effective chunking strategies consider sentence boundaries, paragraph structure, and semantic coherence. Overlap between chunks helps maintain context across boundaries, which is important for downstream retrieval tasks.",

    "Information retrieval is the activity of obtaining information system resources that are relevant to an information need from a collection. Boolean retrieval, vector space models, and probabilistic models are classical approaches. Modern neural information retrieval leverages dense representations and learned similarity functions.",

    "The BM25 algorithm is a probabilistic information retrieval function that ranks documents based on the query terms appearing in each document. It considers term frequency, inverse document frequency, and document length normalization. Despite its simplicity, BM25 remains a strong baseline for many retrieval tasks.",
]


def generate_pdf(output_path: Path, num_pages: int, title: str) -> None:
    """Generate a single PDF with realistic ML content.

    Args:
        output_path: Where to save the PDF.
        num_pages: Number of pages to generate.
        title: Document title.
    """
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'], fontSize=20, spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=10,
    )
    body_style = ParagraphStyle(
        'CustomBody', parent=styles['Normal'], fontSize=11, leading=14, spaceAfter=8,
    )

    story = []

    # Title
    story.append(Paragraph(title, title_style))
    story.append(Paragraph("Synthetic Benchmark Document", styles['Normal']))
    story.append(Spacer(1, 30))

    # Generate enough content to fill the target number of pages
    # ~3-4 paragraphs per page with this font size
    rng = random.Random(hash(title))
    paragraphs = list(PARAGRAPHS)
    target_paragraphs = num_pages * 4

    for section in range(num_pages):
        story.append(Paragraph(f"Section {section + 1}: Analysis and Discussion", heading_style))
        story.append(Spacer(1, 6))

        rng.shuffle(paragraphs)
        for para in paragraphs[:rng.randint(3, 5)]:
            story.append(Paragraph(para, body_style))

        story.append(Spacer(1, 12))

    doc.build(story)


def generate_corpus(
    output_dir: str = "./data/sample_pdfs",
    num_papers: int = 50,
) -> None:
    """Generate a synthetic PDF corpus with varied document lengths.

    Args:
        output_dir: Directory to save PDFs.
        num_papers: Total number of PDFs to generate.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    topics = [
        "Transformer Architecture", "Attention Mechanisms", "BERT Pre-training",
        "GPT Language Models", "Retrieval Augmented Generation", "Vector Search",
        "Embedding Optimization", "Neural Information Retrieval", "Document Chunking",
        "Parallel Processing", "GPU Acceleration", "Mixed Precision Training",
        "Transfer Learning", "Fine-tuning Strategies", "Prompt Engineering",
        "Chain of Thought Reasoning", "Knowledge Distillation", "Model Compression",
        "Quantization Methods", "Sparse Attention", "FlashAttention Algorithm",
        "Sentence Embeddings", "Contrastive Learning", "Curriculum Learning",
        "Data Augmentation", "Active Learning", "Few-shot Learning",
        "Zero-shot Classification", "Multi-task Learning", "Federated Learning",
        "Reinforcement Learning from Human Feedback", "Direct Preference Optimization",
        "Constitutional AI", "Instruction Tuning", "In-context Learning",
        "Scaling Laws", "Compute Optimal Training", "Mixture of Experts",
        "Multimodal Models", "Vision Transformers", "Audio Processing",
        "Code Generation", "Mathematical Reasoning", "Structured Prediction",
        "Named Entity Recognition", "Semantic Parsing", "Text Summarization",
        "Question Answering", "Dialogue Systems", "Sentiment Analysis",
        "Machine Translation",
    ]

    # Varied lengths: short (5-8 pages), medium (10-20), long (25-50)
    rng = random.Random(42)
    length_distribution = (
        [(5, 8)] * (num_papers // 3)
        + [(10, 20)] * (num_papers // 3)
        + [(25, 50)] * (num_papers - 2 * (num_papers // 3))
    )
    rng.shuffle(length_distribution)

    print(f"Generating {num_papers} synthetic PDFs...\n")

    for i in range(num_papers):
        topic = topics[i % len(topics)]
        low, high = length_distribution[i]
        num_pages = rng.randint(low, high)
        filename = f"paper_{i+1:03d}_{topic.lower().replace(' ', '_')}.pdf"
        output_path = out / filename

        if output_path.exists():
            print(f"  [{i+1}/{num_papers}] Skipping {filename} (exists)")
            continue

        generate_pdf(output_path, num_pages, f"On {topic}")
        size_kb = output_path.stat().st_size / 1024
        print(f"  [{i+1}/{num_papers}] {filename} ({num_pages} pages, {size_kb:.0f} KB)")

    total = len(list(out.glob("*.pdf")))
    print(f"\n✓ Corpus ready: {total} PDFs in {output_dir}/")


def create_subsets(source_dir: str = "./data/sample_pdfs") -> None:
    """Create subset folders for scaling benchmarks.

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
    generate_corpus()
    print()
    create_subsets()
