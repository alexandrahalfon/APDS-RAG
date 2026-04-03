"""Local PDF processing using pdfplumber (no API calls)."""

from pathlib import Path
from typing import List, Dict
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF with page markers.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full text with '## Page N' markers between pages.
    """
    pdf_path = Path(pdf_path)
    text_parts: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            text_parts.append(f"\n\n## Page {i}\n\n{page_text}")

    return "".join(text_parts).strip()


def _split_paragraphs(text: str) -> List[Dict]:
    """Split text into paragraphs on double newlines.

    Args:
        text: Raw page text.

    Returns:
        List of paragraph dicts with 'text' and 'word_count'.
    """
    import re

    text = re.sub(r'\n{3,}', '\n\n', text)
    raw_paragraphs = text.split('\n\n')
    paragraphs: List[Dict] = []

    for para in raw_paragraphs:
        cleaned = para.strip().replace('\n', ' ')
        word_count = len(cleaned.split())
        if word_count > 3:
            paragraphs.append({
                'text': cleaned,
                'word_count': word_count,
            })

    return paragraphs


def process_pdf_complete_local(pdf_path: str) -> dict:
    """Process a PDF into structured data matching the original pipeline format.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with keys: file_name, title, pages (list of page dicts).
    """
    pdf_path = Path(pdf_path)

    pages: List[Dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            paragraphs = _split_paragraphs(page_text)

            pages.append({
                'page_number': i,
                'text': page_text,
                'paragraphs': paragraphs,
            })

    # Use the filename (without extension) as the title
    title = pdf_path.stem.replace('_', ' ').replace('-', ' ').title()

    return {
        'file_name': pdf_path.name,
        'title': title,
        'pages': pages,
    }
