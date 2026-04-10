"""Fast PDF text extraction using PyMuPDF (fitz).

PyMuPDF is a thin Python wrapper around MuPDF's C library, making it
typically 5-10x faster than pdfplumber for text extraction. pdfplumber
does extensive Python-level layout analysis; PyMuPDF delegates to C.
"""

import re
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF


def _split_paragraphs(text: str) -> List[Dict]:
    """Split text into paragraphs on double newlines.

    Args:
        text: Raw page text.

    Returns:
        List of paragraph dicts with 'text' and 'word_count'.
    """
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


def process_pdf_pymupdf(pdf_path: str) -> dict:
    """Process a PDF into structured data using PyMuPDF (C library).

    Drop-in replacement for process_pdf_complete_local() with the same
    output format, but uses MuPDF's C engine instead of pdfplumber's
    Python layout analysis.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with keys: file_name, title, pages (list of page dicts).
    """
    pdf_path = Path(pdf_path)
    pages: List[Dict] = []

    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc, start=1):
        page_text = page.get_text() or ""
        paragraphs = _split_paragraphs(page_text)

        pages.append({
            'page_number': i,
            'text': page_text,
            'paragraphs': paragraphs,
        })
    doc.close()

    title = pdf_path.stem.replace('_', ' ').replace('-', ' ').title()

    return {
        'file_name': pdf_path.name,
        'title': title,
        'pages': pages,
    }
