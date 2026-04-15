"""Document ingestion and chunking for military Field Manuals."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.data_models import DocumentChunk


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file. Tries PyMuPDF first, falls back to pdfplumber."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except ImportError:
        pass
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        pass
    with open(pdf_path, "r", errors="ignore") as f:
        return f.read()


def _identify_fm_name(filename: str) -> str:
    """Extract FM designation from filename (e.g., 'ARN43326-FM_3-0-000-WEB-1.pdf' -> 'FM 3-0')."""
    match = re.search(r"FM[_\s]?(\d+[\-\.]\d+)", filename, re.IGNORECASE)
    if match:
        num = match.group(1).replace("_", "-")
        return f"FM {num}"
    return filename


def _detect_section(text: str, position: int) -> tuple[str, str]:
    """Detect the nearest section/chapter heading before this position."""
    heading_patterns = [
        r"(Chapter\s+\d+)\s*\n\s*([A-Z][A-Z\s,]+)",
        r"(Section\s+[IVX]+)\s*[-–]\s*(.+)",
        r"(Appendix\s+[A-Z])\s*\n\s*([A-Z][A-Z\s,]+)",
        r"(\d+-\d+)\.\s",
    ]
    best_id = "unknown"
    best_title = ""
    for pattern in heading_patterns:
        for m in re.finditer(pattern, text[:position]):
            best_id = m.group(1).strip()
            best_title = m.group(2).strip() if m.lastindex >= 2 else ""
    return best_id, best_title


def chunk_text(
    text: str,
    source_document: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    timestamp: Optional[datetime] = None,
) -> list[DocumentChunk]:
    """Split text into overlapping chunks with metadata."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[DocumentChunk] = []
    current_chunk = ""
    current_section_id = "intro"
    current_section_title = ""
    char_position = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        section_id, section_title = _detect_section(text, char_position)
        if section_id != "unknown":
            current_section_id = section_id
            current_section_title = section_title

        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(DocumentChunk(
                id=f"{source_document}_{len(chunks):04d}",
                text=current_chunk.strip(),
                source_document=source_document,
                section_id=current_section_id,
                section_title=current_section_title,
                timestamp=timestamp,
                metadata={
                    "char_start": char_position - len(current_chunk),
                    "chunk_index": len(chunks),
                },
            ))
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap_text

        current_chunk += " " + para if current_chunk else para
        char_position += len(para) + 2

    if current_chunk.strip():
        chunks.append(DocumentChunk(
            id=f"{source_document}_{len(chunks):04d}",
            text=current_chunk.strip(),
            source_document=source_document,
            section_id=current_section_id,
            section_title=current_section_title,
            timestamp=timestamp,
            metadata={"char_start": char_position - len(current_chunk), "chunk_index": len(chunks)},
        ))

    return chunks


def load_corpus(
    corpus_dir: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[DocumentChunk]:
    """Load all PDFs from a directory and return chunked documents."""
    corpus_dir = Path(corpus_dir)
    all_chunks: list[DocumentChunk] = []

    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        fm_name = _identify_fm_name(pdf_path.name)
        print(f"Processing {pdf_path.name} -> {fm_name}")
        text = _extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, fm_name, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    print(f"Total: {len(all_chunks)} chunks from {len(list(corpus_dir.glob('*.pdf')))} documents")
    return all_chunks
