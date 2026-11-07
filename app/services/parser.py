"""Text extraction from PDF and DOCX resumes."""
import io
import re
from pathlib import Path

import pymupdf  # PyMuPDF
from docx import Document

from app.utils.logger import logger

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


class ResumeParsingError(Exception):
    """Base error for file parsing problems."""


class UnsupportedFileTypeError(ResumeParsingError):
    """Raised for unsupported or mismatched file types."""


class EmptyDocumentError(ResumeParsingError):
    """Raised when no text can be extracted (e.g. scanned PDF)."""


def validate_file(filename: str, content: bytes) -> str:
    """Validate extension and file signature. Returns the lowercase extension."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError("Only PDF and DOCX files are supported.")
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        raise UnsupportedFileTypeError("File has .pdf extension but is not a valid PDF.")
    if suffix == ".docx" and not content.startswith(b"PK"):
        raise UnsupportedFileTypeError("File has .docx extension but is not a valid DOCX.")
    return suffix


def _extract_pdf(content: bytes) -> str:
    try:
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            pages = [page.get_text("text", sort=True) for page in doc]
            links = {
                link["uri"]
                for page in doc
                for link in page.get_links()
                if link.get("uri")
            }
    except Exception as exc:  # noqa: BLE001 - library raises many types
        raise ResumeParsingError("Could not read the PDF file. It may be corrupted or encrypted.") from exc

    text = "\n".join(pages)
    if links:  # emails / LinkedIn are often hyperlinks whose visible text is just an icon or label
        text += "\n\nLinks found in document:\n" + "\n".join(sorted(links))
    return text


def _extract_docx(content: bytes) -> str:
    try:
        doc = Document(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise ResumeParsingError("Could not read the DOCX file. It may be corrupted.") from exc

    parts: list[str] = []
    for section in doc.sections:  # contact details are often in the page header
        parts.extend(p.text for p in section.header.paragraphs if p.text.strip())
    parts.extend(p.text for p in doc.paragraphs if p.text.strip())
    for table in doc.tables:  # many resumes use tables for layout
        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                text = cell.text.strip()
                if text and text not in cells:
                    cells.append(text)
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_text(suffix: str, content: bytes) -> str:
    """Extract and clean text from a PDF or DOCX file's bytes."""
    raw = _extract_pdf(content) if suffix == ".pdf" else _extract_docx(content)
    text = re.sub(r"[ \t]+", " ", raw)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 50:
        raise EmptyDocumentError(
            "No readable text found. If this is a scanned/image-based resume, OCR is required."
        )
    logger.info("Extracted {} characters of text from {} file", len(text), suffix)
    return text
