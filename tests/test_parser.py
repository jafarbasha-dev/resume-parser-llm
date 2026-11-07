"""Tests for file validation and text extraction."""
import io

import pytest
from docx import Document

from app.services.parser import (
    EmptyDocumentError,
    ResumeParsingError,
    UnsupportedFileTypeError,
    extract_text,
    validate_file,
)


def test_validate_file_accepts_pdf_and_docx():
    assert validate_file("resume.PDF", b"%PDF-1.7 data") == ".pdf"
    assert validate_file("resume.docx", b"PK\x03\x04 data") == ".docx"


def test_validate_file_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        validate_file("notes.txt", b"hello")


def test_validate_file_rejects_mismatched_signature():
    with pytest.raises(UnsupportedFileTypeError):
        validate_file("fake.pdf", b"this is not a pdf")
    with pytest.raises(UnsupportedFileTypeError):
        validate_file("fake.docx", b"this is not a docx")


def test_extract_text_from_pdf(make_pdf):
    text = extract_text(".pdf", make_pdf())
    assert "Jane Doe" in text
    assert "jane@example.com" in text


def test_extract_text_from_docx_includes_tables():
    doc = Document()
    doc.add_paragraph("Jane Doe - Software Engineer with experience in Python and APIs.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skills"
    table.rows[0].cells[1].text = "Python, FastAPI, Docker"
    buffer = io.BytesIO()
    doc.save(buffer)

    text = extract_text(".docx", buffer.getvalue())
    assert "Jane Doe" in text
    assert "Python, FastAPI, Docker" in text


def test_blank_pdf_raises_empty_document_error(blank_pdf):
    with pytest.raises(EmptyDocumentError):
        extract_text(".pdf", blank_pdf)


def test_corrupt_pdf_raises_parsing_error():
    with pytest.raises(ResumeParsingError):
        extract_text(".pdf", b"%PDF-1.4 corrupted bytes")
