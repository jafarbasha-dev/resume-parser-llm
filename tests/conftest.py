"""Shared fixtures: isolated storage directories and PDF builders (no LLM or network needed)."""
import pymupdf
import pytest

from app import config

SAMPLE_RESUME = (
    "Jane Doe\n"
    "Software Engineer\n"
    "jane@example.com | +91 98765 43210 | linkedin.com/in/jane-doe\n"
    "EXPERIENCE\n"
    "Acme Corp - Backend Engineer - 2020 to Present\n"
    "Built REST APIs with Python and FastAPI.\n"
)


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Send uploads and stored results to a temp folder so tests never touch real data."""
    uploads = tmp_path / "uploads"
    results = tmp_path / "results"
    uploads.mkdir()
    results.mkdir()
    monkeypatch.setattr(config, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(config, "RESULTS_DIR", results)


@pytest.fixture
def make_pdf():
    """Factory that builds a small text PDF and returns its bytes."""

    def _make(text: str = SAMPLE_RESUME) -> bytes:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        data = doc.tobytes()
        doc.close()
        return data

    return _make


@pytest.fixture
def blank_pdf() -> bytes:
    """A valid PDF with no text (like a scanned/image-only resume)."""
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data
