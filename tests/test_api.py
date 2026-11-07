"""End-to-end API tests (FastAPI TestClient) with the LLM mocked out."""
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.models import ContactInfo, ResumeData
from app.services.llm_service import LLMServiceError, LLMUnavailableError

client = TestClient(main.app)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the real LLM call with a canned result."""

    def fake(text: str) -> ResumeData:
        return ResumeData(contact=ContactInfo(name="Jane Doe"), summary="Backend engineer")

    monkeypatch.setattr(main, "extract_resume_data", fake)


def _upload(name: str, content: bytes, mime: str = "application/pdf"):
    return client.post("/api/upload", files={"file": (name, content, mime)})


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_upload_then_fetch_resume(fake_llm, make_pdf):
    response = _upload("resume.pdf", make_pdf())

    assert response.status_code == 201
    body = response.json()
    assert body["data"]["contact"]["name"] == "Jane Doe"
    assert body["data"]["contact"]["email"] == "jane@example.com"  # filled by the regex fallback
    assert len(list(config.UPLOAD_DIR.iterdir())) == 1  # original file was stored

    fetched = client.get(f"/api/resume/{body['document_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["document_id"] == body["document_id"]


def test_get_unknown_document_returns_404():
    response = client.get(f"/api/resume/{uuid4()}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_malformed_id_returns_422():
    assert client.get("/api/resume/not-a-uuid").status_code == 422


def test_upload_rejects_unsupported_type(fake_llm):
    response = _upload("notes.txt", b"hello world", "text/plain")
    assert response.status_code == 400


def test_upload_rejects_empty_file(fake_llm):
    response = _upload("resume.pdf", b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_rejects_oversized_file(fake_llm, make_pdf, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    assert _upload("resume.pdf", make_pdf()).status_code == 413


def test_upload_scanned_pdf_returns_422(fake_llm, blank_pdf):
    response = _upload("scan.pdf", blank_pdf)
    assert response.status_code == 422
    assert "ocr" in response.json()["detail"].lower()


def test_llm_unavailable_returns_503(make_pdf, monkeypatch):
    def fake(text):
        raise LLMUnavailableError("Ollama is not running")

    monkeypatch.setattr(main, "extract_resume_data", fake)
    assert _upload("resume.pdf", make_pdf()).status_code == 503


def test_llm_failure_returns_502(make_pdf, monkeypatch):
    def fake(text):
        raise LLMServiceError("bad output")

    monkeypatch.setattr(main, "extract_resume_data", fake)
    assert _upload("resume.pdf", make_pdf()).status_code == 502


def test_unexpected_error_returns_readable_500(make_pdf, monkeypatch):
    def fake(text):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "extract_resume_data", fake)
    safe_client = TestClient(main.app, raise_server_exceptions=False)

    response = safe_client.post("/api/upload", files={"file": ("r.pdf", make_pdf(), "application/pdf")})

    assert response.status_code == 500
    assert "RuntimeError: boom" in response.json()["detail"]
