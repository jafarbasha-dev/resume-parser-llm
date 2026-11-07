"""Tests for the LLM service with the network call mocked out."""
import json

import httpx
import pytest

from app import config
from app.services import llm_service
from app.services.llm_service import LLMServiceError, LLMUnavailableError, extract_resume_data


@pytest.fixture(autouse=True)
def llm_config(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "LLM_MODE", "multi")
    monkeypatch.setattr(config, "MAX_TEXT_CHARS", 20000)


def _fake_multi_pass(text, schema, system):
    """Return a plausible JSON answer depending on which pass is being run."""
    if llm_service.FOCUS_EXPERIENCE in system:
        return json.dumps(
            {
                "work_experience": [
                    {
                        "company": "Acme",
                        "role": "Engineer",
                        "duration": "2020 - Present",
                        "location": "",
                        "responsibilities": ["Built APIs"],
                    }
                ]
            }
        )
    if llm_service.FOCUS_BASICS in system:
        return json.dumps(
            {"contact": {"name": "Jane Doe", "email": "jane@example.com"}, "summary": "Hi",
             "education": [], "languages": []}
        )
    return json.dumps({"skills": {"technical": ["Python"], "soft": []}, "certifications": [], "projects": []})


def test_multi_mode_merges_all_passes(monkeypatch):
    monkeypatch.setattr(llm_service, "_call_ollama", _fake_multi_pass)

    result = extract_resume_data("some resume text")

    assert result.contact.name == "Jane Doe"
    assert result.work_experience[0].company == "Acme"
    assert result.skills.technical == ["Python"]


def test_single_mode_makes_one_call(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "single")
    calls = []

    def fake(text, schema, system):
        calls.append(system)
        return json.dumps({"contact": {"name": "Jane Doe"}})

    monkeypatch.setattr(llm_service, "_call_ollama", fake)

    result = extract_resume_data("some resume text")

    assert len(calls) == 1
    assert result.contact.name == "Jane Doe"


def test_one_invalid_pass_gives_partial_result(monkeypatch):
    calls = []

    def fake(text, schema, system):
        calls.append(system)
        if llm_service.FOCUS_EXPERIENCE in system:
            return "this is not json"
        return _fake_multi_pass(text, schema, system)

    monkeypatch.setattr(llm_service, "_call_ollama", fake)

    result = extract_resume_data("some resume text")

    assert result.work_experience == []  # failed pass is skipped...
    assert result.contact.name == "Jane Doe"  # ...but other passes still count
    assert len(calls) == 4  # basics(1) + experience(2 attempts) + skills(1)


def test_all_passes_invalid_raises(monkeypatch):
    monkeypatch.setattr(llm_service, "_call_ollama", lambda text, schema, system: "garbage")

    with pytest.raises(LLMServiceError):
        extract_resume_data("some resume text")


def test_unavailable_llm_is_not_swallowed(monkeypatch):
    def fake(text, schema, system):
        raise LLMUnavailableError("Ollama is down")

    monkeypatch.setattr(llm_service, "_call_ollama", fake)

    with pytest.raises(LLMUnavailableError):
        extract_resume_data("some resume text")


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "something-else")

    with pytest.raises(LLMServiceError, match="Unknown LLM_PROVIDER"):
        extract_resume_data("text")


def test_input_text_is_truncated(monkeypatch):
    monkeypatch.setattr(config, "MAX_TEXT_CHARS", 10)
    seen = []

    def fake(text, schema, system):
        seen.append(text)
        return _fake_multi_pass(text, schema, system)

    monkeypatch.setattr(llm_service, "_call_ollama", fake)

    extract_resume_data("x" * 500)

    assert all(len(t) <= 10 for t in seen)


def test_connection_error_becomes_unavailable(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm_service.httpx, "post", boom)

    with pytest.raises(LLMUnavailableError, match="Ollama"):
        llm_service._call_ollama("text", {}, "system")


def test_timeout_becomes_service_error(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(llm_service.httpx, "post", boom)

    with pytest.raises(LLMServiceError, match="timed out"):
        llm_service._call_ollama("text", {}, "system")


def test_missing_model_gives_helpful_message(monkeypatch):
    response = httpx.Response(404, request=httpx.Request("POST", "http://localhost:11434/api/chat"))
    monkeypatch.setattr(llm_service.httpx, "post", lambda *a, **k: response)

    with pytest.raises(LLMServiceError, match="ollama pull"):
        llm_service._call_ollama("text", {}, "system")


def test_anthropic_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")

    with pytest.raises(LLMUnavailableError, match="ANTHROPIC_API_KEY"):
        llm_service._call_anthropic("text", {}, "system")
