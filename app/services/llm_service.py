"""LLM integration for structured resume extraction (Ollama or Anthropic Claude).

Two modes (LLM_MODE):
- "multi"  (default): three small, focused calls (basics / experience / skills).
  Much more reliable with small local models.
- "single": one call for the whole schema (faster; fine for large models or Claude).
"""
import time
from typing import Any, List, Type

import httpx
from pydantic import BaseModel, Field, ValidationError

from app import config
from app.models import (
    Certification,
    ContactInfo,
    Education,
    Project,
    ResumeData,
    Skills,
    WorkExperience,
)
from app.utils.logger import logger

MAX_ATTEMPTS = 2
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_TOOL = "save_resume_data"

COMMON_RULES = """You are an expert resume parser. Return JSON matching the provided schema.

Rules:
- Use ONLY information present in the resume. Never invent or guess values.
- If a field is not present, use an empty string "" or an empty list [].
- Keep dates exactly as written (e.g. "Jan 2020 - Present").
- The candidate's name is normally the first line of the resume.
- Output valid JSON only."""

FOCUS_ALL = (
    "Extract every section. Work experience sections may be titled 'Professional Experience', "
    "'Work History', 'Employment', 'Career' or 'Internship'; include every job with company, role, "
    "dates and responsibilities. education.degree is the degree itself; education.institution is the "
    "college. languages are spoken human languages only. skills.technical are tools, languages, "
    "frameworks and platforms; skills.soft are soft skills."
)
FOCUS_BASICS = (
    "Extract ONLY: contact details (name, email, phone, location, LinkedIn/GitHub/website), the "
    "professional summary or objective, education entries, and spoken human languages (English, Hindi...) "
    "- never programming languages. education.degree is the degree (e.g. 'B.E. Computer Science'); "
    "education.institution is the college."
)
FOCUS_EXPERIENCE = (
    "Extract ONLY the work experience: every job and internship the candidate held. The section may be "
    "titled 'Professional Experience', 'Work History', 'Employment', 'Career' or 'Internship'. For each "
    "job give company, role, duration, location and the responsibilities/achievements as separate short "
    "strings. If the resume mentions any employer or job title, do NOT return an empty list."
)
FOCUS_SKILLS = (
    "Extract ONLY: skills (technical = tools, programming languages, frameworks, cloud platforms, "
    "databases; soft = communication, leadership, teamwork...), certifications/courses, and projects "
    "(name, short description, technologies)."
)


class _BasicsPart(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str = ""
    education: List[Education] = Field(default_factory=list)
    languages: List[str] = Field(
        default_factory=list, description="Spoken human languages only, NOT programming languages"
    )


class _ExperiencePart(BaseModel):
    work_experience: List[WorkExperience] = Field(
        default_factory=list, description="All jobs and internships"
    )


class _SkillsPart(BaseModel):
    skills: Skills = Field(default_factory=Skills)
    certifications: List[Certification] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)


MULTI_PASSES: List[tuple[str, Type[BaseModel], str]] = [
    ("basics", _BasicsPart, FOCUS_BASICS),
    ("experience", _ExperiencePart, FOCUS_EXPERIENCE),
    ("skills", _SkillsPart, FOCUS_SKILLS),
]
SINGLE_PASSES: List[tuple[str, Type[BaseModel], str]] = [("all", ResumeData, FOCUS_ALL)]


class LLMServiceError(Exception):
    """Generic LLM failure (timeout, HTTP error, bad output)."""


class LLMUnavailableError(LLMServiceError):
    """The LLM server cannot be reached or is not configured."""


class LLMOutputError(LLMServiceError):
    """The LLM answered, but the output failed schema validation."""


def _post(url: str, **kwargs: Any) -> httpx.Response:
    """POST with uniform error translation."""
    try:
        response = httpx.post(url, timeout=config.LLM_TIMEOUT, **kwargs)
        response.raise_for_status()
        return response
    except httpx.ConnectError as exc:
        raise LLMUnavailableError(f"Cannot connect to the LLM at {url}.") from exc
    except httpx.TimeoutException as exc:
        raise LLMServiceError("The LLM timed out. Try a smaller model or raise LLM_TIMEOUT.") from exc
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 404 and config.LLM_PROVIDER == "ollama":
            raise LLMServiceError(
                f"Model '{config.LLM_MODEL}' not found. Run: ollama pull {config.LLM_MODEL}"
            ) from exc
        if code in (401, 403):
            raise LLMServiceError("LLM provider rejected the API key.") from exc
        raise LLMServiceError(f"LLM server returned HTTP {code}.") from exc
    except httpx.HTTPError as exc:
        raise LLMServiceError(f"LLM request failed: {type(exc).__name__}: {exc}") from exc


def _call_ollama(text: str, schema: dict, system: str) -> Any:
    """Call Ollama with schema-constrained output; returns the raw JSON string."""
    payload = {
        "model": config.LLM_MODEL,
        "stream": False,
        "keep_alive": config.LLM_KEEP_ALIVE,
        "format": schema,
        "options": {
            "temperature": 0,
            "num_ctx": config.LLM_NUM_CTX,
            "num_predict": config.LLM_NUM_PREDICT,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Resume text:\n\n{text}"},
        ],
    }
    try:
        response = _post(f"{config.LLM_BASE_URL}/api/chat", json=payload)
    except LLMUnavailableError as exc:
        raise LLMUnavailableError(
            f"Cannot connect to Ollama at {config.LLM_BASE_URL}. Is it running?"
        ) from exc
    return response.json().get("message", {}).get("content", "")


def _call_anthropic(text: str, schema: dict, system: str) -> Any:
    """Call Claude using a forced tool call so the output matches the schema; returns a dict."""
    if not config.ANTHROPIC_API_KEY:
        raise LLMUnavailableError("ANTHROPIC_API_KEY is not set.")
    payload = {
        "model": config.ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": f"Resume text:\n\n{text}"}],
        "tools": [
            {
                "name": ANTHROPIC_TOOL,
                "description": "Save the structured data extracted from the resume.",
                "input_schema": schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": ANTHROPIC_TOOL},
    }
    headers = {
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = _post(ANTHROPIC_URL, json=payload, headers=headers).json()
    for block in body.get("content", []):
        if block.get("type") == "tool_use":
            return block.get("input", {})
    return {}


def _run_pass(name: str, model_cls: Type[BaseModel], focus: str, text: str) -> BaseModel:
    """Run one extraction pass with retry on invalid output."""
    call = _call_anthropic if config.LLM_PROVIDER == "anthropic" else _call_ollama
    system = f"{COMMON_RULES}\n\nTask: {focus}"
    schema = model_cls.model_json_schema()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        raw = call(text, schema, system)
        elapsed = time.perf_counter() - started
        logger.info(
            "LLM pass '{}' attempt {}/{} finished in {:.1f}s (provider={}, model={})",
            name, attempt, MAX_ATTEMPTS, elapsed, config.LLM_PROVIDER, config.ACTIVE_MODEL,
        )
        try:
            if isinstance(raw, dict):
                return model_cls.model_validate(raw)
            return model_cls.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("Pass '{}' output failed validation (attempt {}): {}", name, attempt, exc)

    raise LLMOutputError(f"LLM returned invalid output for '{name}' after {MAX_ATTEMPTS} attempts.")


def extract_resume_data(text: str) -> ResumeData:
    """Extract structured data from resume text using the configured provider and mode."""
    if config.LLM_PROVIDER not in ("ollama", "anthropic"):
        raise LLMServiceError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use 'ollama' or 'anthropic'.")

    text = text[: config.MAX_TEXT_CHARS]
    passes = SINGLE_PASSES if config.LLM_MODE == "single" else MULTI_PASSES

    merged: dict[str, Any] = {}
    failures: list[str] = []
    for name, model_cls, focus in passes:
        try:
            part = _run_pass(name, model_cls, focus, text)
        except LLMOutputError as exc:  # keep partial results if one pass produces junk
            logger.error("{}", exc)
            failures.append(name)
            continue
        merged.update(part.model_dump())

    if not merged:
        raise LLMServiceError("LLM returned invalid output for every extraction pass.")
    if failures:
        logger.warning("Partial result: passes failed: {}", ", ".join(failures))
    return ResumeData.model_validate(merged)
