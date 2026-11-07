"""Post-processing: fill gaps deterministically and drop values the resume does not support.

Small LLMs sometimes invent values (e.g. spoken languages or soft skills that are not in the
resume). This module verifies LLM output against the source text ("grounding") and merges in
rule-based extraction where it is more reliable.
"""
import re

from app.models import ResumeData
from app.services.fallback import apply_contact_fallbacks, dedupe, parse_skill_lines
from app.utils.logger import logger


def compact(value: str) -> str:
    """Lowercase and strip everything except letters/digits (for fuzzy containment checks)."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def ground_list(items: list[str], compact_text: str) -> list[str]:
    """Keep only items whose text appears in the resume."""
    return [i for i in items if compact(i) and compact(i) in compact_text]


def appears_as_item(term: str, text: str) -> bool:
    """True if `term` appears as a standalone list item (not inside a sentence)."""
    pattern = r"(?:^|[,;:•·|\n])\s*" + re.escape(term.strip()) + r"\s*(?:[,;.|\n]|$)"
    return re.search(pattern, text, re.I | re.M) is not None


def ground_degree(degree: str, text_tokens: set[str]) -> str:
    """Drop degree words that never appear in the resume (e.g. an invented major)."""
    kept = [word for word in degree.split() if compact(word) in text_tokens]
    return " ".join(kept) if kept else degree


def finalize_resume(data: ResumeData, text: str) -> ResumeData:
    """Apply contact fallbacks, rule-based skills, grounding and junk-entry cleanup."""
    data = apply_contact_fallbacks(data, text)
    compact_text = compact(text)
    text_tokens = {compact(w) for w in text.split()} - {""}

    # Skills: prefer the deterministic parse of the SKILLS section; otherwise use grounded LLM output.
    parsed_tech, parsed_soft = parse_skill_lines(text)
    technical = parsed_tech or ground_list(data.skills.technical, compact_text)
    soft = parsed_soft or [s for s in data.skills.soft if appears_as_item(s, text)]
    skills = data.skills.model_copy(update={"technical": dedupe(technical), "soft": dedupe(soft)})

    # Work experience: drop entries with neither company nor role (usually mis-parsed lines).
    experience = [j for j in data.work_experience if j.company.strip() or j.role.strip()]

    # Education: remove invented degree words and empty entries.
    education = []
    for edu in data.education:
        if not (edu.degree.strip() or edu.institution.strip()):
            continue
        education.append(edu.model_copy(update={"degree": ground_degree(edu.degree, text_tokens)}))

    projects = [
        p.model_copy(update={"technologies": dedupe(ground_list(p.technologies, compact_text))})
        for p in data.projects
    ]
    languages = dedupe(ground_list(data.languages, compact_text))

    logger.info(
        "Post-processing: skills={} experience={} education={} languages={}",
        len(skills.technical), len(experience), len(education), len(languages),
    )
    return data.model_copy(
        update={
            "skills": skills,
            "work_experience": experience,
            "education": education,
            "projects": projects,
            "languages": languages,
        }
    )
