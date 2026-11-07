"""Tests for the regex/heuristic extractors and the skills-section parser."""
from app.models import ContactInfo, ResumeData
from app.services.fallback import (
    apply_contact_fallbacks,
    find_email,
    find_name,
    find_phone,
    parse_skill_lines,
)

SKILLS_TEXT = """SKILLS
GEN AI : RAG, Agentic AI, Langchain, Fine
Tuning - Open AI Models, MCP
Cloud : AWS, AZURE, AWS S3
Soft Skills : Communication, Teamwork
PROJECTS
Demo : bar, baz
"""


def test_find_email():
    assert find_email("Contact: jane.doe+cv@example.co.in today") == "jane.doe+cv@example.co.in"
    assert find_email("no email here") == ""


def test_find_phone_accepts_real_numbers():
    assert find_phone("Call +91 98765 43210 anytime") == "+91 98765 43210"
    assert find_phone("6369650898") == "6369650898"


def test_find_phone_ignores_year_ranges():
    assert find_phone("Worked there 2019 - 2023 as engineer") == ""


def test_find_name_skips_headings_and_contact_lines():
    text = "Resume\njane@example.com\nJane Doe\nSoftware Engineer"
    assert find_name(text) == "Jane Doe"


def test_apply_contact_fallbacks_fills_only_empty_fields():
    text = "Jane Doe\njane@example.com | +91 98765 43210 | linkedin.com/in/jane-doe\n"
    data = ResumeData(contact=ContactInfo(email="keep@example.com"))

    result = apply_contact_fallbacks(data, text)

    assert result.contact.email == "keep@example.com"  # existing value is never overwritten
    assert result.contact.name == "Jane Doe"
    assert result.contact.phone == "+91 98765 43210"
    assert result.contact.linkedin == "linkedin.com/in/jane-doe"


def test_parse_skill_lines_handles_wrapped_lines_and_stops_at_next_section():
    technical, soft = parse_skill_lines(SKILLS_TEXT)

    assert technical == [
        "RAG", "Agentic AI", "Langchain", "Fine Tuning - Open AI Models", "MCP",
        "AWS", "AZURE", "AWS S3",
    ]
    assert soft == ["Communication", "Teamwork"]
    assert "bar" not in technical  # PROJECTS section must not leak in


def test_parse_skill_lines_without_skills_section():
    assert parse_skill_lines("Jane Doe\nExperience\nAcme Corp") == ([], [])
