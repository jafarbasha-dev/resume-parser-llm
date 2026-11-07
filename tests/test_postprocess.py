"""Tests for grounding checks and result clean-up."""
from app.models import ContactInfo, Education, ResumeData, Skills, WorkExperience
from app.services.postprocess import (
    appears_as_item,
    compact,
    finalize_resume,
    ground_degree,
    ground_list,
)

RESUME_TEXT = """Jane Doe
jane@example.com
EDUCATION
BE Sri Eshwar College 2014 - 2018
SKILLS
Tools : Python, FastAPI
Cloud : AWS
PROJECTS
Demo project
"""


def test_ground_list_drops_values_not_in_text():
    assert ground_list(["Python", "Kubernetes"], compact("Built things with Python")) == ["Python"]


def test_appears_as_item_distinguishes_lists_from_sentences():
    assert appears_as_item("Leadership", "Soft skills: Teamwork, Leadership\n")
    assert not appears_as_item("communication", "tool integration, and agent communication to develop")


def test_ground_degree_removes_invented_words():
    assert ground_degree("B.E. Computer Science", {"be", "sri"}) == "B.E."


def test_ground_degree_keeps_original_when_nothing_matches():
    assert ground_degree("MTech", {"foo"}) == "MTech"


def test_finalize_resume_cleans_llm_output():
    data = ResumeData(
        contact=ContactInfo(name="Jane Doe"),
        work_experience=[
            WorkExperience(company="Acme Corp", role="Engineer"),
            WorkExperience(company="", role="", duration="BE - Sri Eshwar"),  # junk entry
        ],
        education=[Education(degree="B.E. Computer Science", institution="Sri Eshwar College")],
        skills=Skills(technical=["Hallucinated Tool"], soft=["teamwork"]),
        languages=["English", "Hindi"],  # not present in the resume
    )

    result = finalize_resume(data, RESUME_TEXT)

    assert [j.company for j in result.work_experience] == ["Acme Corp"]
    assert result.education[0].degree == "B.E."
    assert result.skills.technical == ["Python", "FastAPI", "AWS"]  # parsed from SKILLS section
    assert result.skills.soft == []
    assert result.languages == []


def test_finalize_resume_uses_grounded_llm_skills_when_no_skills_section():
    text = "Built services with Python and Docker.\n"
    data = ResumeData(skills=Skills(technical=["Python", "Kubernetes"]))

    result = finalize_resume(data, text)

    assert result.skills.technical == ["Python"]
