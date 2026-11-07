"""Pydantic models for LLM output and API request/response bodies."""
from datetime import datetime
from typing import List
from uuid import UUID

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """Candidate contact details."""

    name: str = Field("", description="Full name of the candidate, usually the first line of the resume")
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""


class WorkExperience(BaseModel):
    """A single job entry."""

    company: str = ""
    role: str = ""
    duration: str = Field("", description="e.g. 'Jan 2020 - Present'")
    location: str = ""
    responsibilities: List[str] = Field(default_factory=list)


class Education(BaseModel):
    """A single education entry."""

    degree: str = Field("", description="Degree and field, e.g. 'B.E. Computer Science'")
    institution: str = Field("", description="College/university name")
    year: str = Field("", description="Graduation year or range, e.g. '2015 - 2019'")
    grade: str = Field("", description="CGPA / percentage if present")


class Skills(BaseModel):
    """Technical and soft skills."""

    technical: List[str] = Field(default_factory=list)
    soft: List[str] = Field(default_factory=list)


class Certification(BaseModel):
    """A certification entry."""

    name: str = ""
    issuer: str = ""
    year: str = ""


class Project(BaseModel):
    """A personal/academic project."""

    name: str = ""
    description: str = ""
    technologies: List[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    """Structured information extracted from a resume."""

    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str = ""
    work_experience: List[WorkExperience] = Field(
        default_factory=list, description="All jobs and internships, most recent first as written"
    )
    education: List[Education] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    certifications: List[Certification] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    languages: List[str] = Field(
        default_factory=list, description="Spoken/written human languages only (e.g. English, Hindi), NOT programming languages"
    )


class ResumeRecord(BaseModel):
    """Stored/returned resume extraction result."""

    document_id: UUID
    filename: str
    uploaded_at: datetime
    data: ResumeData


class ErrorResponse(BaseModel):
    """Standard error body."""

    detail: str


class HealthResponse(BaseModel):
    """Health-check body."""

    status: str
    llm_model: str
