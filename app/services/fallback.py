"""Deterministic (regex/heuristic) extraction used to fill gaps left by the LLM."""
import re

from app.models import ResumeData

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_CANDIDATE_RE = re.compile(r"\+?\d[\d\s\-().]{8,}\d")
YEAR_RANGE_RE = re.compile(r"^(19|20)\d{2}\s*[-–]\s*((19|20)\d{2}|present)$", re.I)
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:[\w]+\.)?linkedin\.com/in/[\w\-%]+/?", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+", re.I)

NOT_NAMES = {"resume", "curriculum vitae", "cv", "profile", "summary", "contact", "objective"}


def find_email(text: str) -> str:
    """Return the first email address found, or ''."""
    match = EMAIL_RE.search(text)
    return match.group(0) if match else ""


def find_phone(text: str) -> str:
    """Return the first plausible phone number, or ''."""
    for match in PHONE_CANDIDATE_RE.finditer(text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 13 and not YEAR_RANGE_RE.match(candidate):
            return candidate
    return ""


def find_name(text: str) -> str:
    """Guess the name from the first few lines (2-4 alphabetic words)."""
    for line in text.splitlines()[:8]:
        line = line.strip()
        if not line or line.lower() in NOT_NAMES or "@" in line or any(c.isdigit() for c in line):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z.\-']+", w) for w in words):
            return line
    return ""


def apply_contact_fallbacks(data: ResumeData, text: str) -> ResumeData:
    """Fill empty contact fields using regex/heuristics on the raw text."""
    contact = data.contact.model_copy()
    contact.email = contact.email or find_email(text)
    contact.phone = contact.phone or find_phone(text)
    contact.name = contact.name or find_name(text)
    if not contact.linkedin and (m := LINKEDIN_RE.search(text)):
        contact.linkedin = m.group(0)
    if not contact.github and (m := GITHUB_RE.search(text)):
        contact.github = m.group(0)
    return data.model_copy(update={"contact": contact})


# ---------------------------------------------------------------------------
# Skills section parser: handles "Category : item, item, item" layouts.
# ---------------------------------------------------------------------------
SKILLS_HEADING_RE = re.compile(
    r"^(?:(?:key|core|technical|it)\s+)?skills?(?:\s+(?:summary|set|&\s*tools|and\s+tools))?\s*:?$"
    r"|^technical\s+(?:expertise|proficiency)\s*:?$",
    re.I,
)
SECTION_STOP_RE = re.compile(
    r"^(?:projects?|education|certifications?|certificates?|experience|work\s+experience|"
    r"professional\s+experience|career\s+background|achievements?|awards?|languages?|interests|"
    r"hobbies|references?|declaration|personal\s+details|summary|objective)\s*:?$",
    re.I,
)
LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z&/'’\- ]{0,40}?)\s*:\s*(.*)$")
SOFT_LABEL_RE = re.compile(r"soft|interpersonal", re.I)


def _is_section_heading(line: str) -> bool:
    if ":" in line and not line.rstrip().endswith(":"):
        return False
    if SECTION_STOP_RE.match(line):
        return True
    letters = re.sub(r"[^A-Za-z]", "", line)
    return line.isupper() and len(letters) >= 5 and len(line.split()) <= 4 and not any(c.isdigit() for c in line)


def dedupe(items: list[str]) -> list[str]:
    """Remove duplicates (case-insensitive), keeping first occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def parse_skill_lines(text: str) -> tuple[list[str], list[str]]:
    """Parse 'Label : a, b, c' lines inside the SKILLS section.

    Returns (technical, soft). Both are empty when no labelled skills section is found,
    in which case the LLM result is used instead.
    """
    lines = [line.strip() for line in text.splitlines()]
    start = next((i for i, line in enumerate(lines) if SKILLS_HEADING_RE.match(line)), None)
    if start is None:
        return [], []

    entries: list[list[str]] = []  # [label, accumulated text]
    for line in lines[start + 1 :]:
        if not line:
            continue
        if _is_section_heading(line):
            break
        match = LABEL_RE.match(line)
        if match and len(match.group(1).split()) <= 5:
            entries.append([match.group(1).strip(), match.group(2).strip()])
        elif entries:  # wrapped continuation of the previous line
            entries[-1][1] += " " + line

    technical: list[str] = []
    soft: list[str] = []
    for label, body in entries:
        items = [i.strip(" .\t") for i in re.split(r"[,;•·|]", body)]
        items = [i for i in items if i and len(i) <= 60]
        (soft if SOFT_LABEL_RE.search(label) else technical).extend(items)
    return dedupe(technical), dedupe(soft)
