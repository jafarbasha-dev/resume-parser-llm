"""Run the full extraction pipeline from the terminal (no API) to see raw errors.

Usage (from project root, venv active):
    python scripts/check_pipeline.py path/to/resume.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.services.llm_service import extract_resume_data  # noqa: E402
from app.services.parser import extract_text, validate_file  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/check_pipeline.py path/to/resume.(pdf|docx)")
    path = Path(sys.argv[1])
    content = path.read_bytes()
    suffix = validate_file(path.name, content)
    text = extract_text(suffix, content)
    print(f"Extracted {len(text)} characters. Provider={config.LLM_PROVIDER} model={config.ACTIVE_MODEL}")
    print("----- EXTRACTED TEXT (first 1500 chars) -----")
    print(text[:1500])
    print("---------------------------------------------")
    print("Calling LLM (can take a minute)...")
    print(extract_resume_data(text).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
