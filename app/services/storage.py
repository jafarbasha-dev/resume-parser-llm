"""Simple JSON-file persistence for extraction results."""
from pathlib import Path
from typing import Optional
from uuid import UUID

from app import config
from app.models import ResumeRecord


def _path(document_id: UUID) -> Path:
    return config.RESULTS_DIR / f"{document_id}.json"


def save_record(record: ResumeRecord) -> None:
    """Persist a record to disk."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _path(record.document_id).write_text(record.model_dump_json(indent=2), encoding="utf-8")


def load_record(document_id: UUID) -> Optional[ResumeRecord]:
    """Load a record, or None if it does not exist."""
    path = _path(document_id)
    if not path.exists():
        return None
    return ResumeRecord.model_validate_json(path.read_text(encoding="utf-8"))
