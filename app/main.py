"""FastAPI application: resume upload and retrieval endpoints."""
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.concurrency import run_in_threadpool

from app import config
from app.models import ErrorResponse, HealthResponse, ResumeRecord
from app.services import storage
from app.services.postprocess import finalize_resume
from app.services.llm_service import LLMServiceError, LLMUnavailableError, extract_resume_data
from app.services.parser import (
    EmptyDocumentError,
    ResumeParsingError,
    UnsupportedFileTypeError,
    extract_text,
    validate_file,
)
from app.utils.logger import logger, setup_logger

setup_logger(config.LOG_LEVEL)
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Resume Parser API",
    description="Upload a PDF/DOCX resume and get structured data extracted by an LLM.",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected errors with a traceback and return a readable message."""
    logger.exception("Unhandled error on {}", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the root URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse(status="ok", llm_model=f"{config.LLM_PROVIDER}:{config.ACTIVE_MODEL}")


@app.post(
    "/api/upload",
    response_model=ResumeRecord,
    status_code=status.HTTP_201_CREATED,
    tags=["resume"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or unsupported file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "No readable text in file"},
        502: {"model": ErrorResponse, "description": "LLM returned an error/invalid output"},
        503: {"model": ErrorResponse, "description": "LLM service unavailable"},
    },
)
async def upload_resume(file: UploadFile = File(...)) -> ResumeRecord:
    """Accept a PDF/DOCX resume, extract structured data, and return it with a document ID."""
    filename = file.filename or "resume"
    content = await file.read()
    logger.info("Upload received: {} ({} bytes)", filename, len(content))

    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds the {config.MAX_UPLOAD_MB} MB limit.",
        )

    try:
        suffix = validate_file(filename, content)
        text = await run_in_threadpool(extract_text, suffix, content)
    except EmptyDocumentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except (UnsupportedFileTypeError, ResumeParsingError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    document_id = uuid4()
    (config.UPLOAD_DIR / f"{document_id}{suffix}").write_bytes(content)

    try:
        data = await run_in_threadpool(extract_resume_data, text)
    except LLMUnavailableError as exc:
        logger.error("LLM unavailable: {}", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMServiceError as exc:
        logger.error("LLM error: {}", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    data = finalize_resume(data, text)

    record = ResumeRecord(
        document_id=document_id,
        filename=filename,
        uploaded_at=datetime.now(timezone.utc),
        data=data,
    )
    storage.save_record(record)
    logger.info("Resume processed successfully: {}", document_id)
    return record


@app.get(
    "/api/resume/{document_id}",
    response_model=ResumeRecord,
    tags=["resume"],
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
def get_resume(document_id: UUID) -> ResumeRecord:
    """Return previously extracted resume data."""
    record = storage.load_record(document_id)
    if record is None:
        logger.warning("Document not found: {}", document_id)
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document {document_id} not found.")
    return record
