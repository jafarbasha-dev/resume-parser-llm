"""Central configuration loaded from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "results"

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()  # "ollama" or "anthropic"

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_NUM_CTX: int = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_MODE: str = os.getenv("LLM_MODE", "multi").lower()  # "multi" (3 focused calls) or "single"
LLM_KEEP_ALIVE: str = os.getenv("LLM_KEEP_ALIVE", "30m")
LLM_NUM_PREDICT: int = int(os.getenv("LLM_NUM_PREDICT", "3000"))
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "300"))

MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
MAX_TEXT_CHARS: int = int(os.getenv("MAX_TEXT_CHARS", "20000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

ACTIVE_MODEL: str = ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else LLM_MODEL
