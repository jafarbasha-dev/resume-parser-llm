"""Structured logging setup using loguru."""
import sys

from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    """Configure console and rotating file sinks."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        ),
    )
    logger.add("logs/app.log", rotation="5 MB", retention="7 days", level=level, serialize=True)


__all__ = ["logger", "setup_logger"]
