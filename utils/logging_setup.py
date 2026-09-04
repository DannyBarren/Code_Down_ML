"""Centralised structlog configuration.

A single ``configure_logging`` call wires up structlog so every module can grab
a logger with ``get_logger(__name__)`` and emit consistent, structured logs to
both the console and a rotating file.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

import structlog

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: Optional[str | Path] = None,
) -> None:
    """Configure structlog + stdlib logging once for the whole process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Keep the optional HF stack quiet and telemetry-free.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handlers.append(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _quiet_noisy_libraries()
    _CONFIGURED = True


def _quiet_noisy_libraries() -> None:
    """Silence chatty third-party loggers (HF model downloads, HTTP, etc.)."""
    for name in ("httpx", "httpcore", "huggingface_hub", "filelock",
                 "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str = "fill_down") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger. Safe to call before configuration."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
