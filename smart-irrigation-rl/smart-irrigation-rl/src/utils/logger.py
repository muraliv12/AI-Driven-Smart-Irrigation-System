"""
logger.py
---------
Centralized logging configuration used across every module in the project.
Provides a single `get_logger` factory so all modules share consistent
formatting and log level, and writes both to console and to a rotating
log file under outputs/logs/.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR = "outputs/logs"
_LOG_FILE = os.path.join(_LOG_DIR, "smart_irrigation.log")
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def _configure_root(log_level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Return a configured logger instance for the given module name.

    Parameters
    ----------
    name : str
        Usually `__name__` of the calling module.
    log_level : str
        One of DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Returns
    -------
    logging.Logger
    """
    _configure_root(log_level)
    return logging.getLogger(name)
