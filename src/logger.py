"""Centralised logging setup so every module logs in one consistent format."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once (idempotent).

    Why a function and not import-time side effects: importing a module should
    never reconfigure global state; only the entry point (``main.py``) decides
    verbosity and handlers.
    """
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. re-running in a notebook)
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger (thin wrapper for easy future swapping)."""
    return logging.getLogger(name)
