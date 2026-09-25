"""Split file logging for the train*.py wrappers.

Routes lerobot's training logs to two files in the repo root:
  - train_log.txt    every INFO+ record (the `step:... loss:...` lines)
  - train_error.txt  ERROR+ records only, plus any uncaught traceback

Why a monkeypatch: lerobot's `train()` calls `init_logging()`, which CLEARS all
root-logger handlers (see lerobot/utils/utils.py) and then adds a console
handler only. So handlers added *before* `train()` get wiped. We wrap
`init_logging` so our two file handlers are re-attached every time it runs.

Usage: call `setup_split_logging()` once at the top of `main()`, before
`train(cfg)`.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import lerobot.scripts.lerobot_train as _train_mod

_ROOT = Path(__file__).resolve().parent
LOG_FILE = _ROOT / "train_log.txt"
ERROR_FILE = _ROOT / "train_error.txt"


def _custom_format(record: logging.LogRecord) -> str:
    # Mirror lerobot.utils.utils.init_logging's console format so the file lines
    # look identical to what used to land in train_error.txt.
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fnameline = f"{record.pathname}:{record.lineno}"
    return f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.getMessage()}"


def _make_handler(path: Path, level: int) -> logging.FileHandler:
    formatter = logging.Formatter()
    formatter.format = _custom_format  # type: ignore[method-assign]
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _attach_file_handlers() -> None:
    root = logging.getLogger()
    root.addHandler(_make_handler(LOG_FILE, logging.INFO))
    root.addHandler(_make_handler(ERROR_FILE, logging.ERROR))


def setup_split_logging() -> None:
    """INFO+ -> train_log.txt, ERROR+ -> train_error.txt (idempotent)."""
    if getattr(_train_mod.init_logging, "_split_log_patched", False):
        return

    # Truncate both files once at process start, mimicking a shell `2> file`
    # redirect, so each run starts with clean logs instead of appending.
    LOG_FILE.write_text("", encoding="utf-8")
    ERROR_FILE.write_text("", encoding="utf-8")

    _orig_init_logging = _train_mod.init_logging

    def _patched(*args, **kwargs):
        # init_logging() clears all root handlers, so re-attach ours afterward.
        _orig_init_logging(*args, **kwargs)
        _attach_file_handlers()

    _patched._split_log_patched = True
    _train_mod.init_logging = _patched

    # Route uncaught exceptions/tracebacks into train_error.txt too (these go
    # straight to stderr, not through logging, so logging handlers miss them).
    def _excepthook(exc_type, exc, tb):
        logging.getLogger().error("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook
