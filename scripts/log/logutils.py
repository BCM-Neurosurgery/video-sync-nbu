# scripts/log/logutils.py
"""
Logging utilities for the video-sync pipeline.

Goals
-----
1) One place to configure logs (console + run file + per-camera file).
2) Consistent, concise formatting with safe [seg/cam] stamping.
3) Library modules stay handler-free; drivers control logging.
4) Standalone scripts can enable a minimal console logger without
   interfering with the driver.

Usage (driver)
--------------
from pathlib import Path
import logging
from scripts.log.logutils import configure_logging, log_context, attach_cam_logger

log = configure_logging(Path("/tmp/out"), "INFO")

with log_context(seg="TRBD002_20250806_104707", cam="-"):
    log.info("Decoding…")

adapter, h = attach_cam_logger("TRBD002_20250806_104707", "23512909",
                               Path("/tmp/out/TRBD.../23512909"))
try:
    with log_context(seg="TRBD002_20250806_104707", cam="23512909"):
        adapter.info("Anchors collected.")
finally:
    logging.getLogger().removeHandler(h)
    h.close()

Usage (standalone module)
-------------------------
from scripts.log.logutils import configure_standalone_logging
configure_standalone_logging("INFO", seg="SEGID", cam="CAMSERIAL")
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Tuple, Union

__all__ = [
    "configure_logging",
    "attach_cam_logger",
    "log_context",
    "configure_standalone_logging",
    "sanitize_existing_loggers",
]

# -----------------------------------------------------------------------------
# Context (segment/camera) used by root handlers to stamp logs
# -----------------------------------------------------------------------------
_seg_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("seg", default="-")
_cam_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("cam", default="-")


@contextmanager
def log_context(seg: str = "-", cam: str = "-"):
    """
    Temporarily set current [seg/cam] for all logs within this block.
    Root handlers stamp any missing/placeholder seg/cam from this context.
    """
    t1 = _seg_ctx.set(seg)
    t2 = _cam_ctx.set(cam)
    try:
        yield
    finally:
        _seg_ctx.reset(t1)
        _cam_ctx.reset(t2)


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
class _ContextFilter(logging.Filter):
    """Guarantee %(seg)s and %(cam)s exist so formatters never break."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "seg"):
            record.seg = "-"
        if not hasattr(record, "cam"):
            record.cam = "-"
        return True


class _ContextVarStampingFilter(logging.Filter):
    """
    For root handlers (console + run file): if seg/cam are missing or '-',
    stamp them from current contextvars.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "seg") or record.seg in (None, "-", ""):
            record.seg = _seg_ctx.get()
        if not hasattr(record, "cam") or record.cam in (None, "-", ""):
            record.cam = _cam_ctx.get()
        return True


class _SegCamStampFilter(logging.Filter):
    """
    For per-camera file handler: force seg/cam to the specific pair so that
    file is always correctly tagged—even for logs from other modules.
    """

    def __init__(self, seg: str, cam: str) -> None:
        super().__init__()
        self._seg = seg
        self._cam = cam

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "seg") or record.seg in (None, "-", ""):
            record.seg = self._seg
        if not hasattr(record, "cam") or record.cam in (None, "-", ""):
            record.cam = self._cam
        return True


class _StandaloneSegCamFilter(logging.Filter):
    """
    For standalone console logging: stamp seg/cam using CLI-provided values.
    Only used when the root has no handlers (i.e., not under the driver).
    """

    def __init__(self, seg: str, cam: str) -> None:
        super().__init__()
        self.seg = seg or "-"
        self.cam = cam or "-"

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "seg") or record.seg in (None, "-", ""):
            record.seg = self.seg
        if not hasattr(record, "cam") or record.cam in (None, "-", ""):
            record.cam = self.cam
        return True


# -----------------------------------------------------------------------------
# Formats & constants
# -----------------------------------------------------------------------------
CONSOLE_FORMAT = "[%(levelname)s] [%(seg)s/%(cam)s] %(message)s"
RUN_FILE_FORMAT = "%(asctime)s %(levelname)s [%(seg)s/%(cam)s] %(name)s: %(message)s"
CAM_FILE_FORMAT = "%(asctime)s %(levelname)s [%(seg)s/%(cam)s] %(message)s"

MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _level_to_int(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def sanitize_existing_loggers() -> None:
    """
    Remove handlers installed by other modules so logs propagate to root.
    Prevents duplicate/mismatched console lines like 'INFO:foo:...'.
    """
    mgr = logging.root.manager.loggerDict
    for name, obj in list(mgr.items()):
        if isinstance(obj, logging.Logger):
            for h in obj.handlers[:]:
                obj.removeHandler(h)
            obj.propagate = True


def _attach_rotating_file_handler(
    owner: logging.Logger,
    file_path: Path,
    level: int,
    fmt: str,
    *,
    add_context_filters: bool = True,
) -> RotatingFileHandler:
    """
    Attach a RotatingFileHandler to `owner` unless an equivalent one exists.
    Returns the handler (existing or newly created).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    for h in owner.handlers:
        if isinstance(h, RotatingFileHandler):
            try:
                if (
                    Path(getattr(h, "baseFilename", "")).resolve()
                    == file_path.resolve()
                ):
                    return h
            except Exception:
                continue

    fh = RotatingFileHandler(
        filename=str(file_path),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(level)
    if add_context_filters:
        fh.addFilter(_ContextFilter())
        fh.addFilter(_ContextVarStampingFilter())
    fh.setFormatter(logging.Formatter(fmt))
    owner.addHandler(fh)
    return fh


# -----------------------------------------------------------------------------
# Public: configure root logging for the driver (console + run file)
# -----------------------------------------------------------------------------
def configure_logging(out_dir: Path, level: Union[str, int] = "INFO") -> logging.Logger:
    """
    Configure process-wide logging:
      - Clear root handlers and sanitize sub-loggers.
      - Add one console handler (context-stamped).
      - Add one rotating run-log handler at <out_dir>/sync-run.log.
      - Return a project logger (name 'sync') for convenience.

    Call once at program start, before running the pipeline.
    """
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    sanitize_existing_loggers()

    lvl = _level_to_int(level)
    root.setLevel(lvl)

    # Console
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.addFilter(_ContextFilter())
    console.addFilter(_ContextVarStampingFilter())
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)

    # Aggregated run file
    _attach_rotating_file_handler(
        root,
        Path(out_dir) / "sync-run.log",
        lvl,
        RUN_FILE_FORMAT,
        add_context_filters=True,
    )

    return logging.getLogger("sync")


# -----------------------------------------------------------------------------
# Public: attach per-camera rotating file logger
# -----------------------------------------------------------------------------
def attach_cam_logger(
    seg_id: str,
    cam_serial: str,
    cam_dir: Path,
    level: Optional[Union[str, int]] = None,
    *,
    logger_name: str = "sync",
) -> Tuple[logging.LoggerAdapter, RotatingFileHandler]:
    """
    Attach a per-camera rotating file handler at <cam_dir>/sync.log to the ROOT,
    stamping all records with [seg/cam]. Returns (LoggerAdapter, handler).

    You MUST remove the returned handler when finished:
        adapter, h = attach_cam_logger(...)
        try:
            with log_context(seg, cam):
                adapter.info("...")
        finally:
            logging.getLogger().removeHandler(h)
            h.close()
    """
    root = logging.getLogger()
    lvl = root.level if level is None else _level_to_int(level)

    fh = _attach_rotating_file_handler(
        root, Path(cam_dir) / "sync.log", lvl, CAM_FILE_FORMAT, add_context_filters=True
    )
    fh.addFilter(_SegCamStampFilter(seg_id, cam_serial))

    adapter = logging.LoggerAdapter(
        logging.getLogger(logger_name), extra={"seg": seg_id, "cam": cam_serial}
    )
    return adapter, fh


# -----------------------------------------------------------------------------
# Public: standalone console logging (for modules run directly)
# -----------------------------------------------------------------------------
def configure_standalone_logging(
    level: Union[str, int] = "INFO", seg: str = "-", cam: str = "-"
) -> None:
    """
    Minimal, non-intrusive console logging for `python -m ...`.

    - Installs ONE console handler ONLY if the root has no handlers (i.e., not under
      the driver).
    - Uses a filter to ensure [seg/cam] are present (typically from CLI flags).
    - Does NOT add file handlers or modify sub-loggers.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # respect existing (driver) configuration

    lvl = _level_to_int(level)
    root.setLevel(lvl)

    h = logging.StreamHandler()
    h.setLevel(lvl)
    h.addFilter(_StandaloneSegCamFilter(seg=seg, cam=cam))
    h.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(h)
