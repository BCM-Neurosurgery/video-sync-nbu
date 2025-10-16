from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from prefect import flow, get_run_logger

from scripts.cli import cli_emu_time


@dataclass
class TimeSyncRunConfig:
    """Configuration for a single cli_emu_time execution."""

    patient_dir: Path
    video_dir: Path
    out_dir: Path
    name: Optional[str] = None
    keywords: Optional[Sequence[str]] = None
    cam_serials: Optional[Sequence[str]] = None
    room_mic: str = "roommic1"
    log_level: str = "INFO"
    overwrite: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TimeSyncRunConfig":
        """Build a config object from a raw dictionary."""

        def _ensure_list(value: object) -> Optional[List[str]]:
            if value is None:
                return None
            if isinstance(value, str):
                stripped = value.strip()
                return [item for item in stripped.split() if item]
            if isinstance(value, Iterable):
                result: List[str] = []
                for item in value:
                    if item is None:
                        continue
                    result.append(str(item))
                return result or None
            return [str(value)]

        def _as_bool(value: object) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        missing = [
            key
            for key in ("patient_dir", "video_dir", "out_dir")
            if key not in payload or payload[key] in (None, "")
        ]
        if missing:
            raise ValueError(
                f"Run configuration missing required field(s): {', '.join(missing)}. "
                "Double-check the JSON parameters in Prefect."
            )

        name_raw = payload.get("name")
        name = None
        if name_raw is not None:
            name_str = str(name_raw).strip()
            if name_str:
                name = name_str

        return cls(
            patient_dir=Path(str(payload["patient_dir"])).expanduser(),
            video_dir=Path(str(payload["video_dir"])).expanduser(),
            out_dir=Path(str(payload["out_dir"])).expanduser(),
            name=name,
            keywords=_ensure_list(payload.get("keywords")),
            cam_serials=_ensure_list(
                payload.get("cam_serials") or payload.get("cam_serial")
            ),
            room_mic=str(payload.get("room_mic", "roommic1")),
            log_level=str(payload.get("log_level", "INFO")),
            overwrite=_as_bool(payload.get("overwrite", False)),
        )

    def as_cli_args(self) -> List[str]:
        """Convert the configuration to cli_emu_time command-line arguments."""
        args: List[str] = [
            "--patient-dir",
            str(self.patient_dir),
            "--video-dir",
            str(self.video_dir),
            "--out-dir",
            str(self.out_dir),
            "--room-mic",
            self.room_mic,
            "--log-level",
            self.log_level,
        ]
        if self.keywords:
            args.append("--keywords")
            args.extend(self.keywords)
        if self.cam_serials:
            for serial in self.cam_serials:
                args.extend(["--cam-serial", serial])
        if self.overwrite:
            args.append("--overwrite")
        return args

    @property
    def display_name(self) -> str:
        base = self.name or self.patient_dir.name or str(self.patient_dir)
        return base

    @property
    def label(self) -> str:
        """Human-friendly label to identify the run."""
        details = f"{self.patient_dir.name}:{self.video_dir.name}"
        base = self.display_name
        return details if base == details else f"{base} [{details}]"


class _PrefectLogHandler(logging.Handler):
    """Bridge standard logging records to the Prefect run logger."""

    def __init__(self, prefect_logger: logging.Logger) -> None:
        super().__init__()
        self.prefect_logger = prefect_logger
        self.formatter = logging.Formatter("%(message)s")
        self.addFilter(self._allow_record)

    @staticmethod
    def _allow_record(record: logging.LogRecord) -> bool:
        if getattr(record, "_prefect_forwarded", False):
            return False
        if record.name.startswith(("prefect.", "httpx.", "httpcore.", "anyio.")):
            return False
        if "HTTP Request" in record.getMessage():
            return False
        return True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.prefect_logger.log(
                record.levelno,
                message,
                extra={"_prefect_forwarded": True},
            )
        except Exception:  # pragma: no cover - logging safety
            self.handleError(record)


def _extract_single_config(
    data: TimeSyncRunConfig | Mapping[str, object] | Sequence[object] | object,
) -> TimeSyncRunConfig:
    if isinstance(data, TimeSyncRunConfig):
        return data

    if isinstance(data, Mapping):
        if "runs" in data:
            return _extract_single_config(data["runs"])
        keys = set(data.keys())
        required = {"patient_dir", "video_dir", "out_dir"}
        if required <= keys:
            return TimeSyncRunConfig.from_mapping(data)
        if len(data) == 1:
            return _extract_single_config(next(iter(data.values())))
        raise ValueError(
            "Unable to determine which patient to run. Provide a single patient "
            "configuration (mapping with patient_dir/video_dir/out_dir)."
        )

    if isinstance(data, (list, tuple, SequenceABC)) and not isinstance(
        data, (str, bytes)
    ):
        seq = list(data)
        if not seq:
            raise ValueError("No patient configuration provided.")
        if len(seq) > 1:
            raise ValueError(
                "Multiple patient configurations provided. Run one patient per flow run."
            )
        return _extract_single_config(seq[0])

    raise TypeError(
        f"Unsupported configuration type {type(data)!r}; "
        "provide a mapping or TimeSyncRunConfig."
    )


@flow(name="EMU-Time-Sync", flow_run_name="{run_label}")
def time_sync_flow(
    runs: TimeSyncRunConfig | Mapping[str, object] | Sequence[object],
    run_label: Optional[str] = None,
) -> str:
    """Trigger cli_emu_time for a single patient configuration."""
    config = _extract_single_config(runs)
    if not run_label:
        run_label = config.name
    logger = get_run_logger()
    logger.info("Starting cli_emu_time for %s", config.label)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    handler = _PrefectLogHandler(logger)
    cli_emu_time.register_extra_log_handler(handler)
    try:
        exit_code = cli_emu_time.main(config.as_cli_args())
    finally:
        cli_emu_time.unregister_extra_log_handler(handler)
        root_logger = logging.getLogger()
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        handler.close()

    if exit_code != 0:
        raise RuntimeError(
            f"cli_emu_time failed for {config.label} with exit code {exit_code}"
        )

    completed_path = config.out_dir.resolve()
    logger.info("Completed cli_emu_time for %s -> %s", config.label, completed_path)
    return str(completed_path)
