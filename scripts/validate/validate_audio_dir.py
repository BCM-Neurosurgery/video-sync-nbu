from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from scripts.index.audiodiscover import AudioDiscoverer
from scripts.index.filepatterns import FilePatterns


def _now_iso() -> str:
    # Local time for user-facing clarity.
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_duration(seconds: float) -> str:
    if seconds is None or seconds <= 0:
        return ""
    total = int(round(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _sniff_wav_duration(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    if rate <= 0:
        return 0.0
    return float(frames) / float(rate)


def _sniff_mp3_duration(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        return 0.0
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return 0.0
    try:
        return float(out)
    except Exception:
        return 0.0


Check = Dict[str, Any]


def _init_checks() -> List[Check]:
    return [
        {"name": "Find audio files", "status": "pending", "message": ""},
        {"name": "Naming pattern", "status": "pending", "message": ""},
        {"name": "Serial channel present", "status": "pending", "message": ""},
        {"name": "Program channel present", "status": "pending", "message": ""},
        {"name": "Sample rate detected", "status": "pending", "message": ""},
        {"name": "Duration detected", "status": "pending", "message": ""},
    ]


def _set_check(
    checks: List[Check],
    name: str,
    status: str,
    message: str = "",
) -> None:
    for c in checks:
        if c.get("name") == name:
            c["status"] = status
            c["message"] = message
            return


def _choose_best_per_channel(parsed: List[Tuple[int, str, Path]]) -> Dict[int, Path]:
    """
    Select one file per detected channel, preferring WAV over MP3.
    """
    chosen: Dict[int, Path] = {}
    for ch, ext, p in parsed:
        existing = chosen.get(ch)
        if existing is None:
            chosen[ch] = p
        else:
            if existing.suffix.lower() == ".mp3" and ext == "wav":
                chosen[ch] = p
    return chosen


def _finalize_checks_on_fail(checks: List[Check]) -> None:
    for c in checks:
        if c.get("status") in {"pending", "running"}:
            c["status"] = "skipped"


def validate_audio_dir_progress(
    audio_dir: str,
    *,
    logger: Optional[logging.Logger] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Validate an audio directory, updating per-check statuses as it runs.
    Intended for UIs that want to show per-check progress.
    """
    log = logger or logging.getLogger("validate.audio")
    audio_dir = (audio_dir or "").strip()
    p = Path(audio_dir).expanduser()

    payload: Dict[str, Any] = {
        "audio_dir": str(p),
        "ok": False,
        "running": True,
        "files": [],
        "nested_files": [],
        "nested_count": 0,
        "checks": _init_checks(),
        "error": None,
        "checked_at": _now_iso(),
    }

    def emit() -> None:
        if on_progress is not None:
            on_progress(payload)

    emit()

    if not audio_dir:
        payload["error"] = "Audio dir is required."
        _set_check(payload["checks"], "Find audio files", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    if not p.exists():
        payload["error"] = f"Audio dir does not exist: {p}"
        _set_check(payload["checks"], "Find audio files", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    if not p.is_dir():
        payload["error"] = f"Audio dir is not a directory: {p}"
        _set_check(payload["checks"], "Find audio files", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload

    # Pre-scan: show matching candidates even if strict discovery fails.
    _set_check(payload["checks"], "Find audio files", "running")
    emit()
    candidates: List[Path] = []
    for c in p.iterdir():
        if not c.is_file():
            continue
        if c.suffix.lower() in {".wav", ".mp3"}:
            candidates.append(c)
    candidates = sorted(candidates)
    payload["files"] = [{"name": c.name, "path": str(c)} for c in candidates]

    # Helpful hint: users often select a parent folder; surface audio files in subfolders.
    if not candidates:
        nested: List[Dict[str, str]] = []
        nested_count = 0
        max_examples = 20
        max_depth = 2
        for root, dirs, files in os.walk(p):
            root_path = Path(root)
            try:
                depth = len(root_path.relative_to(p).parts)
            except Exception:
                depth = 0
            if depth >= max_depth:
                dirs[:] = []
            for fn in files:
                if Path(fn).suffix.lower() not in {".wav", ".mp3"}:
                    continue
                nested_count += 1
                if len(nested) < max_examples:
                    full = root_path / fn
                    try:
                        rel = str(full.relative_to(p))
                    except Exception:
                        rel = fn
                    nested.append(
                        {"name": full.name, "path": str(full), "relpath": rel}
                    )
        payload["nested_count"] = nested_count
        payload["nested_files"] = nested

    if not candidates:
        payload["error"] = "No .wav/.mp3 files found in this folder."
        _set_check(payload["checks"], "Find audio files", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload

    _set_check(
        payload["checks"], "Find audio files", "pass", f"{len(candidates)} file(s)"
    )
    emit()

    ad = AudioDiscoverer(audio_dir=p, log=log)

    # Parse & naming pattern
    _set_check(payload["checks"], "Naming pattern", "running")
    emit()
    parsed: List[Tuple[int, str, Path]] = []
    invalid: List[str] = []
    for c in candidates:
        r = FilePatterns.parse_audio_filename(c)
        if not r:
            invalid.append(c.name)
            continue
        ch, ext = r
        parsed.append((int(ch), str(ext), c))
    if invalid:
        payload["error"] = "Audio files with unexpected name pattern: " + ", ".join(
            sorted(invalid)
        )
        _set_check(payload["checks"], "Naming pattern", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    _set_check(payload["checks"], "Naming pattern", "pass")
    emit()

    # Serial channel present (exactly one -03)
    _set_check(payload["checks"], "Serial channel present", "running")
    emit()
    ch_counts: Dict[int, int] = {}
    for ch, ext, c in parsed:
        ch_counts[ch] = ch_counts.get(ch, 0) + 1
    serial_count = ch_counts.get(ad.default_serial_channel, 0)
    if serial_count != 1:
        payload["error"] = (
            f"Expected exactly one channel {ad.default_serial_channel:02d} file "
            f"(e.g., *-{ad.default_serial_channel:02d}.wav/mp3); found {serial_count}."
        )
        _set_check(
            payload["checks"], "Serial channel present", "fail", payload["error"]
        )
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    _set_check(payload["checks"], "Serial channel present", "pass")
    emit()

    # Program channel present (-01 or -02)
    _set_check(payload["checks"], "Program channel present", "running")
    emit()
    has_01 = 1 in ch_counts
    has_02 = 2 in ch_counts
    if not (has_01 or has_02):
        payload["error"] = (
            "Expected at least one program channel: found neither -01 nor -02 in AUDIO_DIR."
        )
        _set_check(
            payload["checks"], "Program channel present", "fail", payload["error"]
        )
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    if has_01 ^ has_02:
        missing = "01" if not has_01 else "02"
        _set_check(
            payload["checks"], "Program channel present", "warn", f"Missing -{missing}"
        )
    else:
        _set_check(payload["checks"], "Program channel present", "pass")
    emit()

    # Sample rate detection (build files summary)
    _set_check(payload["checks"], "Sample rate detected", "running")
    emit()

    chosen = _choose_best_per_channel(parsed)
    found_files: List[Dict[str, Any]] = []
    for ch in sorted(chosen.keys()):
        path = chosen[ch]
        ext = path.suffix.lower().lstrip(".")
        sr = 0
        if ext == "wav":
            sr = ad._sniff_wav_samplerate(path)  # type: ignore[attr-defined]
        elif ext == "mp3":
            if shutil.which("ffprobe") is None:
                sr = 0
            else:
                sr = ad._sniff_mp3_samplerate(path)  # type: ignore[attr-defined]

        if sr <= 0:
            payload["error"] = (
                f"Could not determine sample rate for {path.name} (ext={ext}). "
                "Fix the audio file or ensure required tools (e.g. ffprobe) are available."
            )
            _set_check(
                payload["checks"], "Sample rate detected", "fail", payload["error"]
            )
            _finalize_checks_on_fail(payload["checks"])
            payload["running"] = False
            emit()
            return payload

        found_files.append(
            {
                "channel": int(ch),
                "name": path.name,
                "path": str(path),
                "extension": ext,
                "sample_rate": int(sr),
                "duration_sec": None,
                "duration": "",
                "is_serial": bool(int(ch) == ad.default_serial_channel),
            }
        )

    payload["files"] = found_files
    _set_check(payload["checks"], "Sample rate detected", "pass")
    emit()

    # Duration detection (fast header-based for WAV; ffprobe for MP3)
    _set_check(payload["checks"], "Duration detected", "running")
    emit()
    for f in payload["files"]:
        ext = str(f.get("extension") or "").lower()
        path = Path(str(f.get("path") or ""))
        dur = 0.0
        try:
            if ext == "wav":
                dur = _sniff_wav_duration(path)
            elif ext == "mp3":
                dur = _sniff_mp3_duration(path)
            else:
                dur = 0.0
        except Exception:
            dur = 0.0

        if dur <= 0:
            payload["error"] = (
                f"Could not determine duration for {path.name} (ext={ext}). "
                "Fix the audio file or ensure required tools (e.g. ffprobe) are available."
            )
            _set_check(payload["checks"], "Duration detected", "fail", payload["error"])
            _finalize_checks_on_fail(payload["checks"])
            payload["running"] = False
            emit()
            return payload

        f["duration_sec"] = float(dur)
        f["duration"] = _format_duration(dur)

    _set_check(payload["checks"], "Duration detected", "pass")
    payload["ok"] = True
    payload["running"] = False
    emit()
    return payload


def validate_audio_dir(
    audio_dir: str, *, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    result = validate_audio_dir_progress(audio_dir, logger=logger, on_progress=None)
    result["running"] = False
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an audio directory for video-sync-nbu."
    )
    parser.add_argument(
        "audio_dir", help="Path to folder containing channelized audio files."
    )
    parser.add_argument("--json", action="store_true", help="Output JSON (default).")
    args = parser.parse_args(argv)

    result = validate_audio_dir(args.audio_dir)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
