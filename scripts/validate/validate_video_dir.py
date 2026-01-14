from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _now_iso() -> str:
    # Local time for user-facing clarity.
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


Check = Dict[str, Any]


def _init_checks() -> List[Check]:
    return [
        {"name": "Segment JSON present", "status": "pending", "message": ""},
        {"name": "Camera MP4 present", "status": "pending", "message": ""},
        {"name": "Segments discovered", "status": "pending", "message": ""},
        {"name": "Companion JSON per segment", "status": "pending", "message": ""},
        {"name": "Cameras discovered", "status": "pending", "message": ""},
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


def _finalize_checks_on_fail(checks: List[Check]) -> None:
    for c in checks:
        if c.get("status") in {"pending", "running"}:
            c["status"] = "skipped"


def _sort_segment_ids(segment_ids: List[str]) -> List[str]:
    """
    Sort segment IDs chronologically when they contain a YYYYMMDD_HHMMSS suffix.
    Any non-matching IDs are placed after, in lexicographic order.
    """

    def key(seg: str) -> tuple:
        m = re.search(r"(\d{8})_(\d{6})", seg)
        if not m:
            return (1, seg)
        return (0, m.group(1), m.group(2), seg)

    return sorted(segment_ids, key=key)


def discover_from_video_dir(video_dir: Path) -> Dict[str, List[str]]:
    """
    Discover segment IDs and camera IDs from a video directory.

    - Segment JSON: `<SEG>.json`
    - Camera MP4: `<SEG>.<CAM>.mp4`
    """
    if not video_dir.exists() or not video_dir.is_dir():
        raise FileNotFoundError(video_dir)

    segments: set[str] = set()
    cameras: set[str] = set()

    mp4_re = re.compile(r"^(?P<seg>.+)\.(?P<cam>[0-9A-Za-z]+)\.mp4$", re.IGNORECASE)
    for p in video_dir.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".json":
            segments.add(p.stem)
        elif suf == ".mp4":
            m = mp4_re.match(p.name)
            if not m:
                continue
            segments.add(m.group("seg"))
            cameras.add(m.group("cam"))

    segs = _sort_segment_ids(list(segments))
    cams = sorted(cameras, key=lambda s: int(s) if s.isdigit() else s)
    return {"segments": segs, "cameras": cams}


def discover_video_pairs(video_dir: Path) -> set[tuple[str, str]]:
    """
    Discover available (segment, camera) pairs from <SEG>.<CAM>.mp4 files.
    """
    if not video_dir.exists() or not video_dir.is_dir():
        raise FileNotFoundError(video_dir)

    pairs: set[tuple[str, str]] = set()
    mp4_re = re.compile(r"^(?P<seg>.+)\.(?P<cam>[0-9A-Za-z]+)\.mp4$", re.IGNORECASE)
    for p in video_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".mp4":
            continue
        m = mp4_re.match(p.name)
        if not m:
            continue
        pairs.add((m.group("seg"), m.group("cam")))
    return pairs


def validate_video_dir(video_dir: str) -> Dict[str, Any]:
    result = validate_video_dir_progress(video_dir, on_progress=None)
    result["running"] = False
    return result


def validate_video_dir_progress(
    video_dir: str,
    *,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Validate a video directory, updating per-check statuses as it runs.
    Intended for UIs that want to show per-check progress.
    """
    video_dir = (video_dir or "").strip()
    p = Path(video_dir).expanduser()
    payload: Dict[str, Any] = {
        "video_dir": str(p),
        "ok": False,
        "running": True,
        "segments_count": 0,
        "cameras_count": 0,
        "segments_first": None,
        "segments_last": None,
        "segments_preview": [],
        "cameras": [],
        "cameras_preview": [],
        "available_pairs": [],
        "missing_json_segments": [],
        "duplicate_json_segments": [],
        "json_count": 0,
        "mp4_count": 0,
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

    if not video_dir:
        payload["error"] = "Video dir is required."
        _set_check(payload["checks"], "Segment JSON present", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    if not p.exists():
        payload["error"] = f"Video dir does not exist: {p}"
        _set_check(payload["checks"], "Segment JSON present", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    if not p.is_dir():
        payload["error"] = f"Video dir is not a directory: {p}"
        _set_check(payload["checks"], "Segment JSON present", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload

    # Pre-scan top-level for quick counts (case-insensitive) and stems.
    mp4_count = 0
    json_count = 0
    json_stem_counts: Dict[str, int] = {}
    mp4_re = re.compile(r"^(?P<seg>.+)\.(?P<cam>[0-9A-Za-z]+)\.mp4$", re.IGNORECASE)
    mp4_segments: set[str] = set()
    mp4_pairs: set[tuple[str, str]] = set()
    for c in p.iterdir():
        if not c.is_file():
            continue
        suf = c.suffix.lower()
        if suf == ".mp4":
            mp4_count += 1
            m = mp4_re.match(c.name)
            if m:
                seg = m.group("seg")
                cam = m.group("cam")
                mp4_segments.add(seg)
                mp4_pairs.add((seg, cam))
        elif suf == ".json":
            json_count += 1
            json_stem_counts[c.stem] = json_stem_counts.get(c.stem, 0) + 1
    payload["mp4_count"] = mp4_count
    payload["json_count"] = json_count
    payload["available_pairs"] = [
        {"segment": s, "camera": c} for s, c in sorted(mp4_pairs)
    ]
    emit()

    # Helpful hint: users often select a parent folder; surface MP4/JSON in subfolders.
    if mp4_count == 0 and json_count == 0:
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
                if Path(fn).suffix.lower() not in {".mp4", ".json"}:
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
        emit()

    # 1) Segment JSON present
    _set_check(payload["checks"], "Segment JSON present", "running")
    emit()
    if json_count <= 0:
        payload["error"] = "No segment JSON files found in this folder."
        _set_check(payload["checks"], "Segment JSON present", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    _set_check(payload["checks"], "Segment JSON present", "pass")
    emit()

    # 2) Camera MP4 present
    _set_check(payload["checks"], "Camera MP4 present", "running")
    emit()
    if mp4_count <= 0:
        payload["error"] = "No camera MP4 files found in this folder."
        _set_check(payload["checks"], "Camera MP4 present", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    _set_check(payload["checks"], "Camera MP4 present", "pass")
    emit()

    # 3) Segments discovered
    _set_check(payload["checks"], "Segments discovered", "running")
    emit()
    try:
        idx = discover_from_video_dir(p)
        segs = idx["segments"]
        cams = idx["cameras"]
    except Exception as e:
        payload["error"] = str(e)
        _set_check(payload["checks"], "Segments discovered", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload

    payload["segments_count"] = len(segs)
    payload["cameras_count"] = len(cams)
    payload["segments_first"] = segs[0] if segs else None
    payload["segments_last"] = segs[-1] if segs else None
    payload["segments_preview"] = segs[:25]
    payload["cameras"] = cams
    payload["cameras_preview"] = cams[:25]
    emit()

    if payload["segments_count"] <= 0:
        payload["error"] = "No segments discovered from JSON/MP4 filenames."
        _set_check(payload["checks"], "Segments discovered", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload
    _set_check(payload["checks"], "Segments discovered", "pass")
    emit()

    # 4) Companion JSON per segment
    _set_check(payload["checks"], "Companion JSON per segment", "running")
    emit()
    if not mp4_segments:
        payload["error"] = (
            "MP4 files were found, but none matched the expected naming pattern "
            "`<SEG>.<CAM>.mp4` (example: `TRBD002_20250709_102609.23512909.mp4`)."
        )
        _set_check(
            payload["checks"], "Companion JSON per segment", "fail", payload["error"]
        )
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload

    missing_json: List[str] = []
    dup_json: List[str] = []
    for seg in sorted(mp4_segments):
        cnt = json_stem_counts.get(seg, 0)
        if cnt == 0:
            missing_json.append(seg)
        elif cnt > 1:
            dup_json.append(seg)

    payload["missing_json_segments"] = missing_json
    payload["duplicate_json_segments"] = dup_json
    if missing_json or dup_json:
        parts: List[str] = []
        if missing_json:
            preview = ", ".join(missing_json[:10])
            extra = len(missing_json) - 10
            if extra > 0:
                preview += f" … (+{extra} more)"
            parts.append(f"Missing JSON for segment(s): {preview}.")
        if dup_json:
            preview = ", ".join(dup_json[:10])
            extra = len(dup_json) - 10
            if extra > 0:
                preview += f" … (+{extra} more)"
            parts.append(f"Duplicate JSON for segment(s): {preview}.")
        warning = " ".join(parts).strip()
        payload["warning"] = warning
        _set_check(payload["checks"], "Companion JSON per segment", "warn", warning)
        emit()
    else:
        _set_check(payload["checks"], "Companion JSON per segment", "pass")
        emit()

    # 5) Cameras discovered
    _set_check(payload["checks"], "Cameras discovered", "running")
    emit()
    if payload["cameras_count"] <= 0:
        payload["error"] = "No cameras discovered from MP4 filenames."
        _set_check(payload["checks"], "Cameras discovered", "fail", payload["error"])
        _finalize_checks_on_fail(payload["checks"])
        payload["running"] = False
        emit()
        return payload

    _set_check(payload["checks"], "Cameras discovered", "pass")
    payload["ok"] = True
    payload["running"] = False
    emit()
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a video directory for video-sync-nbu."
    )
    parser.add_argument(
        "video_dir", help="Path to folder containing segment JSON and camera MP4 files."
    )
    parser.add_argument("--json", action="store_true", help="Output JSON (default).")
    args = parser.parse_args(argv)

    result = validate_video_dir(args.video_dir)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
