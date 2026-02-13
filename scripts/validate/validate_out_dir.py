from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Set


def _now_iso() -> str:
    # Local time for user-facing clarity.
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


Check = Dict[str, Any]


def _init_checks() -> List[Check]:
    return [
        {"name": "Directory empty", "status": "pending", "message": ""},
        {"name": "Audio metadata present", "status": "pending", "message": ""},
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


def _list_nontrivial_entries(p: Path) -> List[Path]:
    """
    List directory entries ignoring common OS metadata files.
    """
    entries: List[Path] = []
    for c in p.iterdir():
        if c.name in {".DS_Store"}:
            continue
        entries.append(c)
    return entries


def _count_synced_segments(
    out_dir: Path, segments: Optional[List[str]]
) -> Dict[str, Any]:
    """
    Count segments that appear to have completed syncing in this output dir.

    Heuristics:
    - <out>/runs/runNNNN/<segment>/<camera>/synced_video/*.mp4 exists, OR
    - <out>/runs/runNNNN/<segment>/<camera>/sync.log exists
    """
    segs = [str(s).strip() for s in (segments or []) if str(s).strip()]
    seg_set = set(segs)
    synced: Set[str] = set()

    runs_dir = out_dir / "runs"
    if not runs_dir.is_dir():
        return {
            "synced_segments_count": 0,
            "synced_segments_preview": [],
            "total_segments": len(segs) if segs else None,
        }

    run_dirs = sorted(
        [
            d
            for d in runs_dir.iterdir()
            if d.is_dir() and re.fullmatch(r"run\d+", d.name)
        ],
        key=lambda p: p.name,
    )
    for run_dir in run_dirs:
        for seg_dir in run_dir.iterdir():
            if not seg_dir.is_dir():
                continue
            seg = seg_dir.name
            if seg_set and seg not in seg_set:
                continue
            try:
                for cam_dir in seg_dir.iterdir():
                    if not cam_dir.is_dir():
                        continue
                    if (cam_dir / "sync.log").exists():
                        synced.add(seg)
                        break
                    sv = cam_dir / "synced_video"
                    if sv.is_dir():
                        mp4s = list(sv.glob("*.mp4"))
                        if mp4s:
                            synced.add(seg)
                            break
            except Exception:
                continue

    return {
        "synced_segments_count": len(synced),
        "synced_segments_preview": sorted(list(synced))[:25],
        "total_segments": len(segs) if segs else None,
    }


def discover_synced_pairs(
    out_dir: str,
    *,
    segments: Optional[List[str]] = None,
    cameras: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Discover which segment/camera combinations appear already synced in an output folder.

    Evidence:
    - <out>/runs/runNNNN/<segment>/<camera>/sync.log exists, OR
    - <out>/runs/runNNNN/<segment>/<camera>/synced_video/*.mp4 exists
    """
    out_dir = (out_dir or "").strip()
    p = Path(out_dir).expanduser()
    if not out_dir or not p.exists() or not p.is_dir():
        return {"synced_pairs": [], "synced_pairs_count": 0}

    seg_set = {str(s).strip() for s in (segments or []) if str(s).strip()} or None
    cam_set = {str(c).strip() for c in (cameras or []) if str(c).strip()} or None

    runs_dir = p / "runs"
    if not runs_dir.is_dir():
        return {"synced_pairs": [], "synced_pairs_count": 0}

    run_dirs = sorted(
        [
            d
            for d in runs_dir.iterdir()
            if d.is_dir() and re.fullmatch(r"run\d+", d.name)
        ],
        key=lambda q: q.name,
    )
    pairs: Set[tuple[str, str]] = set()
    for run_dir in run_dirs:
        for seg_dir in run_dir.iterdir():
            if not seg_dir.is_dir():
                continue
            seg = seg_dir.name
            if seg_set is not None and seg not in seg_set:
                continue
            try:
                for cam_dir in seg_dir.iterdir():
                    if not cam_dir.is_dir():
                        continue
                    cam = cam_dir.name
                    if cam_set is not None and cam not in cam_set:
                        continue
                    if (cam_dir / "sync.log").exists():
                        pairs.add((seg, cam))
                        continue
                    sv = cam_dir / "synced_video"
                    if sv.is_dir():
                        try:
                            if any(sv.glob("*.mp4")):
                                pairs.add((seg, cam))
                        except Exception:
                            pass
            except Exception:
                continue

    synced_pairs = [{"segment": s, "camera": c} for s, c in sorted(pairs)]
    return {"synced_pairs": synced_pairs, "synced_pairs_count": len(synced_pairs)}


def validate_out_dir(
    out_dir: str, *, segments: Optional[List[str]] = None
) -> Dict[str, Any]:
    result = validate_out_dir_progress(out_dir, segments=segments, on_progress=None)
    result["running"] = False
    return result


def validate_out_dir_progress(
    out_dir: str,
    *,
    segments: Optional[List[str]] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Validate an output directory, updating per-check statuses as it runs.
    Intended for UIs that want to show per-check progress.
    """
    out_dir = (out_dir or "").strip()
    p = Path(out_dir).expanduser()

    payload: Dict[str, Any] = {
        "out_dir": str(p),
        "ok": False,
        "running": True,
        "exists": False,
        "will_create": False,
        "empty": None,
        "entries_count": 0,
        "audio_decoded": {
            "exists": False,
            "raw_csv": False,
            "filtered_csv": False,
        },
        "can_reuse_audio_decoded": False,
        "split_decoded_count": 0,
        "serial_chunks_count": 0,
        "synced_segments_count": 0,
        "synced_segments_preview": [],
        "total_segments": None,
        "synced_pairs": [],
        "synced_pairs_count": 0,
        "audio_metadata": {
            "exists": False,
            "abs_json": False,
        },
        "checks": _init_checks(),
        "error": None,
        "checked_at": _now_iso(),
    }

    def emit() -> None:
        if on_progress is not None:
            on_progress(payload)

    emit()

    if not out_dir:
        payload["error"] = "Output dir is required."
        _set_check(payload["checks"], "Directory empty", "fail", payload["error"])
        _set_check(
            payload["checks"],
            "Audio metadata present",
            "skipped",
            "Output dir not set",
        )
        payload["running"] = False
        emit()
        return payload

    if p.exists() and not p.is_dir():
        payload["error"] = f"Output dir is not a directory: {p}"
        _set_check(payload["checks"], "Directory empty", "fail", payload["error"])
        _set_check(
            payload["checks"],
            "Audio metadata present",
            "skipped",
            "Output dir is not a directory",
        )
        payload["running"] = False
        emit()
        return payload

    if not p.exists():
        payload["exists"] = False
        payload["will_create"] = True
        payload["empty"] = True
        _set_check(
            payload["checks"], "Directory empty", "pass", "Will be created (empty)"
        )
        _set_check(
            payload["checks"],
            "Audio metadata present",
            "skipped",
            "Output dir not created yet",
        )
        payload["ok"] = True
        payload["running"] = False
        emit()
        return payload

    payload["exists"] = True
    payload["will_create"] = False

    _set_check(payload["checks"], "Directory empty", "running")
    _set_check(payload["checks"], "Audio metadata present", "running")
    emit()
    entries = _list_nontrivial_entries(p)
    payload["entries_count"] = len(entries)
    payload["empty"] = len(entries) == 0
    if payload["empty"]:
        _set_check(payload["checks"], "Directory empty", "pass", "Empty")
    else:
        _set_check(
            payload["checks"],
            "Directory empty",
            "warn",
            f"{payload['entries_count']} item(s)",
        )
    emit()

    # Audio decoded artifacts (enables --skip-decode)
    ad = p / "audio_decoded"
    raw_csv = ad / "raw.csv"
    filtered_csv = ad / "raw-gapfilled-filtered.csv"
    payload["audio_decoded"]["exists"] = ad.is_dir()
    payload["audio_decoded"]["raw_csv"] = raw_csv.is_file()
    payload["audio_decoded"]["filtered_csv"] = filtered_csv.is_file()
    payload["can_reuse_audio_decoded"] = bool(
        raw_csv.is_file() and filtered_csv.is_file()
    )

    # Split artifacts (informational)
    split_decoded = p / "split_decoded"
    if split_decoded.is_dir():
        try:
            payload["split_decoded_count"] = len(list(split_decoded.glob("*.csv")))
        except Exception:
            payload["split_decoded_count"] = 0
    serial_chunks = p / "serial_audio_splitted"
    if serial_chunks.is_dir():
        try:
            payload["serial_chunks_count"] = len(list(serial_chunks.glob("*.wav")))
        except Exception:
            payload["serial_chunks_count"] = 0

    metadata_dir = p / "audio_metadata"
    abs_json = metadata_dir / "audio_abs_start.json"
    payload["audio_metadata"]["exists"] = metadata_dir.is_dir()
    payload["audio_metadata"]["abs_json"] = abs_json.is_file()
    if payload["audio_metadata"]["abs_json"]:
        _set_check(
            payload["checks"],
            "Audio metadata present",
            "pass",
            "audio_abs_start.json found",
        )
    elif payload["audio_metadata"]["exists"]:
        _set_check(
            payload["checks"],
            "Audio metadata present",
            "warn",
            "audio_metadata/ present (no audio_abs_start.json)",
        )
    else:
        _set_check(
            payload["checks"],
            "Audio metadata present",
            "warn",
            "audio_metadata/ not found",
        )

    sync_stats = _count_synced_segments(p, segments)
    payload.update(sync_stats)
    synced_pairs = discover_synced_pairs(str(p), segments=segments)
    payload["synced_pairs"] = synced_pairs.get("synced_pairs", [])
    payload["synced_pairs_count"] = synced_pairs.get(
        "synced_pairs_count", len(payload["synced_pairs"])
    )

    payload["ok"] = True
    payload["running"] = False
    emit()
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an output directory for video-sync-nbu."
    )
    parser.add_argument("out_dir", help="Path to output folder.")
    parser.add_argument(
        "--segments",
        type=str,
        default="",
        help="Optional comma-separated segment IDs (used to estimate how many are already synced).",
    )
    args = parser.parse_args(argv)

    segments = [s.strip() for s in (args.segments or "").split(",") if s.strip()]
    result = validate_out_dir(args.out_dir, segments=segments or None)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
