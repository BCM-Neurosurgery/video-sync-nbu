from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    # Local time for user-facing clarity.
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def validate_video_dir(video_dir: str) -> Dict[str, Any]:
    """
    Validate a video directory for expected segment JSON and camera MP4 files.

    Returns a JSON-friendly payload suitable for UIs and CLIs.
    """
    video_dir = (video_dir or "").strip()
    p = Path(video_dir).expanduser()
    payload: Dict[str, Any] = {
        "video_dir": str(p),
        "ok": False,
        "segments_count": 0,
        "cameras_count": 0,
        "segments_preview": [],
        "cameras_preview": [],
        "json_count": 0,
        "mp4_count": 0,
        "nested_files": [],
        "nested_count": 0,
        "checks": [],
        "error": None,
        "checked_at": _now_iso(),
    }
    if not video_dir:
        payload["error"] = "Video dir is required."
        return payload
    if not p.exists():
        payload["error"] = f"Video dir does not exist: {p}"
        return payload
    if not p.is_dir():
        payload["error"] = f"Video dir is not a directory: {p}"
        return payload

    # Pre-scan top-level for quick counts (case-insensitive).
    mp4_count = 0
    json_count = 0
    for c in p.iterdir():
        if not c.is_file():
            continue
        suf = c.suffix.lower()
        if suf == ".mp4":
            mp4_count += 1
        elif suf == ".json":
            json_count += 1
    payload["mp4_count"] = mp4_count
    payload["json_count"] = json_count

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

    try:
        idx = discover_from_video_dir(p)
        segs = idx["segments"]
        cams = idx["cameras"]
        payload["segments_count"] = len(segs)
        payload["cameras_count"] = len(cams)
        payload["segments_preview"] = segs[:25]
        payload["cameras_preview"] = cams[:25]

        payload["checks"].append(
            {
                "name": "Segment JSON present",
                "status": "pass" if json_count > 0 else "fail",
                "message": f"{json_count} found",
            }
        )
        payload["checks"].append(
            {
                "name": "Camera MP4 present",
                "status": "pass" if mp4_count > 0 else "fail",
                "message": f"{mp4_count} found",
            }
        )
        payload["checks"].append(
            {
                "name": "Segments discovered",
                "status": "pass" if payload["segments_count"] > 0 else "fail",
                "message": f"{payload['segments_count']} found",
            }
        )
        payload["checks"].append(
            {
                "name": "Cameras discovered",
                "status": "pass" if payload["cameras_count"] > 0 else "fail",
                "message": f"{payload['cameras_count']} found",
            }
        )

        payload["ok"] = bool(
            json_count > 0
            and mp4_count > 0
            and payload["segments_count"] > 0
            and payload["cameras_count"] > 0
        )
        if not payload["ok"]:
            if mp4_count > 0 and payload["segments_count"] == 0:
                payload["error"] = (
                    "MP4 files were found, but none matched the expected naming pattern "
                    "`<SEG>.<CAM>.mp4` (example: `TRBD002_20250709_102609.23512909.mp4`)."
                )
            elif json_count > 0 and payload["segments_count"] == 0 and mp4_count == 0:
                payload["error"] = (
                    "JSON files were found, but no matching MP4 files were found at the same folder level."
                )
            else:
                payload["error"] = (
                    "Video folder does not look valid for this pipeline "
                    "(missing JSON/MP4 or unable to discover segments/cameras)."
                )
        return payload
    except Exception as e:
        payload["checks"].append(
            {"name": "Discovery", "status": "fail", "message": str(e)}
        )
        payload["error"] = str(e)
        payload["ok"] = False
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
