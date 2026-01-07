from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _now_iso() -> str:
    # Local time for user-facing clarity.
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


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
    - <out>/<segment>/<camera>/synced_video/*.mp4 exists, OR
    - <out>/<segment>/<camera>/sync.log exists
    """
    segs = [str(s).strip() for s in (segments or []) if str(s).strip()]
    seg_set = set(segs)
    synced: Set[str] = set()

    # Only consider segment directories that actually exist in out_dir.
    existing_seg_dirs: List[Path] = []
    for d in out_dir.iterdir():
        if not d.is_dir():
            continue
        if seg_set and d.name not in seg_set:
            continue
        existing_seg_dirs.append(d)

    for seg_dir in existing_seg_dirs:
        seg = seg_dir.name
        # Any camera subdir with evidence of syncing?
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


def validate_out_dir(
    out_dir: str, *, segments: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate an output directory:
    - whether it exists / is empty
    - whether it contains reusable audio decode artifacts
    - how many segments appear to have already been synced
    """
    out_dir = (out_dir or "").strip()
    p = Path(out_dir).expanduser()

    payload: Dict[str, Any] = {
        "out_dir": str(p),
        "ok": False,
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
        "checks": [],
        "error": None,
        "checked_at": _now_iso(),
    }

    if not out_dir:
        payload["error"] = "Output dir is required."
        return payload

    if p.exists() and not p.is_dir():
        payload["error"] = f"Output dir is not a directory: {p}"
        return payload

    if not p.exists():
        payload["exists"] = False
        payload["will_create"] = True
        payload["empty"] = True
        payload["checks"].append(
            {"name": "Directory", "status": "pass", "message": "Will be created"}
        )
        payload["ok"] = True
        return payload

    payload["exists"] = True
    entries = _list_nontrivial_entries(p)
    payload["entries_count"] = len(entries)
    payload["empty"] = len(entries) == 0

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

    sync_stats = _count_synced_segments(p, segments)
    payload.update(sync_stats)

    payload["checks"].append(
        {
            "name": "Directory exists",
            "status": "pass",
            "message": (
                "Empty" if payload["empty"] else f"{payload['entries_count']} item(s)"
            ),
        }
    )
    payload["checks"].append(
        {
            "name": "Reusable audio decode",
            "status": "pass" if payload["can_reuse_audio_decoded"] else "warn",
            "message": (
                "Found raw.csv + raw-gapfilled-filtered.csv"
                if payload["can_reuse_audio_decoded"]
                else "Not found"
            ),
        }
    )
    payload["checks"].append(
        {
            "name": "Previously synced segments",
            "status": "pass",
            "message": f"{payload['synced_segments_count']} found",
        }
    )

    # Output dir is always usable (pipeline creates subfolders), but warn if not empty.
    payload["ok"] = True
    if not payload["empty"]:
        payload["error"] = None
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
