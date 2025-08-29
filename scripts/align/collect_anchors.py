#!/usr/bin/env python3
"""
scripts/align/collect_anchors.py — Per-segment, per-camera anchor extractor

Inputs
------
1) Serial index CSV from serial-audio decode (columns: serial,start_sample,end_sample)
2) Video directory containing <segment_id>.json and <segment_id>.<CAM>.mp4
3) Segment ID and camera serial

Output
------
JSON list of anchors:
[
  {
    "serial": <int>,
    "audio_sample": <int>,
    "cam_serial": "<str>",
    "segment_id": "<str>",
    "frame_id": <int>,
    "frame_id_reidx": <int>
  }, ...
]

CLI
---
python -m scripts.align.collect_anchors collect \
  --serial-index /path/to/serial_index.csv \
  --video-dir    /path/to/videos \
  --segment-id   20240806_153012 \
  --cam-serial   21401234 \
  --out          /tmp/anchors.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
# Library module: do NOT add handlers or set levels here. Let the caller (driver)
# configure logging. When run standalone (__main__), we'll install a minimal
# console handler that does not interfere with project-wide logging.
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small utils (concise names in logs)
# -----------------------------------------------------------------------------
def _short_name(p: str | Path) -> str:
    """basename with extension (e.g., 'file.csv')."""
    try:
        return Path(p).name
    except Exception:
        return str(p)


def _short_stem(p: str | Path) -> str:
    """stem without extension (e.g., 'file')."""
    try:
        return Path(p).stem
    except Exception:
        return str(p)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Anchor:
    """
    A synchronization anchor tying a camera frame to an audio sample index.

    Attributes
    ----------
    serial         : Positive serial embedded in the video JSON for that frame.
    audio_sample   : Start-sample index of the matching serial block in the
                     recorder timeline (units: samples).
    cam_serial     : Stable camera hardware serial (string).
    segment_id     : VideoGroup identifier (e.g., basename with timestamp tail).
    frame_id       : Raw (zero-based) frame id of that video frame.
    frame_id_reidx : Re-indexed frame id (from JSON), typically contiguous
                     after any repairs / filtering.
    """

    serial: int
    audio_sample: int
    cam_serial: str
    segment_id: str
    frame_id: int
    frame_id_reidx: int


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_serial_index_csv(path: Path) -> Dict[int, int]:
    """
    Load CSV with columns: serial,start_sample,end_sample → {serial: start_sample}.
    First occurrence wins; malformed rows are skipped.
    """
    mapping: Dict[int, int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = int((row.get("serial") or "").strip())
                start = int((row.get("start_sample") or "").strip())
            except Exception:
                continue
            mapping.setdefault(s, start)

    if not mapping:
        raise ValueError(f"No valid rows found in serial index CSV: {path}")
    logger.info(
        "Loaded %d serial→sample entries from %s", len(mapping), _short_name(path)
    )
    return mapping


def label_frames(serials: Sequence[int], frame_ids: Sequence[int]) -> List[str]:
    """
    Label each frame for anchor selection.

    NORMAL:    Δfid == 1 and Δserial == 1
    DUPLICATE: Δfid == 1 and Δserial == 0
    DROP:      Δfid  > 1
    MISSING:   serial <= 0 or None
    """
    labels: List[str] = []
    prev_fid: Optional[int] = None
    prev_s: Optional[int] = None
    for s, f in zip(serials, frame_ids):
        if s is None or s <= 0:
            labels.append("MISSING")
        elif prev_fid is None:
            labels.append("NORMAL")
        else:
            df = f - prev_fid
            ds = s - (prev_s if prev_s is not None else s)
            if df == 1 and ds == 1:
                labels.append("NORMAL")
            elif df == 1 and ds == 0:
                labels.append("DUPLICATE")
            elif df > 1:
                labels.append("DROP")
            else:
                labels.append("MISSING")
        prev_fid, prev_s = f, s
    return labels


def _extract_cam_arrays(
    videogroup, cam_serial: str
) -> Tuple[List[int], List[int], List[int]]:
    """
    From a VideoGroup, fetch CamJson for `cam_serial` and return:
        (fixed_serials, fixed_frame_ids, fixed_reidx_frame_ids).

    Strict requirement: all three must exist, be integer-like, non-empty,
    and have identical lengths. Raises RuntimeError otherwise.
    """
    # locate CamJson
    cj = None
    if getattr(videogroup, "json", None) and getattr(
        videogroup.json, "cam_jsons", None
    ):
        cj = videogroup.json.cam_jsons.get(str(cam_serial))
    if cj is None:
        raise RuntimeError(
            f"Group '{videogroup.group_id}': no cam_json for camera {cam_serial}"
        )

    serials = getattr(cj, "fixed_serials", None)
    frame_ids = getattr(cj, "fixed_frame_ids", None)
    frame_ids_reidx = getattr(cj, "fixed_reidx_frame_ids", None)

    if serials is None:
        raise RuntimeError("CamJson missing required field `fixed_serials`.")
    if frame_ids is None:
        raise RuntimeError("CamJson missing required field `fixed_frame_ids`.")
    if frame_ids_reidx is None:
        raise RuntimeError("CamJson missing required field `fixed_reidx_frame_ids`.")

    try:
        serials_list = [int(x) for x in list(serials)]
        frame_ids_list = [int(x) for x in list(frame_ids)]
        frame_ids_reidx_list = [int(x) for x in list(frame_ids_reidx)]
    except Exception as e:
        raise RuntimeError(f"CamJson fixed_* fields are not integer-like: {e}") from e

    if not serials_list or not frame_ids_list or not frame_ids_reidx_list:
        raise RuntimeError("CamJson fixed_* fields must be non-empty.")
    if not (len(serials_list) == len(frame_ids_list) == len(frame_ids_reidx_list)):
        raise RuntimeError(
            "CamJson length mismatch among fixed_serials / fixed_frame_ids / "
            f"fixed_reidx_frame_ids: "
            f"{len(serials_list)}, {len(frame_ids_list)}, {len(frame_ids_reidx_list)}"
        )

    return serials_list, frame_ids_list, frame_ids_reidx_list


def _collect_anchors_for_cam(
    index_map: Dict[int, int],
    segment_id: str,
    cam_serial: str,
    serials: Sequence[int],
    frame_ids: Sequence[int],
    frame_ids_reidx: Sequence[int],
    *,
    min_k: int = 3,
    min_span_ratio: float = 0.05,
) -> List[Anchor]:
    """
    Build anchors for one (segment_id, cam_serial) pair from per-frame arrays.
    """
    labels = label_frames(serials, frame_ids)

    cand: List[Tuple[int, int]] = [
        (i, int(s))
        for i, (lab, s) in enumerate(zip(labels, serials))
        if lab == "NORMAL" and s in index_map
    ]

    if len(cand) < min_k:
        logger.warning(
            "Few anchors for %s cam %s: %d", segment_id, cam_serial, len(cand)
        )

    if cand:
        s_vals = [s for _, s in cand]
        local_span = (max(s_vals) - min(s_vals)) if len(s_vals) > 1 else 0
        expected_span = max(1, int(min_span_ratio * (max(s_vals) - min(s_vals) + 1)))
        if local_span < expected_span:
            logger.warning(
                "Low anchor span for %s cam %s: span=%d (expect ≥ %d)",
                segment_id,
                cam_serial,
                local_span,
                expected_span,
            )

    anchors: List[Anchor] = [
        Anchor(
            serial=s,
            audio_sample=int(index_map[s]),
            cam_serial=str(cam_serial),
            segment_id=str(segment_id),
            frame_id=int(frame_ids[i]),
            frame_id_reidx=int(frame_ids_reidx[i]),
        )
        for i, s in cand
    ]
    return anchors


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def save_anchors_for_camera(
    serial_csv: Path | str,
    video_dir: Path | str,
    segment_id: str,
    cam_serial: str,
    out_json: Path | str,
    *,
    min_k: int = 3,
    min_span_ratio: float = 0.05,
) -> Path:
    """
    Collect anchors for (segment_id, cam_serial) and save them as JSON.

    Parameters
    ----------
    serial_csv      : CSV with (serial,start_sample,end_sample) from serial-audio decode.
    video_dir       : Root directory with <segment_id>.json and MP4s.
    segment_id      : VideoGroup id to process.
    cam_serial      : Camera hardware serial.
    out_json        : Destination JSON file.
    min_k           : Warn if fewer anchors than this.
    min_span_ratio  : Span heuristic for serial coverage.

    Returns
    -------
    Path to the written JSON.
    """
    from scripts.discover import (
        discover_segment,
    )  # local import to avoid heavy import at module load

    serial_csv = Path(serial_csv)
    video_dir = Path(video_dir)
    out_json = Path(out_json)

    index_map = load_serial_index_csv(serial_csv)

    vg = discover_segment(video_dir, segment_id)
    if vg is None:
        raise RuntimeError(f"Segment '{segment_id}' not found under {video_dir}")

    serials, frame_ids, frame_ids_reidx = _extract_cam_arrays(vg, cam_serial)
    anchors = _collect_anchors_for_cam(
        index_map,
        segment_id,
        cam_serial,
        serials,
        frame_ids,
        frame_ids_reidx,
        min_k=min_k,
        min_span_ratio=min_span_ratio,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps([asdict(a) for a in anchors], indent=2))
    logger.info(
        "Saved %d anchors for seg=%s cam=%s → %s",
        len(anchors),
        segment_id,
        cam_serial,
        _short_name(out_json),
    )
    return out_json


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anchor-collect",
        description="Collect anchors for one camera in one segment and save as JSON.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="Collect anchors and write JSON")
    c.add_argument(
        "--serial-index",
        required=True,
        help="CSV with columns: serial,start_sample,end_sample",
    )
    c.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing <segment_id>.json and MP4s",
    )
    c.add_argument("--segment-id", required=True, help="VideoGroup (segment) id")
    c.add_argument("--cam-serial", required=True, help="Camera hardware serial")
    c.add_argument("--out", required=True, help="Output JSON path")
    c.add_argument(
        "--min-k", type=int, default=3, help="Warn if fewer anchors than this"
    )
    c.add_argument(
        "--min-span",
        type=float,
        default=0.05,
        help="Span heuristic ratio for serial coverage",
    )
    c.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (standalone only; ignored when called from driver)",
    )
    return p


# Standalone logging that won't interfere with the driver:
# - Install a single console handler ONLY if root has no handlers (i.e., standalone).
# - Compact format with [seg/cam] tagging using CLI values.
class _StandaloneSegCamFilter(logging.Filter):
    def __init__(self, seg: str, cam: str) -> None:
        super().__init__()
        self.seg = seg
        self.cam = cam

    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure seg/cam exist for the format string
        if not hasattr(record, "seg") or record.seg in (None, "-", ""):
            record.seg = self.seg or "-"
        if not hasattr(record, "cam") or record.cam in (None, "-", ""):
            record.cam = self.cam or "-"
        return True


def configure_standalone_logging(level: str, seg: str, cam: str) -> None:
    """
    Minimal, non-intrusive console logging for `python -m ...` use.
    If root already has handlers (i.e., invoked by the driver), this is a no-op.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # Respect project's global logging config

    lvl = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(lvl)

    h = logging.StreamHandler()
    h.setLevel(lvl)
    h.addFilter(_StandaloneSegCamFilter(seg=seg, cam=cam))
    h.setFormatter(logging.Formatter("[%(levelname)s] [%(seg)s/%(cam)s] %(message)s"))
    root.addHandler(h)


def _cmd_collect(ns: argparse.Namespace) -> int:
    # Standalone: concise console logging with seg/cam tag.
    configure_standalone_logging(ns.log_level, ns.segment_id, ns.cam_serial)

    save_anchors_for_camera(
        ns.serial_index,
        ns.video_dir,
        ns.segment_id,
        ns.cam_serial,
        ns.out,
        min_k=ns.min_k,
        min_span_ratio=ns.min_span,
    )
    # Extra concise summary for humans:
    logger.info(
        "Anchors written → %s (seg=%s cam=%s)",
        _short_name(ns.out),
        ns.segment_id,
        ns.cam_serial,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "collect":
            return _cmd_collect(args)
        parser.error("unknown command")  # pragma: no cover
    except Exception as e:
        # If standalone logging is active, this will print a readable stack trace.
        logger.exception(e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
