#!/usr/bin/env python3
"""
scripts/anchor_collect.py — Per-segment, per-camera anchor extractor

Uses your lightweight discover path:
    from scripts.discover import discover_segment

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
    "frame_id": <int>
  }, ...
]

CLI
---
python -m scripts.anchor_collect collect \
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


# ---------------------------- logging ----------------------------------------
logger = logging.getLogger("anchor_collect")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# --------------------------- data model --------------------------------------
@dataclass(frozen=True)
class Anchor:
    """
    A synchronization anchor tying a camera frame to an audio sample index.

    Attributes
    ----------
    serial       : Positive serial embedded in the video JSON for that frame.
    audio_sample : Start-sample index of the matching serial block in the
                   recorder timeline (units: samples).
    cam_serial   : Stable camera hardware serial (string).
    segment_id   : VideoGroup identifier (e.g., basename with timestamp tail).
    frame_id     : Actual (zero-based) frame id of that video frame.
    """

    serial: int
    audio_sample: int
    cam_serial: str
    segment_id: str
    frame_id: int


# --------------------------- helpers -----------------------------------------
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
    logger.info("Loaded %d serial→sample entries from %s", len(mapping), path)
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


def _extract_cam_arrays(videogroup, cam_serial: str) -> Tuple[List[int], List[int]]:
    """
    From a VideoGroup, fetch CamJson for `cam_serial` and return
    (fixed_serials, fixed_frame_ids).

    Strict requirement: both `fixed_serials` and `fixed_frame_ids` must exist.
    No fallbacks. Raises on missing fields, empty arrays, or length mismatch.
    """
    # Find CamJson
    cj = None
    if getattr(videogroup, "json", None) and getattr(
        videogroup.json, "cam_jsons", None
    ):
        cj = videogroup.json.cam_jsons.get(str(cam_serial))
    if cj is None:
        raise RuntimeError(
            f"Group '{videogroup.group_id}': no cam_json for camera {cam_serial}"
        )

    # Require fixed_* fields
    serials = getattr(cj, "fixed_serials", None)
    frame_ids = getattr(cj, "fixed_frame_ids", None)
    if serials is None:
        raise RuntimeError("CamJson missing required field `fixed_serials`.")
    if frame_ids is None:
        raise RuntimeError("CamJson missing required field `fixed_frame_ids`.")

    # Normalize to Python lists of ints
    try:
        serials_list = [int(x) for x in list(serials)]
        frame_ids_list = [int(x) for x in list(frame_ids)]
    except Exception as e:
        raise RuntimeError(f"CamJson fixed_* fields are not integer-like: {e}") from e

    # Basic validations
    if not serials_list or not frame_ids_list:
        raise RuntimeError("CamJson fixed_* fields must be non-empty.")
    if len(serials_list) != len(frame_ids_list):
        raise RuntimeError(
            f"CamJson length mismatch: len(fixed_serials)={len(serials_list)} "
            f"!= len(fixed_frame_ids)={len(frame_ids_list)}"
        )

    return serials_list, frame_ids_list


def _collect_anchors_for_cam(
    index_map: Dict[int, int],
    segment_id: str,
    cam_serial: str,
    serials: Sequence[int],
    frame_ids: Sequence[int],
    *,
    min_k: int = 3,
    min_span_ratio: float = 0.05,
) -> List[Anchor]:
    """
    Build anchors for one (segment_id, cam_serial) pair from per-frame arrays.
    """
    labels = label_frames(serials, frame_ids)

    cand = [
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
        # basic span heuristic relative to local range
        if local_span < max(1, int(min_span_ratio * (max(s_vals) - min(s_vals) + 1))):
            logger.warning(
                "Low anchor span for %s cam %s: span=%d",
                segment_id,
                cam_serial,
                local_span,
            )

    anchors: List[Anchor] = [
        Anchor(
            serial=s,
            audio_sample=int(index_map[s]),
            cam_serial=str(cam_serial),
            segment_id=str(segment_id),
            frame_id=int(frame_ids[i]),
        )
        for i, s in cand
    ]
    return anchors


# ----------------------------- public API ------------------------------------
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
    serial_csv   : CSV with (serial,start_sample,end_sample) from serial-audio decode.
    video_dir    : Root directory with <segment_id>.json and MP4s.
    segment_id   : VideoGroup id to process.
    cam_serial   : Camera hardware serial.
    out_json     : Destination JSON file.
    min_k        : Warn if fewer anchors than this.
    min_span_ratio : Span heuristic for serial coverage.

    Returns
    -------
    Path to the written JSON.
    """
    from scripts.discover import discover_segment  # local import

    serial_csv = Path(serial_csv)
    video_dir = Path(video_dir)
    out_json = Path(out_json)

    index_map = load_serial_index_csv(serial_csv)

    vg = discover_segment(video_dir, segment_id)
    if vg is None:
        raise RuntimeError(f"Segment '{segment_id}' not found under {video_dir}")

    serials, frame_ids = _extract_cam_arrays(vg, cam_serial)
    anchors = _collect_anchors_for_cam(
        index_map,
        segment_id,
        cam_serial,
        serials,
        frame_ids,
        min_k=min_k,
        min_span_ratio=min_span_ratio,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps([asdict(a) for a in anchors], indent=2))
    logger.info(
        "Saved %d anchors for segment=%s cam=%s → %s",
        len(anchors),
        segment_id,
        cam_serial,
        out_json,
    )
    return out_json


# ------------------------------- CLI -----------------------------------------
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
        help="Logging verbosity",
    )
    return p


def _cmd_collect(ns: argparse.Namespace) -> int:
    logger.setLevel(getattr(logging, ns.log_level))
    save_anchors_for_camera(
        ns.serial_index,
        ns.video_dir,
        ns.segment_id,
        ns.cam_serial,
        ns.out,
        min_k=ns.min_k,
        min_span_ratio=ns.min_span,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "collect":
            return _cmd_collect(args)
        parser.error("unknown command")
    except Exception as e:
        logger.exception(e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
