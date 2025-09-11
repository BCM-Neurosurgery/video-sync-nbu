#!/usr/bin/env python3
"""
scripts/align/collect_anchors.py — Per-segment, per-camera anchor extractor (API-only)

Inputs
------
1) Serial index CSV from serial-audio decode (columns: serial,start_sample,end_sample)
2) A `scripts.models.Video` object whose `companion_json` provides:
   - fixed_serials
   - fixed_frame_ids           (un-reindexed)
   - fixed_reidx_frame_ids     (reindexed/contiguous)

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
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

from scripts.models import Video  # expects .companion_json with fixed_* arrays
from scripts.utility.utils import _name


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
    logger.info("Loaded %d serial→sample entries from %s", len(mapping), _name(path))
    return mapping


def label_frames(frame_ids: Sequence[int]) -> List[str]:
    """
    Label each frame for anchor selection using ONLY frame_ids.

    NORMAL      : Δfid == 1
    DROP        : Δfid > 1
    UNCLASSIFIED: all other cases (including first frame, Δfid <= 0)
    """
    labels: List[str] = []
    prev_fid: Optional[int] = None
    for f in frame_ids:
        if prev_fid is None:
            labels.append("UNCLASSIFIED")
        else:
            df = f - prev_fid
            if df == 1:
                labels.append("NORMAL")
            elif df > 1:
                labels.append("DROP")
            else:
                labels.append("UNCLASSIFIED")
        prev_fid = f
    return labels


def _extract_cam_arrays_from_video(
    video: Video,
) -> Tuple[List[int], List[int], List[int]]:
    """
    From a Video, fetch its companion CamJson arrays and return:
        (fixed_serials, fixed_frame_ids, fixed_reidx_frame_ids).

    Strict requirement: all three must exist, be integer-like, non-empty,
    and have identical lengths. Raises RuntimeError otherwise.
    """
    cj = getattr(video, "companion_json", None)
    if cj is None:
        raise RuntimeError(f"Video {video.path}: missing companion_json")

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

    Selection: frames labeled NORMAL by frame_id deltas (Δfid==1) and whose
    serial exists in the index_map.
    """
    labels = label_frames(frame_ids)

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
    video: Video,
    out_json: Path | str,
    *,
    min_k: int = 3,
    min_span_ratio: float = 0.05,
) -> Path:
    """
    Collect anchors for a single `Video` and save them as JSON.

    Parameters
    ----------
    serial_csv      : CSV with (serial,start_sample,end_sample) from serial-audio decode.
    video           : scripts.models.Video with companion_json providing fixed_* arrays.
    out_json        : Destination JSON file.
    min_k           : Warn if fewer anchors than this.
    min_span_ratio  : Span heuristic for serial coverage.

    Returns
    -------
    Path to the written JSON.
    """
    serial_csv = Path(serial_csv)
    out_json = Path(out_json)

    index_map = load_serial_index_csv(serial_csv)

    serials, frame_ids, frame_ids_reidx = _extract_cam_arrays_from_video(video)
    anchors = _collect_anchors_for_cam(
        index_map,
        str(getattr(video, "segment_id", "")),
        str(getattr(video, "cam_serial", "")),
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
        getattr(video, "segment_id", ""),
        getattr(video, "cam_serial", ""),
        _name(out_json),
    )
    return out_json
