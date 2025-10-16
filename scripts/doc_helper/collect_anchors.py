#!/usr/bin/env python3
"""
collect_anchors_from_camjson.py

Build per-frame anchors by matching camera-side fixed chunk serials to the
audio serial index CSV.

IMPORTANT: Anchor selection uses FIXED frame IDs (Δ fixed_frame_id == 1).

Inputs
------
1) --csv : CSV with columns: serial,start_sample,end_sample
           (first occurrence of a serial wins; maps serial -> start_sample)
2) --json: Camera JSON with:
   {
     "camera_serial": "<str>",
     "frame_ids": [int, ...],
     "chunk_serials": [int, ...],
     "fixed_frame_ids": [int, ...],
     "fixed_chunk_serials": [int, ...]
   }
3) --out-dir: Output directory

Output
------
Writes <json_stem>.<camera_serial>.anchors.json containing:
[
  {
    "serial": <int>,            # fixed_chunk_serial
    "audio_sample": <int>,      # start_sample from CSV
    "cam_serial": "<str>",
    "frame_id": <int>,          # raw frame id at that index
    "frame_id_fixed": <int>     # fixed frame id at that index
  },
  ...
]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ----------------------------- data model ------------------------------------
@dataclass(frozen=True)
class Anchor:
    serial: int
    audio_sample: int
    cam_serial: str
    frame_id: int
    frame_id_fixed: int


# ----------------------------- I/O helpers -----------------------------------
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
    return mapping


def load_cam_json(path: Path) -> Tuple[str, List[int], List[int], List[int], List[int]]:
    """
    Read the camera JSON and coerce arrays to int lists.

    Returns
    -------
    (camera_serial, frame_ids, chunk_serials, fixed_frame_ids, fixed_chunk_serials)
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required = [
        "camera_serial",
        "frame_ids",
        "chunk_serials",
        "fixed_frame_ids",
        "fixed_chunk_serials",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"JSON missing required field(s): {', '.join(missing)}")

    cam_serial = str(data["camera_serial"])
    frame_ids = [int(x) for x in list(data["frame_ids"])]
    chunk_serials = [int(x) for x in list(data["chunk_serials"])]
    fixed_frame_ids = [int(x) for x in list(data["fixed_frame_ids"])]
    fixed_chunk_serials = [int(x) for x in list(data["fixed_chunk_serials"])]

    n = len(frame_ids)
    if not (
        n
        and len(chunk_serials) == n
        and len(fixed_frame_ids) == n
        and len(fixed_chunk_serials) == n
    ):
        raise ValueError(
            "JSON arrays must be non-empty and of equal length: "
            f"len(frame_ids)={len(frame_ids)}, "
            f"len(chunk_serials)={len(chunk_serials)}, "
            f"len(fixed_frame_ids)={len(fixed_frame_ids)}, "
            f"len(fixed_chunk_serials)={len(fixed_chunk_serials)}"
        )
    return cam_serial, frame_ids, chunk_serials, fixed_frame_ids, fixed_chunk_serials


# ----------------------------- core logic ------------------------------------
def label_frames(fid: Sequence[int]) -> List[str]:
    """
    NORMAL if Δfid == 1; DROP if Δfid > 1; else UNCLASSIFIED (incl. first).
    """
    out: List[str] = []
    prev: Optional[int] = None
    for f in fid:
        if prev is None:
            out.append("UNCLASSIFIED")
        else:
            df = f - prev
            if df == 1:
                out.append("NORMAL")
            elif df > 1:
                out.append("DROP")
            else:
                out.append("UNCLASSIFIED")
        prev = f
    return out


def collect_anchors(
    index_map: Dict[int, int],
    cam_serial: str,
    frame_ids_raw: Sequence[int],
    fixed_frame_ids: Sequence[int],
    fixed_chunk_serials: Sequence[int],
) -> List[Anchor]:
    # NOTE: Use FIXED frame IDs for selection.
    labels = label_frames(fixed_frame_ids)

    anchors: List[Anchor] = []
    for i, (lab, s) in enumerate(zip(labels, fixed_chunk_serials)):
        if lab != "NORMAL":
            continue
        if s not in index_map:
            continue
        anchors.append(
            Anchor(
                serial=int(s),
                audio_sample=int(index_map[int(s)]),
                cam_serial=str(cam_serial),
                frame_id=int(frame_ids_raw[i]),
                frame_id_fixed=int(fixed_frame_ids[i]),
            )
        )
    return anchors


# ----------------------------- CLI plumbing ----------------------------------
def derive_out_path(in_json: Path, cam_serial: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{in_json.stem}.{cam_serial}.anchors.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect anchors by matching fixed_chunk_serials to a serial index CSV (selection via FIXED frame IDs)."
    )
    p.add_argument(
        "--csv",
        required=True,
        help="Path to serial index CSV (serial,start_sample,end_sample).",
    )
    p.add_argument(
        "--json", required=True, help="Path to camera JSON with required fields."
    )
    p.add_argument("--out-dir", required=True, help="Directory to write anchors JSON.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)

    if not csv_path.is_file():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 2
    if not json_path.is_file():
        print(f"[ERROR] JSON not found: {json_path}")
        return 2

    try:
        index_map = load_serial_index_csv(csv_path)
        cam_serial, frame_ids, _chunk_serials, fixed_frame_ids, fixed_chunk_serials = (
            load_cam_json(json_path)
        )
        anchors = collect_anchors(
            index_map, cam_serial, frame_ids, fixed_frame_ids, fixed_chunk_serials
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    out_path = derive_out_path(json_path, cam_serial, out_dir)
    try:
        out_path.write_text(
            json.dumps([asdict(a) for a in anchors], indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}")
        return 2

    print(f"[OK] {len(anchors)} anchors → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
