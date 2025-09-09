#!/usr/bin/env python3
"""
video_frameid_analysis.py
=========================

Given a video .mp4 path, this script:
1) Infers the companion JSON as <SEGMENT_ID>.json in the same folder.
2) Loads it via JsonParser.
3) Extracts the fixed frame-id stream for that camera (FrameIDFixer).
4) Analyzes monotonicity (expect +1) and missing frames.
5) Writes a formatted text report to --outdir (default: alongside the video) as:
      <video_stem>-frameid.txt
6) Writes a machine-readable JSON analysis (same basename) as:
      <video_stem>-frameid.json

Outputs
-------
- Text report:    <SEGMENT_ID>.<CAM_SERIAL>-frameid.txt
- JSON analysis:  <SEGMENT_ID>.<CAM_SERIAL>-frameid.json
  Schema (key fields):
    {
      "segment_id": "<SEGMENT_ID>",
      "cam_serial": "<CAM_SERIAL>",
      "video": "/full/path/to/<SEGMENT_ID>.<CAM_SERIAL>.mp4",
      "counts": {"ok": ..., "forward_jump": ..., "drop": ..., "duplicate": ...},
      "missing_frames": <int>,
      "events": [{"i": <int>, "prev": <int>, "curr": <int>, "diff": <int>, "type": "duplicate|forward_jump|drop"}, ...]
    }

Logging
-------
Uses scripts.log.logutils. When run standalone, console logs are stamped as [seg/cam].
"""

from __future__ import annotations

import argparse
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

from scripts.log.logutils import (
    configure_standalone_logging,
    log_context,
)

from scripts.parsers.jsonfileparser import JsonParser
from scripts.analysis.serial_analysis import analyze, summarize_text


# -----------------------------
# Public result container
# -----------------------------
@dataclass(frozen=True)
class FrameIDAnalysisResult:
    video_path: Path
    json_path: Path
    out_path: Path
    segment_id: str
    cam_serial: str
    strictly_monotonic: bool
    missing_frames: int
    counts: Dict[str, int]


# -----------------------------
# Helpers
# -----------------------------
def parse_video_name(p: Path) -> Tuple[str, str]:
    """
    Parse <SEGMENT_ID>.<CAM_SERIAL>.mp4 → (segment_id, cam_serial).
    Example: TRBD002_20250806_104707.23512909.mp4 → ("TRBD002_20250806_104707", "23512909")
    """
    stem = p.stem
    if "." not in stem:
        raise ValueError(f"Unexpected video name (no dot before cam serial): {p.name}")
    seg, cam = stem.rsplit(".", 1)
    if not seg or not cam:
        raise ValueError(f"Could not parse segment/camera from: {p.name}")
    return seg, cam


def find_companion_json(video_path: Path) -> Path:
    seg, _ = parse_video_name(video_path)
    return video_path.with_name(f"{seg}.json")


def choose_cam_key(requested: str, available: list) -> object:
    """
    Match camera serial whether JSON stores it as str or int.
    Returns the key to pass into JsonParser getters.
    """
    if requested in available:
        return requested
    try:
        as_int = int(requested)
        if as_int in available:
            return as_int
    except ValueError:
        pass
    raise KeyError(f"Camera serial {requested!r} not in {available}")


# -----------------------------
# Public API
# -----------------------------
def analyze_video_frameids(
    video: Union[str, Path],
    *,
    outdir: Optional[Union[str, Path]] = None,
    expect_step: int = 1,
    hist_cols: int = 2,
    top: int = 5,
    log_level: str = "INFO",
) -> FrameIDAnalysisResult:
    """
    Programmatic API to analyze a video's fixed frame-id stream.

    Parameters
    ----------
    video : str | Path
        Path to the .mp4 named <SEGMENT_ID>.<CAM_SERIAL>.mp4
    outdir : str | Path | None
        Where to write the text report. Defaults to the video's directory.
    expect_step : int
        Expected increment between consecutive frame IDs (default: 1).
    hist_cols : int
        Histogram columns in the text report (default: 2).
    top : int
        How many top forward/drops to list (default: 5).
    log_level : str
        Console log level when used standalone (default: "INFO").

    Returns
    -------
    FrameIDAnalysisResult
        Summary including output path, monotonicity, missing frame count, and counts.

    Raises
    ------
    FileNotFoundError, ValueError, KeyError, RuntimeError
        If inputs are missing/invalid or analysis fails.
    """
    video_path = Path(video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"No such video: {video_path}")

    segment_id, cam_serial = parse_video_name(video_path)

    # Configure concise standalone logging; no-op if handlers already exist
    configure_standalone_logging(log_level, seg=segment_id, cam=cam_serial)
    log = logging.getLogger("sync")

    with log_context(seg=segment_id, cam=cam_serial):
        json_path = find_companion_json(video_path)
        if not json_path.is_file():
            raise FileNotFoundError(
                f"Companion JSON not found next to video: {json_path.name}"
            )

        parser = JsonParser(str(json_path))
        cam_key = choose_cam_key(cam_serial, parser.get_camera_serials())
        fixed_ids = parser.get_fixed_frame_ids_list(cam_key)
        if fixed_ids is None or len(fixed_ids) < 2:
            raise RuntimeError("Insufficient frame_id data (need at least 2 values).")

        result = analyze(fixed_ids, expect_step=int(expect_step), top_k=int(top))

        counts = result.counts or {}
        n_missing = int(result.total_missing_ids or 0)
        strictly_monotonic = (
            counts.get("duplicate", 0) == 0
            and counts.get("drop", 0) == 0
            and counts.get("forward_jump", 0) == 0
        )

        header = [
            f"VIDEO: {video_path.name}",
            f"JSON : {json_path.name}",
            f"CAM  : {cam_serial}",
            f"Expect step: {expect_step}",
            "",
        ]
        body = summarize_text(
            result, include_tops=True, hist_cols=max(1, int(hist_cols))
        )
        report_text = "\n".join(header) + body

        outdir_path = (
            Path(outdir).expanduser().resolve() if outdir else video_path.parent
        )
        outdir_path.mkdir(parents=True, exist_ok=True)
        out_path = outdir_path / f"{video_path.stem}-frameid.txt"
        out_path.write_text(report_text, encoding="utf-8")

        # --- JSON analysis next to the text report (same basename, .json) ---
        # Prefer events from `result` if present; otherwise derive them here.
        events = getattr(result, "events", None)
        if events is None:
            events = []
            exp = int(expect_step)
            prev = int(fixed_ids[0])
            for i in range(1, len(fixed_ids)):
                curr = int(fixed_ids[i])
                diff = curr - prev
                if diff == exp:
                    pass  # OK
                elif diff == 0:
                    events.append(
                        {
                            "i": i,
                            "prev": prev,
                            "curr": curr,
                            "diff": diff,
                            "type": "duplicate",
                        }
                    )
                elif diff > exp:
                    events.append(
                        {
                            "i": i,
                            "prev": prev,
                            "curr": curr,
                            "diff": diff,
                            "type": "forward_jump",
                        }
                    )
                else:
                    events.append(
                        {
                            "i": i,
                            "prev": prev,
                            "curr": curr,
                            "diff": diff,
                            "type": "drop",
                        }
                    )
                prev = curr

        analysis = {
            "segment_id": segment_id,
            "cam_serial": cam_serial,
            "video": str(video_path),
            "counts": counts,
            "missing_frames": n_missing,
            "events": events,
        }
        json_out_path = out_path.with_suffix(".json")
        json_out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        # --------------------------------------------------------------------

        # Concise status
        if strictly_monotonic:
            log.info(
                f"{video_path.name}: frame_id strictly monotonic (+{expect_step}); "
                f"report → {out_path.name}, json → {json_out_path.name}"
            )
        else:
            log.warning(
                f"{video_path.name}: NOT monotonic — missing_frames={n_missing}, "
                f"dups={counts.get('duplicate', 0)}, drops={counts.get('drop', 0)}, "
                f"forward_jumps={counts.get('forward_jump', 0)}; "
                f"report → {out_path.name}, json → {json_out_path.name}"
            )

        return FrameIDAnalysisResult(
            video_path=video_path,
            json_path=json_path,
            out_path=out_path,
            segment_id=segment_id,
            cam_serial=cam_serial,
            strictly_monotonic=strictly_monotonic,
            missing_frames=n_missing,
            counts=counts,
        )


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Analyze fixed frame_id sequence for a given video using its companion JSON."
    )
    ap.add_argument(
        "video", help="Path to the video .mp4 (named <SEGMENT_ID>.<CAM>.mp4)"
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Directory for report output (default: same folder as the video)",
    )
    ap.add_argument(
        "--expect-step",
        type=int,
        default=1,
        help="Expected increment between consecutive frame IDs (default: 1)",
    )
    ap.add_argument(
        "--hist-cols",
        type=int,
        default=2,
        help="Histogram columns in text report (default: 2)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top forward/drops to list (default: 5)",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO",
    )
    return ap


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        analyze_video_frameids(
            args.video,
            outdir=args.outdir,
            expect_step=args.expect_step,
            hist_cols=args.hist_cols,
            top=args.top,
            log_level=args.log_level,
        )
        return 0
    except Exception as e:
        # Minimal fallback formatting if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.error(str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
