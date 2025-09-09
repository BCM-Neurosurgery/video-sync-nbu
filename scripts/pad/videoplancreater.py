#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
videoplancreater.py
===================

Create a *video padding plan* from a frame-id analysis JSON produced by
`video_frameid_analysis.py`.

- Inserts frames only for `forward_jump` events, with insert = diff - expect_step.
- Default padding policy is "dup-prev" (duplicate the frame before the gap).
- Uses scripts.log.logutils for consistent [seg/cam] logging.
- Uses scripts.parsers.videofileparser.VideoFileParser to parse source FPS.

Input (analysis JSON — key fields)
----------------------------------
{
  "segment_id": "<SEGMENT_ID>",
  "cam_serial": "<CAM_SERIAL>",
  "video": "/full/path/to/<SEGMENT_ID>.<CAM_SERIAL>.mp4",
  "counts": {"ok": ..., "forward_jump": ..., "drop": ..., "duplicate": ...},
  "missing_frames": <int>,
  "events": [{"i": <int>, "prev": <int>, "curr": <int>, "diff": <int>, "type": "duplicate|forward_jump|drop"}, ...]
}

Output (padding plan JSON)
--------------------------
{
  "version": 1,
  "segment_id": "...",
  "cam_serial": "...",
  "source_video": ".../TRBD001_...23512909.mp4",
  "source_fps": 29.97,
  "target_fps": 30.0,
  "policy": "dup-prev",
  "total_insertions": 48,
  "operations": [
    {"after_index":10025, "insert":7,  "frame_id_before":30816, "frame_id_after":30824},
    {"after_index":10026, "insert":40, "frame_id_before":30824, "frame_id_after":30865}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from scripts.log.logutils import (
    configure_standalone_logging,
    log_context,
)
from scripts.parsers.videofileparser import VideoFileParser

# Module-level logger; handlers/formatting are controlled by logutils.
log = logging.getLogger(__name__)


# -----------------------------
# Core plan creation
# -----------------------------
def create_video_padding_plan(
    analysis_json: Path,
    *,
    target_fps: float = 30.0,
    policy: str = "dup-prev",
    expect_step: int = 1,
    outdir: Optional[Path] = None,
) -> Path:
    """
    Create a padding plan from a frame-id analysis JSON.

    Parameters
    ----------
    analysis_json : Path
        Path to the `*-frameid.json` produced by `video_frameid_analysis.py`.
    target_fps : float, default 30.0
        Desired output framerate in the padded video plan.
    policy : {"dup-prev"}, default "dup-prev"
        Padding policy; currently supports duplicating the previous frame at each insertion point.
    expect_step : int, default 1
        Expected increment between consecutive frame IDs (used to compute gap size).
    outdir : Path | None
        Where to write the plan JSON. Defaults to the same directory as the analysis JSON.

    Returns
    -------
    Path
        Path to the written padding plan JSON.
    """
    data = json.loads(Path(analysis_json).read_text(encoding="utf-8"))

    # Basic validations (pre-context)
    for key in ("segment_id", "cam_serial", "video", "events"):
        if key not in data:
            raise KeyError(f"Missing required key in analysis JSON: {key!r}")

    segment_id = str(data["segment_id"])
    cam_serial = str(data["cam_serial"])
    video_path = Path(str(data["video"]))
    events: List[Dict[str, Any]] = list(data["events"])

    if policy not in {"dup-prev"}:
        raise ValueError(f"Unsupported policy: {policy!r}. Try 'dup-prev'.")

    adapter = logging.LoggerAdapter(log, extra={"seg": segment_id, "cam": cam_serial})
    with log_context(seg=segment_id, cam=cam_serial):
        # Compute operations from forward jumps only
        ops: List[Dict[str, Any]] = []
        total_insertions = 0
        for ev in events:
            if ev.get("type") != "forward_jump":
                continue
            diff = int(ev.get("diff", 0))
            needed = diff - int(expect_step)
            if needed <= 0:
                # Defensive: skip anomalies that don't imply missing frames per 'expect_step'
                continue
            op = {
                "after_index": int(ev["i"]),
                "insert": int(needed),
                "frame_id_before": int(ev["prev"]),
                "frame_id_after": int(ev["curr"]),
            }
            ops.append(op)
            total_insertions += needed

        # Source FPS via VideoFileParser (best-effort)
        try:
            vinfo = VideoFileParser(str(video_path))
            src_fps = float(vinfo.fps)
        except Exception as e:
            adapter.warning(
                "VideoFileParser failed to parse FPS (%s); falling back to target_fps (%.6g).",
                e,
                target_fps,
            )
            src_fps = float(target_fps)

        # Build plan
        plan: Dict[str, Any] = {
            "version": 1,
            "segment_id": segment_id,
            "cam_serial": cam_serial,
            "source_video": str(video_path),
            "source_fps": float(round(src_fps, 5)),
            "target_fps": float(round(float(target_fps), 5)),
            "policy": policy,
            "total_insertions": int(total_insertions),
            "operations": ops,
        }

        # Name alongside analysis JSON by default
        outdir = Path(outdir) if outdir else Path(analysis_json).parent
        outdir.mkdir(parents=True, exist_ok=True)
        # Use the video stem for intuitive naming: <SEG>.<CAM>-videopad.json
        stem = video_path.stem  # "<SEGMENT_ID>.<CAM_SERIAL>"
        out_path = outdir / f"{stem}-videopad.json"
        out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        # Sanity check vs analysis 'missing_frames' if present
        mf = data.get("missing_frames")
        if isinstance(mf, int) and mf != total_insertions:
            adapter.warning(
                "Total insertions (%d) ≠ analysis.missing_frames (%d). "
                "Proceeding anyway. Check expect_step or analysis events.",
                total_insertions,
                mf,
            )

        adapter.info(
            "Padding plan written: %s (ops=%d, total_insertions=%d, src_fps=%.5f → tgt_fps=%.5f, policy=%s)",
            out_path.name,
            len(ops),
            total_insertions,
            plan["source_fps"],
            plan["target_fps"],
            policy,
        )
        return out_path


# -----------------------------
# CLI
# -----------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Create a video padding plan from a frame-id analysis JSON."
    )
    ap.add_argument("analysis_json", help="Path to <video_stem>-frameid.json")
    ap.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target FPS for the padded video (default: 30.0).",
    )
    ap.add_argument(
        "--policy",
        default="dup-prev",
        choices=["dup-prev"],
        help="Padding policy to apply (default: dup-prev).",
    )
    ap.add_argument(
        "--expect-step",
        type=int,
        default=1,
        help="Expected +step between frame IDs (default: 1).",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Directory for output JSON (default: alongside the analysis JSON).",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    # Standalone console logger (no-op if the driver already configured root)
    configure_standalone_logging(args.log_level, seg="-", cam="-")
    try:
        create_video_padding_plan(
            Path(args.analysis_json),
            target_fps=float(args.target_fps),
            policy=str(args.policy),
            expect_step=int(args.expect_step),
            outdir=Path(args.outdir) if args.outdir else None,
        )
        return 0
    except Exception as e:
        # Use module-level logger for consistent formatting
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
