#!/usr/bin/env python3
"""
video_analysis.py
=================

Given a video .mp4 path, this script:
1) Infers the companion JSON as <SEGMENT_ID>.json in the same folder.
2) Loads it via JsonParser.
3) Extracts the fixed frame-id stream for that camera (FrameIDFixer).
4) Analyzes monotonicity (expect +1) and missing frames.
5) Writes a machine-readable JSON analysis (default: alongside the video) as:
      <video_stem>-analysis.json

Outputs
-------
- JSON analysis:  <SEGMENT_ID>.<CAM_SERIAL>-analysis.json
  Schema (key fields):
    {
      "segment_id": "<SEGMENT_ID>",
      "cam_serial": "<CAM_SERIAL>",
      "video": "/full/path/to/<SEGMENT_ID>.<CAM_SERIAL>.mp4",
      "video_meta": {"fps": <float>, "duration": <float>, "resolution": "<WxH>", "frame_count": <int>},
      "counts": {"ok": ..., "forward_jump": ..., "drop": ..., "duplicate": ...},
      "missing_frames": <int>,
      "events": [{"i": <int>, "prev": <int>, "curr": <int>, "diff": <int>, "type": "duplicate|forward_jump|drop", "serial_prev": <int|null>, "serial_curr": <int|null>}, ...],
      "events_unreidx": [
        {
          "i": <int>,
          "prev": <int>, "curr": <int>, "diff": <int>,
          "type": "duplicate|forward_jump|drop",
          "serial_prev": <int|null>, "serial_curr": <int|null>
        },
        ...]
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
from typing import Dict, Optional, Sequence

from scripts.log.logutils import (
    configure_standalone_logging,
    log_context,
)

from scripts.analysis.serial_analysis import analyze
from scripts.models import Video


# -----------------------------
# Public result container
# -----------------------------
@dataclass(frozen=True)
class FrameIDAnalysisResult:
    video_path: Path
    json_path: Optional[Path]  # companion JSON path (source), if available
    out_json_path: Path  # where we saved the analysis JSON
    segment_id: str
    cam_serial: str
    strictly_monotonic: bool
    missing_frames: int
    counts: Dict[str, int]


# -----------------------------
# Public API
# -----------------------------
def analyze_video(
    video: Video,
    *,
    outdir: Optional[Path | str] = None,
    expect_step: int = 1,
    log_level: str = "INFO",
    suffix: str = "analysis",
) -> FrameIDAnalysisResult:
    """
    Analyze a video's fixed, reindexed frame-id stream using a `scripts.models.Video`.

    Requirements
    ------------
    - `video.segment_id` is used for logging/report headers.
    - `video.companion_json.fixed_reidx_frame_ids` MUST exist and have length >= 2.
    """
    video_path = Path(video.path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"No such video: {video_path}")

    segment_id = str(video.segment_id)
    cam_serial = str(video.cam_serial)

    # Configure concise standalone logging; no-op if handlers already exist
    configure_standalone_logging(log_level, seg=segment_id, cam=cam_serial)
    log = logging.getLogger("sync")

    with log_context(seg=segment_id, cam=cam_serial):
        cj = video.companion_json
        if cj is None:
            raise RuntimeError(f"{video_path.name}: Video has no companion_json.")
        if cj.fixed_reidx_frame_ids is None or len(cj.fixed_reidx_frame_ids) < 2:
            raise RuntimeError(
                f"{video_path.name}: companion_json.fixed_reidx_frame_ids missing/too short."
            )

        fixed_ids = list(cj.fixed_reidx_frame_ids)
        fixed_ids_unreidx = (
            list(getattr(cj, "fixed_frame_ids"))
            if getattr(cj, "fixed_frame_ids", None) is not None
            else None
        )
        fixed_serials = (
            list(getattr(cj, "fixed_serials"))
            if getattr(cj, "fixed_serials", None) is not None
            else None
        )

        # Core analysis
        result = analyze(fixed_ids, expect_step=int(expect_step))

        counts = result.counts or {}
        n_missing = int(result.total_missing_ids or 0)
        strictly_monotonic = (
            counts.get("duplicate", 0) == 0
            and counts.get("drop", 0) == 0
            and counts.get("forward_jump", 0) == 0
        )

        # Events JSON (derive if the analyzer didn't include them)
        events = getattr(result, "events", None)
        if events is None:
            events = []
            exp = int(expect_step)
            prev = int(fixed_ids[0])
            for i in range(1, len(fixed_ids)):
                curr = int(fixed_ids[i])
                diff = curr - prev
                serial_prev = (
                    int(fixed_serials[i - 1])
                    if fixed_serials and len(fixed_serials) > i - 1
                    else None
                )
                serial_curr = (
                    int(fixed_serials[i])
                    if fixed_serials and len(fixed_serials) > i
                    else None
                )
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
                            "serial_prev": serial_prev,
                            "serial_curr": serial_curr,
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
                            "serial_prev": serial_prev,
                            "serial_curr": serial_curr,
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
                            "serial_prev": serial_prev,
                            "serial_curr": serial_curr,
                        }
                    )
                prev = curr

        # Parallel events computed on the un-reindexed series (if available)
        events_unreidx = None
        if fixed_ids_unreidx is not None and len(fixed_ids_unreidx) >= 2:
            evu = []
            exp = int(expect_step)
            prev_u = int(fixed_ids_unreidx[0])
            for i in range(1, len(fixed_ids_unreidx)):
                curr_u = int(fixed_ids_unreidx[i])
                diff_u = curr_u - prev_u
                # serials at the transition (i-1 -> i), if available
                serial_prev = (
                    int(fixed_serials[i - 1])
                    if fixed_serials and len(fixed_serials) > i - 1
                    else None
                )
                serial_curr = (
                    int(fixed_serials[i])
                    if fixed_serials and len(fixed_serials) > i
                    else None
                )
                if diff_u == exp:
                    pass
                elif diff_u == 0:
                    evu.append(
                        {
                            "i": i,
                            "prev": prev_u,
                            "curr": curr_u,
                            "diff": diff_u,
                            "type": "duplicate",
                            "serial_prev": serial_prev,
                            "serial_curr": serial_curr,
                        }
                    )
                elif diff_u > exp:
                    evu.append(
                        {
                            "i": i,
                            "prev": prev_u,
                            "curr": curr_u,
                            "diff": diff_u,
                            "type": "forward_jump",
                            "serial_prev": serial_prev,
                            "serial_curr": serial_curr,
                        }
                    )
                else:
                    evu.append(
                        {
                            "i": i,
                            "prev": prev_u,
                            "curr": curr_u,
                            "diff": diff_u,
                            "type": "drop",
                            "serial_prev": serial_prev,
                            "serial_curr": serial_curr,
                        }
                    )
                prev_u = curr_u
            events_unreidx = evu

        # Include basic video metadata
        video_meta = {
            "fps": float(getattr(video, "frame_rate", 0.0) or 0.0),
            "duration": float(getattr(video, "duration", 0.0) or 0.0),
            "resolution": str(getattr(video, "resolution", "") or ""),
            "frame_count": int(getattr(video, "frame_count", 0) or 0),
        }

        analysis = {
            "segment_id": segment_id,
            "cam_serial": cam_serial,
            "video": str(video_path),
            "video_meta": video_meta,
            "counts": counts,
            "missing_frames": n_missing,
            "events": events,
            "events_unreidx": events_unreidx,
        }

        outdir_path = (
            Path(outdir).expanduser().resolve() if outdir else video_path.parent
        )
        outdir_path.mkdir(parents=True, exist_ok=True)
        json_out_path = outdir_path / f"{video_path.stem}-{suffix}.json"
        json_out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

        # Concise status
        if strictly_monotonic:
            log.info(
                f"{video_path.name}: frame_id strictly monotonic (+{expect_step}); json → {json_out_path.name}"
            )
        else:
            log.warning(
                f"{video_path.name}: NOT monotonic — missing_frames={n_missing}, "
                f"dups={counts.get('duplicate', 0)}, drops={counts.get('drop', 0)}, "
                f"forward_jumps={counts.get('forward_jump', 0)}; "
                f"json → {json_out_path.name}"
            )

        return FrameIDAnalysisResult(
            video_path=video_path,
            json_path=Path(cj.path) if getattr(cj, "path", None) else None,
            out_json_path=json_out_path,
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
        description="Analyze a video's fixed frame_id sequence using its companion JSON and write <video_stem>-analysis.json."
    )
    ap.add_argument(
        "video", help="Path to the video .mp4 (named <SEGMENT_ID>.<CAM>.mp4)"
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Directory for JSON output (default: same folder as the video)",
    )
    ap.add_argument(
        "--expect-step",
        type=int,
        default=1,
        help="Expected increment between consecutive frame IDs (default: 1)",
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
        # NOTE: the public API expects a `scripts.models.Video` object.
        # If you're invoking the CLI, construct the Video upstream and call the API directly.
        analyze_video(  # type: ignore[arg-type]
            args.video,
            outdir=args.outdir,
            expect_step=args.expect_step,
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
