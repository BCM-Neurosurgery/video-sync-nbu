#!/usr/bin/env python3
"""
A/V Sync Driver — multi-segment, multi-camera pipeline with clean, industry-grade logging
========================================================================================

What this is
------------
A command-line orchestrator that runs your end-to-end audio↔video sync workflow:
decode serials → gapfill → filter → collect anchors → clip window → build pad plan →
apply plan to program channels → re-collect anchors → final sync.

You can target **all segments/cameras** or a subset via flags.

Quick start
-----------
python sync_driver.py \
  --audio-dir /path/to/input/audio \
  --video-dir /path/to/input/video \
  --out-dir   /path/to/output \
  --site jamail \
  --log-level INFO

Speed-ups / resume
------------------
- Use **--skip-decode** to reuse an existing decoded CSV at:
  <out_dir>/<segment_id>/audio_decoded/raw.csv
  This skips the slow serial decoding and goes straight to gapfill→filter→… stages.

Input folder layout (discover expects this)
-------------------------------------------
input_dir structure
- audio
    - TRBD002_08062025-01.mp3
    - TRBD002_08062025-03.mp3
- video
    - <SEGMENT_ID>_<TIMESTAMP>.<CAM_SERIAL1>.mp4
    - <SEGMENT_ID>_<TIMESTAMP>.json
    ...
    - <SEGMENT_ID>_<TIMESTAMP>.<CAM_SERIAL2>.mp4
    - <SEGMENT_ID>_<TIMESTAMP>.json
    ...

Output folder layout (what this tool writes)
--------------------------------------------
output_dir structure
- parent_out
    - <segment_id> (e.g. TRBD002_20250806_104707)
        - audio_decoded   (shared across cameras)
            - raw.csv
            - raw.txt
            - raw-gapfilled.csv
            - raw-gapfilled.txt
            - raw-gapfilled-filtered.csv
            - raw-gapfilled-filtered.txt

        - <camera_serial1> (e.g. 23512909)
            - work (intermediate artifacts)
                - gapfilled-filtered-anchors.json             (anchors from filtered CSV)
                - gapfilled-filtered-clipped.csv              (CSV clipped to video window)
                - gapfilled-filtered-clipped.txt
                - gapfilled-filtered-clipped-editplan.json
                - gapfilled-filtered-clipped-padded.csv
                - gapfilled-filtered-clipped-padded.txt
                - gapfilled-filtered-padded-anchors.json      (anchors after padding)
            - audio_padded
                - TRBD002_08062025-padded-01.mp3
                - TRBD002_08062025-padded-03.mp3
            - audio_clips
            - synced  (final synced videos)
            - sync.log  ← per-camera rotating log (5MB x 3 backups), stamped with [seg/cam]

        - <camera_serial2>
        ...

Logging (clean + consistent)
----------------------------
- Console        : one handler, unified format → "[LEVEL] [seg/cam] message"
- Run log        : <out_dir>/sync-run.log (rotating, 5MB x 3), format:
                   "%(asctime)s %(levelname)s [%(seg)s/%(cam)s] %(name)s: %(message)s"
- Per-camera log : <out_dir>/<segment>/<camera>/sync.log (rotating, 5MB x 3), same format but
                   forcibly stamped with the correct [seg/cam] (even for logs from other modules)

Return codes
------------
0 : All segments processed with no camera-level failures
2 : Audio group discovery failed
3 : Target building / validation error
4 : At least one (segment, camera) had failures
5 : Invalid site argument

Notes
-----
- Padded audio file names follow the rule: original stem ending with 01/02/03 becomes
  "<stem>-padded-01/02/03.mp3" (e.g., TRBD002_08062025-03 → TRBD002_08062025-padded-03).
- Site-specific decoding behavior is implemented in your parser modules.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
from logging.handlers import RotatingFileHandler
from typing import Iterable

from scripts.parsers.wavfileparser import decode_to_raw
from scripts.analysis.csv_serial_analysis import analyze_csv_serials
from scripts.analysis.anchor_analysis import analyze_anchors_file
from scripts.fix.audiogapfiller import gapfill_csv_file
from scripts.fix.audiofilter import filter_audio_file
from scripts.align.collect_anchors import save_anchors_for_camera
from scripts.clip.audiocsvclipper import clip_with_anchors
from scripts.pad.audiopadder import AudioPadder
from scripts.pad.audioplanapplier import AudioPlanApplier
from scripts.cli import sync_one_segment
from scripts.discover import AudioDiscoverer
from scripts.models import AudioGroup
from scripts.errors import (
    AudioGroupDiscoverError,
    TargetBuildError,
    AudioDecodingError,
    SyncError,
    SerialAnalysisError,
    GapFillError,
    FilteredError,
    AnchorError,
    ClipError,
    AudioPaddingError,
    AudioPlanError,
)

# --------------------------------------------------------------------------------------
# Logging (industry-grade, no duplicates, consistent formatting)
# --------------------------------------------------------------------------------------


class _ContextFilter(logging.Filter):
    """Ensure %(seg)s and %(cam)s exist so all formatters work cleanly."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "seg"):
            record.seg = "-"
        if not hasattr(record, "cam"):
            record.cam = "-"
        return True


class _SegCamStampFilter(logging.Filter):
    """
    On a per-camera file handler, stamp seg/cam onto *every* record that flows through,
    so lines from other modules (that don't know seg/cam) still render as [seg/cam].
    """

    def __init__(self, seg: str, cam: str) -> None:
        super().__init__()
        self._seg, self._cam = seg, cam

    def filter(self, record: logging.LogRecord) -> bool:
        record.seg = getattr(record, "seg", self._seg) or self._seg
        record.cam = getattr(record, "cam", self._cam) or self._cam
        return True


def _attach_rotating_file_handler(
    log_owner: logging.Logger,
    file_path: Path,
    level: int,
    fmt: str,
) -> RotatingFileHandler:
    """
    Attach a RotatingFileHandler to `log_owner` (skip if same path already attached).
    Returns the handler so callers can remove/close it later if desired.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    for h in log_owner.handlers:
        if isinstance(h, RotatingFileHandler):
            try:
                if (
                    Path(getattr(h, "baseFilename", "")).resolve()
                    == file_path.resolve()
                ):
                    return h  # already attached
            except Exception:
                continue

    fh = RotatingFileHandler(
        filename=str(file_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
        delay=True,
    )
    fh.setLevel(level)
    fh.addFilter(_ContextFilter())
    fh.setFormatter(logging.Formatter(fmt))
    log_owner.addHandler(fh)
    return fh


def configure_logging(out_dir: Path, level: str = "INFO") -> logging.Logger:
    """
    Global logging: one console, one run log. Clears any prior root handlers so you
    don't get duplicate `INFO:logger:` lines on the terminal.
    """
    root = logging.getLogger()
    # Clear existing handlers (including default basicConfig ones)
    for h in root.handlers[:]:
        root.removeHandler(h)

    lvl = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(lvl)

    # Console (stderr)
    console = logging.StreamHandler()
    console.setLevel(lvl)
    console.addFilter(_ContextFilter())
    console.setFormatter(
        logging.Formatter("[%(levelname)s] [%(seg)s/%(cam)s] %(message)s")
    )
    root.addHandler(console)

    # Run file (aggregated)
    _attach_rotating_file_handler(
        root,
        Path(out_dir) / "sync-run.log",
        lvl,
        "%(asctime)s %(levelname)s [%(seg)s/%(cam)s] %(name)s: %(message)s",
    )

    # Return the app logger for convenience
    return logging.getLogger("sync")


def _attach_cam_logger(seg_id: str, cam_serial: str, cam_dir: Path, level: int):
    """
    Create a per-camera file handler on the *root* logger + a LoggerAdapter that
    injects seg/cam context. Stamps seg/cam on all records flowing to this handler.
    Returns (adapter, handler) so caller can remove/close the handler after processing.
    """
    fmt = "%(asctime)s %(levelname)s [%(seg)s/%(cam)s] %(message)s"
    fh = _attach_rotating_file_handler(
        logging.getLogger(), cam_dir / "sync.log", level, fmt
    )
    fh.addFilter(
        _SegCamStampFilter(seg_id, cam_serial)
    )  # force proper [seg/cam] in the per-camera file
    adapter = logging.LoggerAdapter(
        logging.getLogger("sync"), extra={"seg": seg_id, "cam": cam_serial}
    )
    return adapter, fh


# Application logger (no handlers here; handlers live on root)
logger = logging.getLogger("sync")


# --------------------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------------------

SITE_CHOICES = ("jamail", "nbu_lounge", "nbu_sleep")


def _name(p) -> str:
    """Return just the basename for any path-like object."""
    try:
        return Path(p).name
    except Exception:
        return str(p)


def list_segments(video_dir: Path) -> list[str]:
    """
    Discover segment IDs from *.json basenames in video_dir.
    Example: TRBD002_20250806_104707.json → 'TRBD002_20250806_104707'
    """
    return sorted({p.stem for p in Path(video_dir).glob("*.json")})


def list_cameras_for_segment(video_dir: Path, segment_id: str) -> list[str]:
    """
    Discover camera serials for a given segment by scanning <segment>.*.mp4 files.
    Example: TRBD002_20250806_104707.23512909.mp4 → '23512909'
    """
    cams: list[str] = []
    for mp4 in Path(video_dir).glob(f"{segment_id}.*.mp4"):
        cam = mp4.stem.split(".")[-1]
        if cam not in cams:
            cams.append(cam)
    cams.sort()
    return cams


def build_targets(
    video_dir: Path,
    segments: list[str] | None,
    cameras: list[str] | None,
) -> dict[str, list[str]]:
    """
    Build a {segment_id: [cam_serial, ...]} map following selection rules:
      - segments=None  → all segments
      - cameras=None   → all cameras per segment
      - cameras given  → restrict each segment to these cams (skip cams not present)

    Raises
    ------
    TargetBuildError if video_dir is invalid, requested segments are missing, or
    no targets remain after filtering.
    """
    vd = Path(video_dir)
    if not vd.exists() or not vd.is_dir():
        raise TargetBuildError(f"video_dir does not exist or is not a directory: {vd}")

    if segments:
        missing = [s for s in segments if not (vd / f"{s}.json").exists()]
        if missing:
            raise TargetBuildError(
                f"Missing segment JSON for: {', '.join(missing)} (in {vd})"
            )

    segs = segments or list_segments(vd)
    targets: dict[str, list[str]] = {}

    for seg in segs:
        all_cams = set(list_cameras_for_segment(vd, seg))
        if not all_cams:
            logger.warning("[%s] no cameras found", seg)
            continue

        if cameras:
            selected = [c for c in cameras if c in all_cams]
            if not selected:
                logger.warning(
                    "[%s] none of the requested cameras exist: %s",
                    seg,
                    ",".join(cameras),
                )
                continue
        else:
            selected = sorted(all_cams)

        targets[seg] = selected

    if not targets:
        raise TargetBuildError(
            "No targets found. Check --video-dir / --segment / --camera inputs."
        )
    return targets


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def run_pipeline(
    audio_dir: Path,
    video_dir: Path,
    out_dir: Path,
    site: str,
    segments: list[str] | None,
    cameras: list[str] | None,
    log_level: str = "INFO",
    skip_decode: bool = False,
) -> int:
    """
    Orchestrate discovery + per-(segment,camera) processing.
    Returns 0 on success, nonzero if any failures occurred.
    """
    # Threshold (handlers already configured globally)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if site not in SITE_CHOICES:
        logger.error(
            "Invalid --site '%s'. Choose from: %s", site, ", ".join(SITE_CHOICES)
        )
        return 5

    # Discover audio group once (shared across segments/cams) — still needed for A1/A2 paths
    try:
        ad = AudioDiscoverer(audio_dir=audio_dir, log=logger)
        ag = ad.discover()
        logger.info("Audio(s) discovered")
    except AudioGroupDiscoverError as e:
        logger.error("Audio group discovery failed: %s", e)
        return 2

    try:
        targets = build_targets(video_dir, segments, cameras)
    except TargetBuildError as e:
        logger.error("%s", e)
        return 3

    failures = 0
    for seg_id, cam_list in targets.items():
        summary = process_segment(
            video_in=video_dir,
            audiogroup=ag,
            seg_id=seg_id,
            site=site,
            parent_out=out_dir,
            cam_serials=cam_list,
            skip_decode=skip_decode,
        )
        if summary["fail"]:
            failures += 1
            logger.warning(
                "[%s] done with failures: %s", seg_id, ", ".join(summary["fail"])
            )
        else:
            logger.info("[%s] done!", seg_id)

    return 0 if failures == 0 else 4


def process_segment(
    video_in: str,
    audiogroup: AudioGroup,
    seg_id: str,
    site: str,
    parent_out: Path,
    cam_serials: Iterable[str],
    skip_decode: bool = False,
) -> dict:
    """
    Returns a summary dict with successes/failures. Never raises.
    """
    segment_out = parent_out / seg_id
    audio_decoded_dir = segment_out / "audio_decoded"
    summary = {"segment": seg_id, "ok": [], "fail": []}

    # ---- Stage: decode + analyze raw ----
    try:
        if skip_decode:
            decoded_raw_csv = audio_decoded_dir / "raw.csv"
            if not decoded_raw_csv.exists():
                logger.error(
                    "[%s] --skip-decode set but missing %s", seg_id, decoded_raw_csv
                )
                summary["fail"].append("decode-missing")
                return summary
            logger.info("[%s] skip decode: using %s", seg_id, _name(decoded_raw_csv))
            try:
                _, raw_txt = analyze_csv_serials(path=decoded_raw_csv)
                logger.info("[%s] raw.txt → %s", seg_id, _name(raw_txt))
            except SerialAnalysisError as e:
                logger.error("[%s] raw analysis failed: %s", seg_id, e)
                summary["fail"].append("raw-analysis")
        else:
            logger.info("[%s] decoding serial…", seg_id)
            decoded_raw_csv, _, _, _ = decode_to_raw(
                audiogroup.serial_audio.path, audio_decoded_dir, site=site
            )
            logger.info("[%s] raw.csv → %s", seg_id, _name(decoded_raw_csv))
            try:
                _, raw_txt = analyze_csv_serials(path=decoded_raw_csv)
                logger.info("[%s] raw.txt → %s", seg_id, _name(raw_txt))
            except SerialAnalysisError as e:
                logger.error("[%s] raw analysis failed: %s", seg_id, e)
                summary["fail"].append("raw-analysis")
    except AudioDecodingError as e:
        logger.error("[%s] decode failed: %s", seg_id, e)
        summary["fail"].append("decode")
        return summary

    # ---- Stage: gapfill + analyze ----
    try:
        gapfilled_csv = gapfill_csv_file(input_csv=decoded_raw_csv)
        logger.info("[%s] gapfilled.csv → %s", seg_id, _name(gapfilled_csv))
        try:
            _, gapfilled_txt = analyze_csv_serials(path=gapfilled_csv)
            logger.info("[%s] gapfilled.txt → %s", seg_id, _name(gapfilled_txt))
        except SerialAnalysisError as e:
            logger.error("[%s] gapfilled analysis failed: %s", seg_id, e)
            summary["fail"].append("gapfilled-analysis")
    except GapFillError as e:
        logger.error("[%s] gapfill failed: %s", seg_id, e)
        summary["fail"].append("gapfill")
        return summary

    # ---- Stage: filter + analyze ----
    try:
        filtered_csv = filter_audio_file(input_csv=gapfilled_csv)
        logger.info("[%s] filtered.csv → %s", seg_id, _name(filtered_csv))
        try:
            _, filtered_txt = analyze_csv_serials(path=filtered_csv)
            logger.info("[%s] filtered.txt → %s", seg_id, _name(filtered_txt))
        except SerialAnalysisError as e:
            logger.error("[%s] filtered analysis failed: %s", seg_id, e)
            summary["fail"].append("filtered-analysis")
    except FilteredError as e:
        logger.error("[%s] filter failed: %s", seg_id, e)
        summary["fail"].append("filter")
        return summary

    # ---- Per camera workflow ----
    for cam in cam_serials:
        cam_out = segment_out / cam
        (cam_out / "work").mkdir(parents=True, exist_ok=True)
        (cam_out / "audio_padded").mkdir(parents=True, exist_ok=True)
        (cam_out / "audio_clips").mkdir(parents=True, exist_ok=True)
        (cam_out / "synced").mkdir(parents=True, exist_ok=True)

        # Attach per-camera rotating file log + adapter with seg/cam context
        log, cam_handler = _attach_cam_logger(
            seg_id, cam, cam_out, logging.getLogger().level
        )

        filtered_anchors = cam_out / "work" / "gapfilled-filtered-anchors.json"
        padded_anchors = cam_out / "work" / "gapfilled-filtered-padded-anchors.json"

        try:
            # Anchors from filtered CSV
            try:
                save_anchors_for_camera(
                    serial_csv=filtered_csv,
                    video_dir=video_in,
                    segment_id=seg_id,
                    cam_serial=cam,
                    out_json=filtered_anchors,
                )
                log.info("anchors.json → %s", _name(filtered_anchors))
                try:
                    analyze_anchors_file(anchors_json=filtered_anchors)
                    log.info("anchors analyzed")
                except AnchorError as e:
                    log.error("anchors analyze failed: %s", e)
                    summary["fail"].append(f"{cam}:anchors-analyze")
            except AnchorError as e:
                log.error("anchors save failed: %s", e)
                summary["fail"].append(f"{cam}:anchors-save")
                continue

            # Clip CSV to video window
            try:
                clipped_csv = clip_with_anchors(
                    input_csv=filtered_csv,
                    anchors_json=filtered_anchors,
                    output_csv=cam_out / "work" / "gapfilled-filtered-clipped.csv",
                )
                log.info("clipped.csv → %s", _name(clipped_csv))
                try:
                    _, clipped_txt = analyze_csv_serials(path=clipped_csv)
                    log.info("clipped.txt → %s", _name(clipped_txt))
                except SerialAnalysisError as e:
                    log.error("clipped analysis failed: %s", e)
                    summary["fail"].append(f"{cam}:clipped-analysis")
            except ClipError as e:
                log.error("clip failed: %s", e)
                summary["fail"].append(f"{cam}:clip")
                continue

            # Pad (build plan + fixed CSV)
            try:
                apadder = AudioPadder(
                    csv_path=clipped_csv,
                    include_synthetic=True,
                    gap_policy="local",
                    sample_rate=44100,
                )
                _, _, padded_csv, padplan = apadder.run()
                log.info("padded.csv → %s", _name(padded_csv))
                log.info("padplan.json → %s", _name(padplan))
                try:
                    _, padded_txt = analyze_csv_serials(path=padded_csv)
                    log.info("padded.txt → %s", _name(padded_txt))
                except SerialAnalysisError as e:
                    log.error("padded analysis failed: %s", e)
                    summary["fail"].append(f"{cam}:padded-analysis")
            except AudioPaddingError as e:
                log.error("padding failed: %s", e)
                summary["fail"].append(f"{cam}:padding")
                continue

            # Apply plan to each program channel
            for ch, audio in audiogroup.audios.items():
                try:
                    applier = AudioPlanApplier(
                        audio_path=audio.path,
                        plan_path=padplan,
                        out_dir=cam_out / "audio_padded",
                    )
                    out = applier.apply()
                    log.info("plan→ch%02d → %s", ch, _name(out))
                except AudioPlanError as e:
                    log.error("plan apply ch%02d failed: %s", ch, e)
                    summary["fail"].append(f"{cam}:plan-ch{ch}")

            # Anchors from padded CSV
            try:
                save_anchors_for_camera(
                    serial_csv=padded_csv,
                    video_dir=video_in,
                    segment_id=seg_id,
                    cam_serial=cam,
                    out_json=padded_anchors,
                )
                log.info("padded-anchors.json → %s", _name(padded_anchors))
                try:
                    analyze_anchors_file(anchors_json=padded_anchors)
                    log.info("padded-anchors analyzed")
                except AnchorError as e:
                    log.error("padded-anchors analyze failed: %s", e)
                    summary["fail"].append(f"{cam}:padded-anchors-analyze")
            except AnchorError as e:
                log.error("padded-anchors save failed: %s", e)
                summary["fail"].append(f"{cam}:padded-anchors-save")
                continue

            # Final sync
            try:
                sync_one_segment(
                    audio_dir=cam_out / "audio_padded",
                    video_dir=video_in,
                    segment_id=seg_id,
                    cam_serial=cam,
                    anchors_json=padded_anchors,
                    out_audio_dir=cam_out / "audio_clips",
                    out_video_dir=cam_out / "synced",
                    serial_channel=3,
                )
                log.info("synced ✔")
                summary["ok"].append(cam)
            except SyncError as e:
                log.error("sync failed: %s", e)
                summary["fail"].append(f"{cam}:sync")

        finally:
            # Detach per-camera file handler to avoid handler accumulation
            try:
                logging.getLogger().removeHandler(cam_handler)
                cam_handler.close()
            except Exception:
                pass

    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A/V sync driver: process all segments, one segment, or (segment, camera) subset."
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        type=Path,
        help="Directory containing TRBD...-01.mp3 etc.",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        type=Path,
        help="Directory containing <SEGMENT>.json and <SEGMENT>.<CAM>.mp4 files",
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="Parent output directory"
    )
    parser.add_argument(
        "--site",
        default="jamail",
        choices=SITE_CHOICES,
        help="Site label (jamail | nbu_lounge | nbu_sleep)",
    )
    parser.add_argument(
        "-s",
        "--segment",
        dest="segments",
        action="append",
        help="Segment ID to process (repeatable). If omitted, process ALL segments.",
    )
    parser.add_argument(
        "-c",
        "--camera",
        dest="cameras",
        action="append",
        help="Camera serial to process (repeatable). If omitted, process ALL cams per segment.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Reuse existing <out>/<segment>/audio_decoded/raw.csv and skip audio serial decoding.",
    )
    args = parser.parse_args()

    # Configure logging globally once (console + run-file)
    configure_logging(args.out_dir, args.log_level)

    rc = run_pipeline(
        audio_dir=args.audio_dir,
        video_dir=args.video_dir,
        out_dir=args.out_dir,
        site=args.site,
        segments=args.segments,
        cameras=args.cameras,
        log_level=args.log_level,
        skip_decode=args.skip_decode,
    )
    raise SystemExit(rc)
