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
- Per-camera log : <out_dir>/<segment>/<camera>/sync.log (rotating, 5MB x 3), same format;
                   stamped with correct [seg/cam] even for logs from other modules.

Return codes
------------
0 : All segments processed with no camera-level failures
2 : Audio group discovery failed
3 : Target building / validation error
4 : At least one (segment, camera) had failures
5 : Invalid site argument
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
from typing import Iterable

from scripts.parsers.wavfileparser import decode_to_raw
from scripts.analysis.csv_serial_analysis import analyze_csv_serials
from scripts.analysis.anchor_analysis import analyze_anchors_file
from scripts.analysis.video_analysis import analyze_and_write
from scripts.analysis.video_frameid_analysis import analyze_video_frameids
from scripts.fix.audiogapfiller import gapfill_csv_file
from scripts.fix.audiofilter import filter_audio_file
from scripts.align.collect_anchors import save_anchors_for_camera
from scripts.clip.audiocsvclipper import clip_with_anchors
from scripts.pad.audiopadder import AudioPadder
from scripts.pad.audioplanapplier import AudioPlanApplier
from scripts.align.sync import sync_one_segment
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
    VideoAnalysisError,
    VideoFrameIDAnalysisError,
    AnchorError,
    ClipError,
    AudioPaddingError,
    AudioPlanError,
)

# Shared logging utilities
from scripts.log.logutils import (
    configure_logging,  # console + run log at <out_dir>/sync-run.log
    attach_cam_logger,  # per-camera rotating file handler + adapter
    log_context,  # context manager to stamp [seg/cam]
)

# Application logger (handlers live on root, configured by configure_logging)
logger = logging.getLogger("cli")

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
    """Discover segment IDs from *.json basenames in video_dir."""
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
    Build a {segment_id: [cam_serial, ...]} map following selection rules.
    Raises TargetBuildError on invalid inputs or empty results.
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
            logger.warning("no cameras found for %s", seg)
            continue

        if cameras:
            selected = [c for c in cameras if c in all_cams]
            if not selected:
                logger.warning(
                    "segment %s: none of the requested cameras exist: %s",
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

    # Discover audio group once (shared across segments/cams)
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
                "segment %s: done with failures: %s", seg_id, ", ".join(summary["fail"])
            )
        else:
            logger.info("segment %s: done!", seg_id)

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
    """Process one segment across one or more cameras. Returns a summary dict."""
    segment_out = parent_out / seg_id
    audio_decoded_dir = segment_out / "audio_decoded"
    summary = {"segment": seg_id, "ok": [], "fail": []}

    # ---- Stage: decode + analyze raw (segment-scoped context) ----
    with log_context(seg=seg_id, cam="-"):
        try:
            if skip_decode:
                decoded_raw_csv = audio_decoded_dir / "raw.csv"
                if not decoded_raw_csv.exists():
                    logger.error("--skip-decode set but missing %s", decoded_raw_csv)
                    summary["fail"].append("decode-missing")
                    return summary
                logger.info("Skip decode: using %s", _name(decoded_raw_csv))
                try:
                    _, raw_txt = analyze_csv_serials(path=decoded_raw_csv)
                    logger.info("Analyzed raw.csv → %s", _name(raw_txt))
                except SerialAnalysisError as e:
                    logger.error("Raw analysis failed: %s", e)
                    summary["fail"].append("raw-analysis")
            else:
                logger.info("Decoding serial…")
                decoded_raw_csv, _, _, _ = decode_to_raw(
                    audiogroup.serial_audio.path, audio_decoded_dir, site=site
                )
                logger.info("Decoded audio serial → %s", _name(decoded_raw_csv))
                try:
                    _, raw_txt = analyze_csv_serials(path=decoded_raw_csv)
                    logger.info(
                        "Analyzed %s → %s", _name(decoded_raw_csv), _name(raw_txt)
                    )
                except SerialAnalysisError as e:
                    logger.error("Raw analysis failed: %s", e)
                    summary["fail"].append("raw-analysis")
        except AudioDecodingError as e:
            logger.error("Decode failed: %s", e)
            summary["fail"].append("decode")
            return summary

        # ---- Stage: gapfill + analyze ----
        try:
            gapfilled_csv = gapfill_csv_file(input_csv=decoded_raw_csv)
            logger.info(
                "Gap-filled %s → %s", _name(decoded_raw_csv), _name(gapfilled_csv)
            )
            try:
                _, gapfilled_txt = analyze_csv_serials(path=gapfilled_csv)
                logger.info(
                    "Analyzed %s → %s", _name(gapfilled_csv), _name(gapfilled_txt)
                )
            except SerialAnalysisError as e:
                logger.error("Gap-filled analysis failed: %s", e)
                summary["fail"].append("gapfilled-analysis")
        except GapFillError as e:
            logger.error("Gap-fill failed: %s", e)
            summary["fail"].append("gapfill")
            return summary

        # ---- Stage: filter + analyze ----
        try:
            filtered_csv = filter_audio_file(input_csv=gapfilled_csv)
            logger.info("Filtered %s → %s", _name(gapfilled_csv), _name(filtered_csv))
            try:
                _, filtered_txt = analyze_csv_serials(path=filtered_csv)
                logger.info(
                    "Analyzed %s → %s", _name(filtered_csv), _name(filtered_txt)
                )
            except SerialAnalysisError as e:
                logger.error("Filtered analysis failed: %s", e)
                summary["fail"].append("filtered-analysis")
        except FilteredError as e:
            logger.error("Filter failed: %s", e)
            summary["fail"].append("filter")
            return summary

    # ---- Per camera workflow (camera-scoped context) ----
    for cam in cam_serials:
        cam_out = segment_out / cam
        (cam_out / "work").mkdir(parents=True, exist_ok=True)
        (cam_out / "audio_padded").mkdir(parents=True, exist_ok=True)
        (cam_out / "audio_clips").mkdir(parents=True, exist_ok=True)
        (cam_out / "synced").mkdir(parents=True, exist_ok=True)

        # Per-camera file logger (tag all lines with [seg/cam])
        clog, cam_handler = attach_cam_logger(
            seg_id=seg_id,
            cam_serial=cam,
            cam_dir=cam_out,
            level=logging.getLogger().level,
            logger_name="cli",
        )

        with log_context(seg=seg_id, cam=cam):
            filtered_anchors = cam_out / "work" / "gapfilled-filtered-anchors.json"
            padded_anchors = cam_out / "work" / "gapfilled-filtered-padded-anchors.json"

            try:
                # Video analysis
                try:
                    analyze_and_write(
                        video_path=video_in / f"{seg_id}.{cam}.mp4",
                        outdir=cam_out / "work",
                    )
                    clog.info("Video analysis completed")
                except VideoAnalysisError as e:
                    clog.error("Video analysis failed: %s", e)
                    summary["fail"].append(f"{cam}:video-analysis")

                # Video frame id analysis
                try:
                    analyze_video_frameids(
                        video=video_in / f"{seg_id}.{cam}.mp4",
                        outdir=cam_out / "work",
                    )
                    clog.info("Video frame id analysis completed")
                except VideoFrameIDAnalysisError as e:
                    clog.error("Video frame id analysis failed: %s", e)
                    summary["fail"].append(f"{cam}:video-frame-id-analysis")

                # Anchors from filtered CSV
                try:
                    save_anchors_for_camera(
                        serial_csv=filtered_csv,
                        video_dir=video_in,
                        segment_id=seg_id,
                        cam_serial=cam,
                        out_json=filtered_anchors,
                    )
                    clog.info("Saved %s", _name(filtered_anchors))
                    try:
                        analyze_anchors_file(anchors_json=filtered_anchors)
                        clog.info("anchors analyzed")
                    except AnchorError as e:
                        clog.error("anchors analyze failed: %s", e)
                        summary["fail"].append(f"{cam}:anchors-analyze")
                except AnchorError as e:
                    clog.error("anchors save failed: %s", e)
                    summary["fail"].append(f"{cam}:anchors-save")
                    # continue to next cam
                    continue

                # Clip CSV to video window
                try:
                    clipped_csv = clip_with_anchors(
                        input_csv=filtered_csv,
                        anchors_json=filtered_anchors,
                        output_csv=cam_out / "work" / "gapfilled-filtered-clipped.csv",
                    )
                    clog.info("Clipped %s → %s", _name(clipped_csv), _name(clipped_csv))
                    try:
                        _, clipped_txt = analyze_csv_serials(path=clipped_csv)
                        clog.info(
                            "Analyzed %s → %s", _name(clipped_csv), _name(clipped_txt)
                        )
                    except SerialAnalysisError as e:
                        clog.error("clipped analysis failed: %s", e)
                        summary["fail"].append(f"{cam}:clipped-analysis")
                except ClipError as e:
                    clog.error("clip failed: %s", e)
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
                    clog.info("Padded %s → %s", _name(clipped_csv), _name(padded_csv))
                    clog.info("Padding plan saved to %s", _name(padplan))
                    try:
                        _, padded_txt = analyze_csv_serials(path=padded_csv)
                        clog.info(
                            "Analyzed %s → %s", _name(padded_csv), _name(padded_txt)
                        )
                    except SerialAnalysisError as e:
                        clog.error("Padded analysis failed: %s", e)
                        summary["fail"].append(f"{cam}:padded-analysis")
                except AudioPaddingError as e:
                    clog.error("Padding failed: %s", e)
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
                        clog.info("plan→ch%02d → %s", ch, _name(out))
                    except AudioPlanError as e:
                        clog.error("plan apply ch%02d failed: %s", ch, e)
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
                    clog.info("Saved %s", _name(padded_anchors))
                    try:
                        analyze_anchors_file(anchors_json=padded_anchors)
                        clog.info("padded-anchors analyzed")
                    except AnchorError as e:
                        clog.error("padded-anchors analyze failed: %s", e)
                        summary["fail"].append(f"{cam}:padded-anchors-analyze")
                except AnchorError as e:
                    clog.error("padded-anchors save failed: %s", e)
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
                    clog.info("synced ✔")
                    summary["ok"].append(cam)
                except SyncError as e:
                    clog.error("sync failed: %s", e)
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
