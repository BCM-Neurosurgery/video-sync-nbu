"""
input_dir structure
-audio
    - TRBD002_08062025-01.mp3
    - TRBD002_08062025-03.mp3
-video
    - <SEGMENT_ID>_<TIMESTAMP>.<CAM_SERIAL1>.mp4
    - <SEGMENT_ID>_<TIMESTAMP>.json
    ...
    - <SEGMENT_ID>_<TIMESTAMP>.<CAM_SERIAL2>.mp4
    - <SEGMENT_ID>_<TIMESTAMP>.json
    ...

output_dir structure
-parent_out
    - segment_id (e.g. TRBD002_20250806_104707)
        - audio_decoded (this will be shared resource for this segment-id)
            - raw.csv
            - raw.txt
            - raw-gapfilled.csv
            - raw-gapfilled.txt
            - raw-gapfilled-filtered.csv
            - raw-gapfilled-filtered.txt

        - camera_serial1 (e.g. 23512909)
            - work (intermediate work files)
                - gapfilled-filtered-anchors.json (anchors)
                - raw-gapfilled-filtered-clipped.csv
                - raw-gapfilled-filtered-clipped.txt
                - raw-gapfilled-filtered-clipped-editplan.json
                - raw-gapfilled-filtered-clipped-padded.csv
                - raw-gapfilled-filtered-clipped-padded.txt

            - audio_padded
                - TRBD002_08062025-padded-01.mp3
                - TRBD002_08062025-padded-03.mp3

            - audio_clips
            - synced (the synced videos)

        - camera_serial2 (e.g. 24253449)
        ...
"""

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
from pathlib import Path
import logging
from typing import Iterable
from scripts.errors import (
    AudioGroupDiscoverError,
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
import argparse

logger = logging.getLogger("sync")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _name(p) -> str:
    """Return just the basename for any path-like object."""
    try:
        return Path(p).name
    except Exception:
        return str(p)


def list_segments(video_dir: Path) -> list[str]:
    """
    Discover segment IDs from *.json basenames in video_dir.
    Example file: TRBD002_20250806_104707.json → segment_id == TRBD002_20250806_104707
    """
    return sorted({p.stem for p in Path(video_dir).glob("*.json")})


def list_cameras_for_segment(video_dir: Path, segment_id: str) -> list[str]:
    """
    Discover camera serials for a given segment by scanning <segment>.*.mp4 files.
    Example: TRBD002_20250806_104707.23512909.mp4 → '23512909'
    """
    cams = []
    for mp4 in Path(video_dir).glob(f"{segment_id}.*.mp4"):
        # extract the substring between last '.' and '.mp4'
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
    """
    # choose segments
    segs = segments or list_segments(video_dir)
    targets: dict[str, list[str]] = {}

    for seg in segs:
        all_cams = set(list_cameras_for_segment(video_dir, seg))
        if not all_cams:
            logger.warning("[%s] no cameras found", seg)
            continue

        if cameras:
            # keep only cams that actually exist for this segment
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
        logger.error(
            "No targets found. Check --video-dir / --segment / --camera inputs."
        )
    return targets


def run_pipeline(
    audio_dir: Path,
    video_dir: Path,
    out_dir: Path,
    site: str,
    segments: list[str] | None,
    cameras: list[str] | None,
    log_level: str = "INFO",
) -> int:
    """
    Orchestrate discovery + per-(segment,camera) processing.
    Returns 0 on success, nonzero if any failures occurred.
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Discover audio group once (shared across segments/cams)
    try:
        ag = AudioDiscoverer(audio_dir=audio_dir, log=logger)
        logger.info("Audio(s) discovered")
    except AudioGroupDiscoverError as e:
        logger.error("Audio group discovery failed: %s", e)
        return 2

    targets = build_targets(video_dir, segments, cameras)
    if not targets:
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
) -> dict:
    """
    Returns a summary dict with successes/failures. Never raises.
    """
    segment_out = parent_out / seg_id
    audio_decoded_dir = segment_out / "audio_decoded"
    summary = {"segment": seg_id, "ok": [], "fail": []}

    # ---- Stage: decode + analyze raw ----
    try:
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
        tag = f"{seg_id}·{cam}"
        cam_out = segment_out / cam
        (cam_out / "work").mkdir(parents=True, exist_ok=True)
        (cam_out / "audio_padded").mkdir(parents=True, exist_ok=True)
        (cam_out / "audio_clips").mkdir(parents=True, exist_ok=True)
        (cam_out / "synced").mkdir(parents=True, exist_ok=True)

        filtered_anchors = cam_out / "work" / "gapfilled-filtered-anchors.json"
        padded_anchors = cam_out / "work" / "gapfilled-filtered-padded-anchors.json"

        # Anchors from filtered CSV
        try:
            save_anchors_for_camera(
                serial_csv=filtered_csv,
                video_dir=video_in,
                segment_id=seg_id,
                cam_serial=cam,
                out_json=filtered_anchors,
            )
            logger.info("[%s] anchors.json → %s", tag, _name(filtered_anchors))
            try:
                analyze_anchors_file(anchors_json=filtered_anchors)
                logger.info("[%s] anchors analyzed", tag)
            except AnchorError as e:
                logger.error("[%s] anchors analyze failed: %s", tag, e)
                summary["fail"].append(f"{cam}:anchors-analyze")
        except AnchorError as e:
            logger.error("[%s] anchors save failed: %s", tag, e)
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
            logger.info("[%s] clipped.csv → %s", tag, _name(clipped_csv))
            try:
                _, clipped_txt = analyze_csv_serials(path=clipped_csv)
                logger.info("[%s] clipped.txt → %s", tag, _name(clipped_txt))
            except SerialAnalysisError as e:
                logger.error("[%s] clipped analysis failed: %s", tag, e)
                summary["fail"].append(f"{cam}:clipped-analysis")
        except ClipError as e:
            logger.error("[%s] clip failed: %s", tag, e)
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
            logger.info("[%s] padded.csv → %s", tag, _name(padded_csv))
            logger.info("[%s] padplan.json → %s", tag, _name(padplan))
            try:
                _, padded_txt = analyze_csv_serials(path=padded_csv)
                logger.info("[%s] padded.txt → %s", tag, _name(padded_txt))
            except SerialAnalysisError as e:
                logger.error("[%s] padded analysis failed: %s", tag, e)
                summary["fail"].append(f"{cam}:padded-analysis")
        except AudioPaddingError as e:
            logger.error("[%s] padding failed: %s", tag, e)
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
                logger.info("[%s] plan→ch%02d → %s", tag, ch, _name(out))
            except AudioPlanError as e:
                logger.error("[%s] plan apply ch%02d failed: %s", tag, ch, e)
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
            logger.info("[%s] padded-anchors.json → %s", tag, _name(padded_anchors))
            try:
                analyze_anchors_file(anchors_json=padded_anchors)
                logger.info("[%s] padded-anchors analyzed", tag)
            except AnchorError as e:
                logger.error("[%s] padded-anchors analyze failed: %s", tag, e)
                summary["fail"].append(f"{cam}:padded-anchors-analyze")
        except AnchorError as e:
            logger.error("[%s] padded-anchors save failed: %s", tag, e)
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
            logger.info("[%s] synced ✔", tag)
            summary["ok"].append(cam)
        except SyncError as e:
            logger.error("[%s] sync failed: %s", tag, e)
            summary["fail"].append(f"{cam}:sync")

    return summary


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
        "--site", default="jamail", help="Site label for decoding (default: jamail)"
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
    args = parser.parse_args()

    rc = run_pipeline(
        audio_dir=args.audio_dir,
        video_dir=args.video_dir,
        out_dir=args.out_dir,
        site=args.site,
        segments=args.segments,
        cameras=args.cameras,
        log_level=args.log_level,
    )
    raise SystemExit(rc)
