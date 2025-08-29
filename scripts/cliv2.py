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


def process_segment(
    video_in: str,
    audiogroup: AudioGroup,
    seg_id: str,
    serial_audio_path: Path,
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
            serial_audio_path, audio_decoded_dir, site=site
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
    # set inputs and outputs
    serial_audio_path = Path(
        "/home/auto/CODE/utils/video-sync-nbu/data/nbu_lounge_example_2/audio/TRBD002_08062025-03.mp3"
    )
    audio_in = Path(
        "/home/auto/CODE/utils/video-sync-nbu/data/nbu_lounge_example_2/audio"
    )
    video_in = Path(
        "/home/auto/CODE/utils/video-sync-nbu/data/nbu_lounge_example_2/video"
    )
    site = "jamail"
    parent_out = Path(
        "/home/auto/CODE/utils/video-sync-nbu/data/nbu_lounge_example_2/out"
    )
    segments = ["TRBD002_20250806_104707"]
    cameras = ["23512909"]

    try:
        ag = AudioDiscoverer(
            audio_dir=audio_in,
            log=logger,
        )
        logger.info("Discovered Audios")
    except AudioGroupDiscoverError as e:
        logger.exception("Audio group discovery failed: %s", e)

    for seg in segments:
        report = process_segment(
            video_in=video_in,
            audio_in=audio_in,
            seg_id=seg,
            serial_audio_path=serial_audio_path,
            site=site,
            parent_out=parent_out,
            cam_serials=cameras,
        )

    # Decide exit code or follow-up based on failures
    # failed = sum(len(r["failed"]) for r in all_reports)
    # if failed:
    #     raise SystemExit(2)
