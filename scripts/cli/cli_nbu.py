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
- Use **--skip-decode** to reuse existing decoded + processed CSVs in:
  <out_dir>/audio_decoded/raw.csv and raw-gapfilled-filtered.csv
  This skips decoding, gapfill, and filter, and starts at per-camera steps.

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
    - serial_audio_splitted (if --split used)
        - TRBD002_08062025-03-001.wav
        - TRBD002_08062025-03-002.wav
        - ...
        - TRBD002_08062025-03_manifest.json
    - split_decoded (if --split used)
        - TRBD002_08062025-03-001.csv
        - TRBD002_08062025-03-002.csv (global index with --manifest)
        - ...
    - audio_decoded (shared across segments)
        - raw.csv (merged)
        - raw.txt
        - raw-gapfilled.csv
        - raw-gapfilled.txt
        - raw-gapfilled-filtered.csv
        - raw-gapfilled-filtered.txt
    - <segment_id> (e.g. TRBD002_20250806_104707)
        - <camera_serial1> (e.g. 23512909)
            - work (intermediate artifacts)
                - gapfilled-filtered-anchors.json             (anchors from filtered CSV)
                - gapfilled-filtered-anchors.txt

                - gapfilled-filtered-clipped.csv              (CSV clipped to video window)
                - gapfilled-filtered-clipped.txt

                - gapfilled-filtered-clipped-local.csv        (clipped, localized to audio dir)
                - gapfilled-filtered-clipped-local.txt

                - gapfilled-filtered-clipped-local-editplan.json
                - gapfilled-filtered-clipped-local-padded.csv
                - gapfilled-filtered-clipped-local-padded.txt

                - gapfilled-filtered-padded-anchors.json      (anchors after padding)
                - gapfilled-filtered-padded-anchors.txt

                - TRBD002_20250806_103745.24253448.txt
                - TRBD002_20250806_103745.24253448-frameid.txt
                - TRBD002_20250806_103745.24253448-frameid.json
                - TRBD002_20250806_103745.24253448-frameid-padplan.json

            - audio_clipped
                - TRBD002_08062025-clipped-01.mp3
                - TRBD002_08062025-clipped-03.mp3
            - audio_padded
                - TRBD002_08062025-clipped-padded-01.mp3
                - TRBD002_08062025-clipped-padded-03.mp3
            - video_padded (TODO)
                - TRBD002_20250806_103745.24253448.mp4
                - TRBD002_20250806_103745.json
            - synced_audio
            - synced_video  (final synced videos)
            - sync.log  ← per-camera rotating log (5MB x 3 backups), stamped with [seg/cam]

        - <camera_serial2>
        ...
    - <segment_id> (e.g. TRBD002_20250806_105724)

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
import re
import shutil
import subprocess
from typing import Iterable

from scripts.decode.wavfileparser import decode_to_raw, decode_split_dir_to_csvs
from scripts.analysis.csv_serial_analysis import analyze_csv_serials
from scripts.analysis.anchor_analysis import analyze_anchors_file
from scripts.analysis.video_analysis import analyze_video
from scripts.fix.audiogapfiller import gapfill_csv_file
from scripts.fix.audiofilter import filter_audio_file
from scripts.align.collect_anchors import save_anchors_for_camera
from scripts.clip.audiocsvclipper import clip_with_anchors
from scripts.clip.audioclip import clip_from_csv
from scripts.pad.audiopadder import AudioPadder
from scripts.pad.audioplanapplier import AudioPlanApplier
from scripts.pad.videoplancreater import create_video_padding_plan
from scripts.pad.videoplanapplier import apply_video_padding_plan
from scripts.align.sync import sync_one_video
from scripts.index.discover import AudioDiscoverer
from scripts.index.videodiscover import build_video_obj
from scripts.merge.mergecsv import merge_split_csvs
from scripts.split.mp3split import split_mp3_to_wav
from scripts.utility.utils import _name
from scripts.models import AudioGroup, Video
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
    VideoPaddingError,
    VideoDiscoverError,
)

from scripts.log.logutils import (
    configure_logging,
    attach_cam_logger,
    log_context,
)

logger = logging.getLogger("cli")

SITE_CHOICES = ("jamail", "nbu_lounge", "nbu_sleep")
UPSIDE_DOWN_CAMERAS: set[str] = {"24253458"}


def list_segments(video_dir: Path) -> list[str]:
    """
    Discover segment IDs from *.json basenames, returning them chronologically.
    Segment IDs containing YYYYMMDD_HHMMSS parts are sorted by that timestamp;
    any others fall back to a lexicographic order at the end.
    """
    segment_ids = {p.stem for p in Path(video_dir).glob("*.json")}

    def _sort_key(seg: str) -> tuple:
        match = re.search(r"(\d{8})_(\d{6})", seg)
        if match:
            try:
                return (0, int(match.group(1)), int(match.group(2)), seg)
            except ValueError:
                pass
        return (1, seg)

    return sorted(segment_ids, key=_sort_key)


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


def _flip_video_if_needed(
    cam_serial: str, synced_path: Path, log: logging.Logger
) -> None:
    """Flip cameras that record upside down by re-encoding with vflip+hflip."""
    if cam_serial not in UPSIDE_DOWN_CAMERAS:
        return

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; required to flip upside-down video."
        )

    tmp_path = synced_path.with_name(f"{synced_path.stem}.flipped{synced_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    log.info("Flipping upside-down camera %s → %s", cam_serial, synced_path.name)

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(synced_path),
        "-vf",
        "vflip,hflip",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        "-c:a",
        "copy",
        str(tmp_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(
            f"ffmpeg flip failed for {synced_path.name}: {stderr or 'unknown error'}"
        )

    tmp_path.replace(synced_path)
    log.info("Applied flip for upside-down camera %s", cam_serial)


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


def run_pipeline(
    audio_dir: Path,
    video_dir: Path,
    out_dir: Path,
    site: str,
    segments: list[str] | None,
    cameras: list[str] | None,
    log_level: str = "INFO",
    skip_decode: bool = False,
    *,
    do_split: bool = False,
    split_chunk_seconds: int = 3600,
    split_overwrite: bool = False,
    split_clean: bool = False,
    split_outdir: Path | None = None,
    overwrite_clips: bool = False,
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
        ag = ad.get_audio_group()
        logger.info("Audio(s) discovered")
    except AudioGroupDiscoverError as e:
        logger.error("Audio group discovery failed: %s", e)
        return 2

    try:
        targets = build_targets(video_dir, segments, cameras)
    except TargetBuildError as e:
        logger.error("%s", e)
        return 3

    try:
        filtered_csv = prepare_serial_audio(
            audiogroup=ag,
            parent_out=out_dir,
            site=site,
            skip_decode=skip_decode,
            do_split=do_split,
            split_chunk_seconds=split_chunk_seconds,
            split_overwrite=split_overwrite,
            split_clean=split_clean,
            split_outdir=split_outdir,
        )
    except (AudioDecodingError, GapFillError, FilteredError, FileNotFoundError) as e:
        logger.error("Audio preparation failed: %s", e)
        return 4
    except Exception as e:
        logger.error("Unexpected error preparing audio: %s", e)
        return 4

    failures = 0
    for seg_id, cam_list in targets.items():
        summary = process_segment(
            video_in=video_dir,
            audio_in=audio_dir,
            seg_id=seg_id,
            parent_out=out_dir,
            cam_serials=cam_list,
            filtered_csv=filtered_csv,
            overwrite_clips=overwrite_clips,
        )
        if summary["fail"]:
            failures += 1
            logger.warning(
                "segment %s: done with failures: %s", seg_id, ", ".join(summary["fail"])
            )
        else:
            logger.info("segment %s: done!", seg_id)

    return 0 if failures == 0 else 4


def prepare_serial_audio(
    audiogroup: AudioGroup,
    parent_out: Path,
    site: str,
    *,
    skip_decode: bool = False,
    do_split: bool = False,
    split_chunk_seconds: int = 3600,
    split_overwrite: bool = False,
    split_clean: bool = False,
    split_outdir: Path | None = None,
) -> Path:
    """Decode/gapfill/filter the serial audio once per run and return filtered CSV."""
    audio_decoded_dir = parent_out / "audio_decoded"
    audio_decoded_dir.mkdir(parents=True, exist_ok=True)

    with log_context(seg="-", cam="-"):
        if skip_decode:
            if do_split:
                logger.info("--skip-decode is set; ignoring --split.")
            decoded_raw_csv = audio_decoded_dir / "raw.csv"
            prefiltered_csv = audio_decoded_dir / "raw-gapfilled-filtered.csv"

            missing = []
            if not decoded_raw_csv.exists():
                missing.append(decoded_raw_csv)
            if not prefiltered_csv.exists():
                missing.append(prefiltered_csv)

            if missing:
                for path in missing:
                    logger.error("--skip-decode set but missing %s", _name(path))
                raise FileNotFoundError(
                    "Required decoded audio artifacts missing while --skip-decode."
                )

            logger.info(
                "Skip decode: using %s and %s",
                _name(decoded_raw_csv),
                _name(prefiltered_csv),
            )
            return prefiltered_csv

        if do_split:
            serial_mp3 = Path(audiogroup.serial_audio.path)
            chunks_dir = split_outdir or (parent_out / "serial_audio_splitted")
            chunks_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Splitting serial MP3 into %ds chunks at %s",
                split_chunk_seconds,
                _name(chunks_dir),
            )
            split_mp3_to_wav(
                input_mp3=serial_mp3,
                outdir=chunks_dir,
                chunk_seconds=split_chunk_seconds,
                start_number=1,
                overwrite=split_overwrite,
                clean=split_clean,
                ffmpeg_bin="ffmpeg",
                ffmpeg_loglevel="info",
            )

            manifest_path = chunks_dir / f"{serial_mp3.stem}_manifest.json"
            if not manifest_path.exists():
                logger.warning("Manifest not found: %s", _name(manifest_path))
                manifest_path = None

            split_csv_dir = parent_out / "split_decoded"
            split_csv_dir.mkdir(parents=True, exist_ok=True)

            decode_split_dir_to_csvs(
                split_dir=chunks_dir,
                outdir=split_csv_dir,
                site=site,
                threshold=0.5,
                pattern=f"{serial_mp3.stem}-[0-9][0-9][0-9].wav",
                manifest=manifest_path,
            )

            merged = merge_split_csvs(
                split_dir=split_csv_dir,
                outdir=audio_decoded_dir,
                pattern="*.csv",
                manifest=manifest_path,
                output_name="raw.csv",
                gzip_output=False,
                dedupe=True,
                tolerance_samples=0,
                logger=logger,
            )
            decoded_raw_csv = merged
            logger.info("Merged per-chunk CSVs → %s", _name(decoded_raw_csv))
        else:
            logger.info("Decoding serial…")
            decoded_raw_csv, _, _, _ = decode_to_raw(
                audiogroup.serial_audio.path, audio_decoded_dir, site=site
            )
            logger.info("Decoded audio serial → %s", _name(decoded_raw_csv))

        try:
            _, raw_txt = analyze_csv_serials(path=decoded_raw_csv)
            logger.info("Analyzed %s → %s", _name(decoded_raw_csv), _name(raw_txt))
        except SerialAnalysisError as e:
            logger.error("Raw analysis failed: %s", e)

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
        except GapFillError:
            raise

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
        except FilteredError:
            raise

    return filtered_csv


def process_segment(
    video_in: str,
    audio_in: str,
    seg_id: str,
    parent_out: Path,
    cam_serials: Iterable[str],
    filtered_csv: Path,
    *,
    overwrite_clips: bool,
) -> dict:
    """Process one segment across one or more cameras. Returns a summary dict."""
    segment_out = parent_out / seg_id
    summary = {"segment": seg_id, "ok": [], "fail": []}

    with log_context(seg=seg_id, cam="-"):
        logger.info("Reusing filtered serial CSV %s", _name(filtered_csv))

    # ---- Per camera workflow (camera-scoped context) ----
    for cam in cam_serials:
        cam_out = segment_out / cam
        (cam_out / "work").mkdir(parents=True, exist_ok=True)
        (cam_out / "audio_padded").mkdir(parents=True, exist_ok=True)

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
                # Discover a Video object
                try:
                    vid = build_video_obj(
                        video_dir=video_in,
                        segment_id=seg_id,
                        cam_serial=cam,
                        log=logger,
                    )
                    clog.info("Built Video object: %s", _name(vid.path))
                except VideoDiscoverError as e:
                    clog.error("Video discovery failed: %s", e)
                    summary["fail"].append(f"{cam}:video-discover")
                    continue

                # Video analysis
                # Whether there's dropped frames during acquisition
                try:
                    vid_res = analyze_video(
                        video=vid,
                        outdir=cam_out / "work",
                    )
                    clog.info("Video analysis completed")
                except VideoAnalysisError as e:
                    clog.error("Video analysis failed: %s", e)
                    summary["fail"].append(f"{cam}:video-analysis")
                    vid_res = None

                # ---------- Video padding (always-on: only if missing frames) ----------
                if vid_res is not None and vid_res.missing_frames > 0:
                    clog.warning("Video has %d missing frames", vid_res.missing_frames)
                    try:
                        # 1) From video analysis json, build a padding plan json
                        # with the name <video>_videopad.json
                        try:
                            vid_padjson = create_video_padding_plan(
                                analysis_json=vid_res.out_json_path,
                                target_fps=30.0,
                                expect_step=1,
                                policy="dup-prev",
                                outdir=cam_out / "work",
                            )
                            clog.info(
                                "Video padding plan created → %s", _name(vid_padjson)
                            )
                        except VideoPaddingError as e:
                            clog.error("Video padding plan creation failed: %s", e)
                            summary["fail"].append(f"{cam}:video-pad-plan")
                            vid_padjson = None

                        # 2) Apply plan (if available)
                        if vid_padjson is not None:
                            padded_dir = cam_out / "video_padded"
                            padded_dir.mkdir(parents=True, exist_ok=True)

                            try:
                                # pad and save the video
                                # pad Video object's fixed serial with 0s
                                # pad fixed_frames_ids, and fixed_frame_idx_reidx
                                out_path, vid = apply_video_padding_plan(
                                    plan_json=vid_padjson,
                                    video=vid,
                                    out_dir=padded_dir,
                                    crf=20,
                                    preset="veryfast",
                                    override_target_fps=30.0,
                                )
                                if out_path.exists():
                                    clog.info(
                                        "Video padding plan applied → %s",
                                        _name(out_path),
                                    )
                                else:
                                    clog.error("Padded video not found after apply")
                                    summary["fail"].append(
                                        f"{cam}:video-pad-apply-missing"
                                    )
                            except VideoPaddingError as e:
                                clog.error("Video padding apply failed: %s", e)
                                summary["fail"].append(f"{cam}:video-pad-apply")

                            try:
                                analyze_video(
                                    video=vid,
                                    outdir=padded_dir,
                                )
                                clog.info("Video analysis completed")
                            except VideoAnalysisError as e:
                                clog.error("Video analysis failed: %s", e)
                                summary["fail"].append(f"{cam}:video-analysis-postpad")

                    except Exception as e:
                        # Catch-all for any unexpected failure inside the padding block
                        clog.error("Video padding step failed: %s", e)
                        summary["fail"].append(f"{cam}:video-pad")
                else:
                    clog.info("No missing frames detected; padding not needed.")

                # Anchors from filtered CSV
                try:
                    save_anchors_for_camera(
                        serial_csv=filtered_csv,
                        video=vid,
                        out_json=filtered_anchors,
                    )
                    clog.info("Saved %s", _name(filtered_anchors))
                    try:
                        analyze_anchors_file(anchors_json=filtered_anchors)
                        clog.info("anchors analyzed")
                    except (AnchorError, ValueError) as e:
                        clog.error("anchors analyze failed: %s", e)
                        reason = (
                            "anchors-empty"
                            if isinstance(e, ValueError)
                            else "anchors-analyze"
                        )
                        summary["fail"].append(f"{cam}:{reason}")
                        continue
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
                    clog.info(
                        "Clipped %s → %s", _name(filtered_csv), _name(clipped_csv)
                    )
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

                # Clip audio and write localized CSV
                try:
                    _, _, local_csv = clip_from_csv(
                        csv_path=clipped_csv,
                        audio_dir=audio_in,
                        out_dir=cam_out / "audio_clipped",
                        sr=44100,
                        margin_sec=5,
                        overwrite=overwrite_clips,
                    )
                    clog.info(
                        "Clipped Audio, localized clipped CSV → %s", _name(local_csv)
                    )
                    try:
                        _, local_txt = analyze_csv_serials(path=local_csv)
                        clog.info(
                            "Analyzed %s → %s", _name(local_csv), _name(local_txt)
                        )
                    except SerialAnalysisError as e:
                        clog.error("local analysis failed: %s", e)
                        summary["fail"].append(f"{cam}:local-analysis")
                except ClipError as e:
                    clog.error("Audio clip failed: %s", e)
                    summary["fail"].append(f"{cam}:audio-clip")
                    continue

                try:
                    apadder = AudioPadder(
                        csv_path=local_csv,
                        include_synthetic=True,
                        gap_policy="local",
                        sample_rate=44100,
                    )
                    _, _, padded_csv, padplan = apadder.run()
                    clog.info("Padded CSV → %s", _name(padded_csv))
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
                wavs = sorted((cam_out / "audio_clipped").glob("*.wav"))
                if not wavs:
                    clog.warning(
                        "no clipped WAVs found in %s", _name(cam_out / "audio_clipped")
                    )

                for wav_path in wavs:
                    try:
                        applier = AudioPlanApplier(
                            audio_path=wav_path,
                            plan_path=padplan,
                            out_dir=cam_out / "audio_padded",
                        )
                        out = applier.apply()
                        clog.info("plan applied → %s", _name(out))
                    except AudioPlanError as e:
                        clog.error("plan apply failed for %s: %s", wav_path.name, e)
                        summary["fail"].append(f"{cam}:plan-{wav_path.stem}")

                # Anchors from padded CSV
                # This anchor SHOULD BE perfect
                try:
                    save_anchors_for_camera(
                        serial_csv=padded_csv,
                        video=vid,
                        out_json=padded_anchors,
                    )
                    clog.info("Saved %s", _name(padded_anchors))
                    try:
                        analyze_anchors_file(anchors_json=padded_anchors)
                        clog.info("padded-anchors analyzed")
                    except (AnchorError, ValueError) as e:
                        clog.error("padded-anchors analyze failed: %s", e)
                        reason = (
                            "padded-anchors-empty"
                            if isinstance(e, ValueError)
                            else "padded-anchors-analyze"
                        )
                        summary["fail"].append(f"{cam}:{reason}")
                        continue
                except AnchorError as e:
                    clog.error("padded-anchors save failed: %s", e)
                    summary["fail"].append(f"{cam}:padded-anchors-save")
                    continue

                # Final sync
                try:
                    synced_video_path = sync_one_video(
                        audio_dir=cam_out / "audio_padded",
                        video=vid,
                        anchors_json=padded_anchors,
                        out_audio_dir=cam_out / "synced_audio",
                        out_video_dir=cam_out / "synced_video",
                    )
                except SyncError as e:
                    clog.error("sync failed: %s", e)
                    summary["fail"].append(f"{cam}:sync")
                else:
                    try:
                        _flip_video_if_needed(cam, synced_video_path, clog)
                    except RuntimeError as e:
                        clog.error("post-sync flip failed: %s", e)
                        summary["fail"].append(f"{cam}:flip")
                        continue
                    clog.info("synced ✔")
                    summary["ok"].append(cam)

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
        help="Reuse existing <out>/audio_decoded/raw.csv and raw-gapfilled-filtered.csv; skip decoding+gapfill+filter and start per-camera workflow.",
    )
    parser.add_argument(
        "--split",
        dest="do_split",
        action="store_true",
        help="Batch mode: split the serial MP3 into chunks, then decode+merge (manifest used automatically).",
    )
    parser.add_argument(
        "--split-chunk-seconds",
        type=int,
        default=3600,
        help="Chunk size in seconds when splitting.",
    )
    gsplit = parser.add_mutually_exclusive_group()
    gsplit.add_argument(
        "--split-overwrite",
        action="store_true",
        help="Allow overwriting existing chunk files when splitting.",
    )
    gsplit.add_argument(
        "--split-clean",
        action="store_true",
        help="Delete existing chunk files before splitting.",
    )
    parser.add_argument(
        "--split-outdir",
        type=Path,
        help="Where to place chunks when splitting (default: <out>/serial_audio_splitted).",
    )
    parser.add_argument(
        "--overwrite-clips",
        action="store_true",
        help="Allow ffmpeg to overwrite existing clipped audio files.",
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
        do_split=args.do_split,
        split_chunk_seconds=args.split_chunk_seconds,
        split_overwrite=args.split_overwrite,
        split_clean=args.split_clean,
        split_outdir=args.split_outdir,
        overwrite_clips=args.overwrite_clips,
    )
    raise SystemExit(rc)
