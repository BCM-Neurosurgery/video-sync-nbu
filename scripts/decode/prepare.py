"""Shared serial-audio preparation pipeline.

Consolidates the split → decode → merge → gapfill → filter workflow used by
``cli_nbu``, ``find_audio_abs_time``, and the WebUI decode workflow into a
single reusable function.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from scripts.decode.wavfileparser import (
    decode_to_raw,
    decode_split_dir_to_csvs,
    WavSerialDecoder,
    MP3_PYDUB_SIZE_HARD_LIMIT,
    MAX_DECODE_BYTES_DEFAULT,
)
from scripts.fix.audiogapfiller import gapfill_csv_file
from scripts.fix.audiofilter import filter_audio_file
from scripts.merge.mergecsv import merge_split_csvs
from scripts.split.mp3split import split_audio_to_wav

_LOG = logging.getLogger(__name__)


def _needs_split(path: Path) -> bool:
    """Return True if *path* is an audio file that exceeds the safe direct-decode cap."""
    ext = path.suffix.lower()
    if ext not in (".mp3", ".wav"):
        return False
    try:
        fbytes = os.path.getsize(path)
    except OSError:
        return False
    env = os.getenv("VIDEOSYNC_DECODE_MAX_BYTES")
    try:
        soft_cap = int(env) if env else MAX_DECODE_BYTES_DEFAULT
    except Exception:
        soft_cap = MAX_DECODE_BYTES_DEFAULT
    if ext == ".mp3" and fbytes >= MP3_PYDUB_SIZE_HARD_LIMIT:
        return True
    return fbytes >= soft_cap


def _prepare_serial_csv_per_file(
    serial_segments: list[dict],
    artifact_root: Path,
    site: str,
    *,
    logger: logging.Logger,
) -> Path:
    """Decode each Reaper-split WAV separately, merge per-file CSVs into raw.csv.

    Each entry in *serial_segments* must contain ``path`` (absolute path to a
    Reaper WAV) and ``start_sample`` (cumulative sample offset in the merged
    timeline).  Returns the path to the merged ``raw.csv``; the caller still
    runs gapfill + filter.
    """
    audio_decoded_dir = artifact_root / "audio_decoded"
    audio_decoded_dir.mkdir(parents=True, exist_ok=True)
    split_decoded = artifact_root / "split_decoded"
    split_decoded.mkdir(parents=True, exist_ok=True)

    # Wipe stale per-file CSVs from prior runs to avoid mixing.
    for stale in split_decoded.glob("*.csv"):
        stale.unlink(missing_ok=True)

    manifest_segments: list[dict] = []
    for seg in serial_segments:
        wav_path = Path(seg["path"])
        offset = int(seg["start_sample"])
        csv_out = split_decoded / f"{wav_path.stem}.csv"

        logger.info(
            "Decoding %s independently (offset=%d samples)", wav_path.name, offset
        )
        dec = WavSerialDecoder(str(wav_path))
        counts, _stats = dec.parse_counts(site=site, threshold=0.5)
        dec.save_counts_csv(csv_out, counts, site=site, offset_samples=offset)
        logger.info("→ %s (%d rows)", csv_out.name, len(counts))

        manifest_segments.append({"file": csv_out.name, "start_sample": offset})

    manifest_path = split_decoded / "_segments_manifest.json"
    manifest_path.write_text(
        json.dumps({"segments": manifest_segments}, indent=2), encoding="utf-8"
    )

    decoded_raw_csv = merge_split_csvs(
        split_dir=split_decoded,
        outdir=audio_decoded_dir,
        pattern="*.csv",
        manifest=manifest_path,
        output_name="raw.csv",
        gzip_output=False,
        dedupe=True,
        tolerance_samples=0,
        logger=logger,
    )
    logger.info("Merged per-file CSVs → %s", decoded_raw_csv.name)
    return decoded_raw_csv


def prepare_serial_csv(
    serial_audio_path: Path,
    artifact_root: Path,
    site: str,
    *,
    skip_decode: bool = False,
    do_split: bool | None = None,
    split_chunk_seconds: int = 3600,
    split_overwrite: bool = False,
    split_clean: bool = False,
    split_outdir: Path | None = None,
    run_analysis: bool = False,
    logger: logging.Logger | None = None,
) -> Path:
    """Decode/gapfill/filter serial audio and return the filtered CSV path.

    Parameters
    ----------
    serial_audio_path:
        Path to the serial audio file (MP3 or WAV, typically channel -03).
    artifact_root:
        Root output directory.  Intermediate files are written to standard
        subdirectories (``audio_decoded/``, ``serial_audio_splitted/``,
        ``split_decoded/``).
    site:
        Decoder site preset (``jamail``, ``nbu_lounge``, ``nbu_sleep``).
    skip_decode:
        If *True*, assume decoded artifacts already exist and skip all
        decoding.  Raises :class:`FileNotFoundError` if they are missing.
    do_split:
        ``True``  – force the split→decode→merge path.
        ``False`` – force direct single-file decode.
        ``None``  – auto-detect based on file size.
    split_chunk_seconds:
        Chunk duration for MP3 splitting (default 3600 s = 1 hour).
    split_overwrite:
        Allow ffmpeg to overwrite existing chunk files.
    split_clean:
        Delete existing chunks before splitting.
    split_outdir:
        Custom directory for WAV chunks; defaults to
        ``<artifact_root>/serial_audio_splitted``.
    run_analysis:
        If *True*, run ``analyze_csv_serials`` after each stage (raw,
        gapfilled, filtered).  Only needed by the CLI driver.
    logger:
        Optional caller-specific logger; falls back to module logger.

    Returns
    -------
    Path
        Path to the filtered CSV (``raw-gapfilled-filtered.csv``).
    """
    log = logger or _LOG

    audio_decoded_dir = artifact_root / "audio_decoded"
    audio_decoded_dir.mkdir(parents=True, exist_ok=True)

    # -- Auto-detect split need ------------------------------------------------
    if do_split is None:
        do_split = _needs_split(serial_audio_path)
        if do_split:
            log.info(
                "Auto-detected large audio (%s); using split→decode→merge path.",
                serial_audio_path.name,
            )

    # -- Skip-decode path ------------------------------------------------------
    if skip_decode:
        if do_split:
            log.info("--skip-decode is set; ignoring --split.")
        decoded_raw_csv = audio_decoded_dir / "raw.csv"
        prefiltered_csv = audio_decoded_dir / "raw-gapfilled-filtered.csv"

        missing = [p for p in (decoded_raw_csv, prefiltered_csv) if not p.exists()]
        if missing:
            for path in missing:
                log.error("--skip-decode set but missing %s", path.name)
            raise FileNotFoundError(
                "Required decoded audio artifacts missing while --skip-decode."
            )

        log.info(
            "Skip decode: using %s and %s",
            decoded_raw_csv.name,
            prefiltered_csv.name,
        )
        return prefiltered_csv

    # -- Reuse existing filtered CSV -------------------------------------------
    filtered_csv = audio_decoded_dir / "raw-gapfilled-filtered.csv"
    if filtered_csv.exists() and not split_overwrite:
        log.info("Reusing existing filtered CSV: %s", filtered_csv.name)
        return filtered_csv

    # -- Reaper merged-segments path (per-file decode) -------------------------
    # When the input is a merged Reaper recording (multiple WAVs concatenated
    # via merge_channel_wavs), decode each source WAV independently to avoid
    # the decoder losing bit-alignment at the WAV-to-WAV seams.  The sidecar
    # JSON written by merge_channel_wavs contains the per-file sample offsets.
    sidecar = serial_audio_path.with_suffix(".json")
    if (
        not do_split
        and serial_audio_path.suffix.lower() == ".wav"
        and serial_audio_path.name.startswith("merged-")
        and sidecar.exists()
    ):
        try:
            meta = json.loads(sidecar.read_text())
            segments = meta.get("segments") or []
        except (OSError, ValueError) as exc:
            log.warning(
                "Failed to read merge sidecar %s: %s; falling back to direct decode.",
                sidecar.name,
                exc,
            )
            segments = []
        if len(segments) > 1:
            log.info(
                "Detected merged Reaper WAV with %d source segments; "
                "decoding per-file to avoid bit-alignment loss at seams.",
                len(segments),
            )
            decoded_raw_csv = _prepare_serial_csv_per_file(
                segments, artifact_root, site, logger=log
            )
            if run_analysis:
                _run_analysis(decoded_raw_csv, log, stage="Raw")

            gapfilled_csv = gapfill_csv_file(input_csv=decoded_raw_csv)
            log.info("Gap-filled %s → %s", decoded_raw_csv.name, gapfilled_csv.name)
            if run_analysis:
                _run_analysis(gapfilled_csv, log, stage="Gap-filled")

            filtered_csv = filter_audio_file(input_csv=gapfilled_csv)
            log.info("Filtered %s → %s", gapfilled_csv.name, filtered_csv.name)
            if run_analysis:
                _run_analysis(filtered_csv, log, stage="Filtered")

            return filtered_csv
        elif segments:
            log.info(
                "Merge sidecar lists a single source segment; using direct decode."
            )
        else:
            log.warning(
                "Merge sidecar %s lacks 'segments'; using direct decode "
                "(decoder may misalign at WAV seams). Re-run merge to refresh.",
                sidecar.name,
            )

    # -- Split → decode → merge ------------------------------------------------
    if do_split:
        chunks_dir = split_outdir or (artifact_root / "serial_audio_splitted")
        chunks_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "Splitting serial audio into %ds chunks at %s",
            split_chunk_seconds,
            chunks_dir.name,
        )
        split_audio_to_wav(
            input_audio=serial_audio_path,
            outdir=chunks_dir,
            chunk_seconds=split_chunk_seconds,
            start_number=1,
            overwrite=split_overwrite,
            clean=split_clean,
            ffmpeg_bin="ffmpeg",
            ffmpeg_loglevel="info",
        )

        manifest_path = chunks_dir / f"{serial_audio_path.stem}_manifest.json"
        if not manifest_path.exists():
            log.warning("Manifest not found: %s", manifest_path.name)
            manifest_path = None

        split_csv_dir = artifact_root / "split_decoded"
        split_csv_dir.mkdir(parents=True, exist_ok=True)

        decode_split_dir_to_csvs(
            split_dir=chunks_dir,
            outdir=split_csv_dir,
            site=site,
            threshold=0.5,
            pattern=f"{serial_audio_path.stem}-[0-9][0-9][0-9].wav",
            manifest=manifest_path,
        )

        decoded_raw_csv = merge_split_csvs(
            split_dir=split_csv_dir,
            outdir=audio_decoded_dir,
            pattern="*.csv",
            manifest=manifest_path,
            output_name="raw.csv",
            gzip_output=False,
            dedupe=True,
            tolerance_samples=0,
            logger=log,
        )
        log.info("Merged per-chunk CSVs → %s", decoded_raw_csv.name)

    # -- Direct decode ---------------------------------------------------------
    else:
        log.info("Decoding serial…")
        decoded_raw_csv, _, _, _ = decode_to_raw(
            serial_audio_path, audio_decoded_dir, site=site
        )
        log.info("Decoded audio serial → %s", decoded_raw_csv.name)

    # -- Optional analysis -----------------------------------------------------
    if run_analysis:
        _run_analysis(decoded_raw_csv, log, stage="Raw")

    # -- Gapfill ---------------------------------------------------------------
    gapfilled_csv = gapfill_csv_file(input_csv=decoded_raw_csv)
    log.info("Gap-filled %s → %s", decoded_raw_csv.name, gapfilled_csv.name)

    if run_analysis:
        _run_analysis(gapfilled_csv, log, stage="Gap-filled")

    # -- Filter ----------------------------------------------------------------
    filtered_csv = filter_audio_file(input_csv=gapfilled_csv)
    log.info("Filtered %s → %s", gapfilled_csv.name, filtered_csv.name)

    if run_analysis:
        _run_analysis(filtered_csv, log, stage="Filtered")

    return filtered_csv


def _run_analysis(csv_path: Path, log: logging.Logger, *, stage: str) -> None:
    """Run serial analysis on *csv_path*, logging warnings on failure."""
    try:
        from scripts.analysis.csv_serial_analysis import analyze_csv_serials

        _, txt = analyze_csv_serials(path=csv_path)
        log.info("%s analysis → %s", stage, txt.name)
    except Exception as exc:
        log.error("%s analysis failed: %s", stage, exc)
