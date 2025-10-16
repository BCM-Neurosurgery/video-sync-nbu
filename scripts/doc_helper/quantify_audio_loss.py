#!/usr/bin/env python3
"""
batch_audio_loss_report.py — NBU lounge/sleep + Jamail clinic pipeline:
discover → split/decode/merge → GAPFILL → analyze → per-audio + global reports.

What's new
----------
• After merging the per-chunk CSVs, we call:
      gapfilled_csv = gapfill_csv_file(input_csv=merged_csv)
  and use the gap-filled CSV for analysis.

Scans ONLY:
<root>/
  TRBD001|TRBD002/NBU/<YYYY-MM-DD>/audio/<room>/...*03.(mp3|wav)
    where <room> is "lounge" (nbu_lounge) or "sleep" (nbu_sleep).
  TRBD001|TRBD002/clinic/<YYYY-MM-DD>/audio/...*03.(mp3|wav)  (jamail site)

Artifacts
---------
Per-audio JSON:
  <out_dir>/reports/<patient>/<date>/<room>/<audio_stem>.audio_loss.json

Decoder work dirs:
  <out_dir>/work/<patient>/<date>/<room>/<audio_stem>/
    chunks/          # wav chunks (real for mp3, pseudo for wav)
    split_decoded/   # csv per chunk

Merged raw CSV:
  <out_dir>/raw/<patient>/<date>/<room>/<audio_stem>.raw.csv

Gap-filled CSV (default naming by gapfiller):
  <out_dir>/raw/<patient>/<date>/<room>/<audio_stem>.raw-gapfilled.csv

Global summary:
  <out_dir>/summary/audio_loss_summary.csv
  <out_dir>/summary/audio_loss_overall.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Analyzer (START-based) — reused directly
from scripts.analysis.audio_loss_analysis import (  # type: ignore
    load_serial_csv,
    apply_prefilter,
    analyze_loss_json,
)

# Split/decode/merge utilities
from scripts.split.mp3split import split_mp3_to_wav  # type: ignore
from scripts.decode.wavfileparser import decode_split_dir_to_csvs  # type: ignore
from scripts.merge.mergecsv import merge_split_csvs  # type: ignore

# GAP FILLER
from scripts.fix.audiogapfiller import gapfill_csv_file  # type: ignore

LOG = logging.getLogger("batch-audio-loss")

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$", re.ASCII)
SERIAL_AUDIO_RE = re.compile(r".*03\.(mp3|wav)$", re.IGNORECASE)


@dataclass(frozen=True)
class SiteLayout:
    patient_subdir: str  # e.g., "NBU" or "clinic"
    audio_subdir: str  # usually "audio"
    include_room_subdir: bool  # whether to append /<room> after audio_subdir
    room: str  # logical room name for reporting


SITE_LAYOUTS: Dict[str, SiteLayout] = {
    "nbu_lounge": SiteLayout(
        patient_subdir="NBU",
        audio_subdir="audio",
        include_room_subdir=True,
        room="lounge",
    ),
    "nbu_sleep": SiteLayout(
        patient_subdir="NBU",
        audio_subdir="audio",
        include_room_subdir=True,
        room="sleep",
    ),
    "jamail": SiteLayout(
        patient_subdir="clinic",
        audio_subdir="audio",
        include_room_subdir=False,
        room="clinic",
    ),
}
DEFAULT_SITE = "nbu_lounge"
SITE_CHOICES = tuple(sorted(SITE_LAYOUTS))


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioItem:
    patient: str
    date: str
    room: str
    path: Path

    @property
    def stem(self) -> str:
        return self.path.stem  # e.g., "TRBD002_20250806_104707-03"


# --------------------------------------------------------------------------------------
# Discovery (per site layout)
# --------------------------------------------------------------------------------------
def find_serial_audios(
    root: Path, patients: Sequence[str], layout: SiteLayout
) -> List[AudioItem]:
    items: List[AudioItem] = []
    for patient in patients:
        base = root / patient / layout.patient_subdir
        if not base.exists():
            continue
        for date_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            if not DATE_RE.match(date_dir.name):
                continue
            audio_root = date_dir / layout.audio_subdir
            room_dir = (
                audio_root / layout.room if layout.include_room_subdir else audio_root
            )
            if not room_dir.exists():
                continue
            for f in sorted(room_dir.iterdir()):
                if f.is_file() and SERIAL_AUDIO_RE.match(f.name):
                    items.append(AudioItem(patient, date_dir.name, layout.room, f))
    items.sort(key=lambda x: (x.patient, x.date, x.room, x.path.name))
    return items


# --------------------------------------------------------------------------------------
# Split → Decode → Merge → GAPFILL
# --------------------------------------------------------------------------------------
def ensure_gapfilled_csv_for_audio(
    audio: AudioItem,
    *,
    out_dir: Path,
    threshold: float,
    decoder_site: str,
    split_chunk_seconds: int,
    split_overwrite: bool,
    split_clean: bool,
) -> Optional[Tuple[Path, Path]]:
    """
    Produce both the merged *raw* CSV and the *gap-filled* CSV for a serial audio.

    Returns
    -------
    (merged_raw_csv, gapfilled_csv) or None on failure.
    """
    # Working directories for this audio
    work_dir = out_dir / "work" / audio.patient / audio.date / audio.room / audio.stem
    chunks_dir = work_dir / "chunks"
    split_csv_dir = work_dir / "split_decoded"
    merged_out_dir = out_dir / "raw" / audio.patient / audio.date / audio.room
    for d in (chunks_dir, split_csv_dir, merged_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    merged_csv_path = merged_out_dir / f"{audio.stem}.raw.csv"
    # If merged exists, we can skip straight to gapfilling
    if not merged_csv_path.exists():
        # Prepare chunk WAVs
        if audio.path.suffix.lower() == ".mp3":
            LOG.info("Splitting MP3 to WAV chunks: %s", audio.path)
            try:
                split_mp3_to_wav(
                    input_mp3=audio.path,
                    outdir=chunks_dir,
                    chunk_seconds=int(split_chunk_seconds),
                    start_number=1,
                    overwrite=bool(split_overwrite),
                    clean=bool(split_clean),
                    ffmpeg_bin="ffmpeg",
                    ffmpeg_loglevel="info",
                )
            except Exception as e:
                LOG.error("FFmpeg split failed for %s: %s", audio.path, e)
                return None
            manifest_path = chunks_dir / f"{audio.stem}_manifest.json"
            if not manifest_path.exists():
                LOG.warning(
                    "Split manifest not found (continuing without): %s", manifest_path
                )
                manifest_path = None
            chunk_glob = f"{audio.stem}-[0-9][0-9][0-9].wav"
        else:
            # WAV source → create a single pseudo-chunk "<stem>-001.wav"
            pseudo = chunks_dir / f"{audio.stem}-001.wav"
            try:
                if pseudo.exists() and split_clean:
                    pseudo.unlink()
                if not pseudo.exists() or split_overwrite:
                    LOG.info(
                        "Creating pseudo chunk for WAV: %s → %s", audio.path, pseudo
                    )
                    shutil.copy2(audio.path, pseudo)
                else:
                    LOG.info("Using existing pseudo chunk: %s", pseudo)
            except Exception as e:
                LOG.error("Failed to prepare pseudo chunk for %s: %s", audio.path, e)
                return None
            manifest_path = None
            chunk_glob = f"{audio.stem}-[0-9][0-9][0-9].wav"

        # Decode per-chunk WAVs → per-chunk CSVs
        try:
            LOG.info("Decoding chunks → CSVs: %s (site=%s)", chunks_dir, decoder_site)
            decode_split_dir_to_csvs(
                split_dir=chunks_dir,
                outdir=split_csv_dir,
                site=decoder_site,
                threshold=float(threshold),
                pattern=chunk_glob,
                manifest=manifest_path,
            )
        except Exception as e:
            LOG.error("Decoding failed for %s: %s", audio.path, e)
            return None

        # Merge per-chunk CSVs → one raw CSV
        try:
            LOG.info("Merging per-chunk CSVs → %s", merged_csv_path)
            merge_split_csvs(
                split_dir=split_csv_dir,
                outdir=merged_out_dir,
                pattern="*.csv",
                manifest=manifest_path,
                output_name=merged_csv_path.name,
                gzip_output=False,
                dedupe=True,
                tolerance_samples=0,
                logger=LOG,
            )
        except Exception as e:
            LOG.error("Merging failed for %s: %s", audio.path, e)
            return None
    else:
        LOG.info("Reusing merged CSV: %s", merged_csv_path)

    # GAPFILL step (idempotent: function writes <stem>-gapfilled.csv by default)
    try:
        LOG.info("Gap-filling serials: %s", merged_csv_path)
        gapfilled_csv = gapfill_csv_file(input_csv=merged_csv_path)
    except Exception as e:
        LOG.error("Gapfilling failed for %s: %s", merged_csv_path, e)
        return None

    return merged_csv_path, Path(gapfilled_csv)


# --------------------------------------------------------------------------------------
# Per-audio analysis (uses GAPFILLED csv)
# --------------------------------------------------------------------------------------
def analyze_one_audio(
    audio: AudioItem,
    *,
    out_dir: Path,
    fs: int,
    prefilter: bool,
    max_fwd_delta: Optional[int],
    local_window: int,
    top: int,
    threshold: float,
    decoder_site: str,
    split_chunk_seconds: int,
    split_overwrite: bool,
    split_clean: bool,
) -> Optional[Dict]:
    """
    Full pipeline for one audio:
      split/decode/merge → GAPFILL → load CSV → (optional) prefilter → analyze → write per-audio JSON
      returns flat summary row for global aggregation
    """
    ensured = ensure_gapfilled_csv_for_audio(
        audio,
        out_dir=out_dir,
        threshold=threshold,
        decoder_site=decoder_site,
        split_chunk_seconds=split_chunk_seconds,
        split_overwrite=split_overwrite,
        split_clean=split_clean,
    )
    if not ensured:
        return None
    merged_csv, gapfilled_csv = ensured

    # Load gap-filled CSV for analysis
    try:
        df = load_serial_csv(gapfilled_csv)
    except Exception as e:
        LOG.error("Bad gapfilled CSV %s: %s", gapfilled_csv, e)
        return None

    pre_stats = None
    pre_impl = None
    if prefilter:
        try:
            df, pre_stats, pre_impl = apply_prefilter(df, max_fwd_delta)
        except Exception as e:
            LOG.error("Prefilter failed for %s: %s", gapfilled_csv, e)
            return None

    if len(df) < 2:
        LOG.warning("Too few rows after prefilter: %s", gapfilled_csv)
        return None

    # Analyze (START-based)
    payload = analyze_loss_json(
        serials=df["serial"].to_numpy(),
        starts=df["start_sample"].to_numpy(),
        ends=df["end_sample"].to_numpy(),
        fs=int(fs),
        local_window=int(local_window),
        top=int(top),
    )

    if prefilter:
        payload.setdefault("meta", {})["prefilter"] = {
            "impl": pre_impl,
            "max_fwd_delta": max_fwd_delta,
            **(pre_stats or {}),
        }

    # Persist per-audio JSON
    report_path = (
        out_dir
        / "reports"
        / audio.patient
        / audio.date
        / audio.room
        / f"{audio.stem}.audio_loss.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Flatten row for global summary
    meta = payload.get("meta", {})
    summary = payload.get("summary", {})
    tb = meta.get("time_bounds_samples") or meta.get("time_bounds_center_samples") or {}
    row = {
        "patient": audio.patient,
        "date": audio.date,
        "room": audio.room,
        "decoder_site": decoder_site,
        "audio_file": audio.path.name,
        "raw_csv": str(merged_csv),
        "gapfilled_csv": str(gapfilled_csv),
        "report_json": str(report_path),
        "fs_hz": int(fs),
        "csv_span_seconds": (
            float(tb.get("observed_total_samples", 0) / int(fs)) if tb else None
        ),
        "analyzed_seconds": float(summary.get("analyzed_seconds", 0.0)),
        "total_missing_seconds": float(summary.get("total_missing_seconds", 0.0)),
        "loss_share_pct": float(summary.get("loss_share_pct", 0.0)),
        "values_kept": int(summary.get("values_kept", 0)),
        "steps": int(summary.get("steps", 0)),
        "ok_steps": int(summary.get("ok_steps", 0)),
        "forward_jumps": int(summary.get("forward_jumps", 0)),
        "prefilter_impl": pre_impl,
        "prefilter_input_rows": (pre_stats or {}).get("input_rows"),
        "prefilter_filtered_rows": (pre_stats or {}).get("filtered_rows"),
        "prefilter_dropped_rows": (pre_stats or {}).get("dropped_rows"),
        "max_fwd_delta": max_fwd_delta if prefilter else None,
    }
    return row


# --------------------------------------------------------------------------------------
# Batch runner + summaries
# --------------------------------------------------------------------------------------
def compute_overall_totals(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "n_audios": 0,
            "sum_analyzed_seconds": 0.0,
            "sum_missing_seconds": 0.0,
            "weighted_loss_pct": 0.0,
        }
    sum_analyzed = float(df["analyzed_seconds"].fillna(0).sum())
    sum_missing = float(df["total_missing_seconds"].fillna(0).sum())
    weighted_loss = (100.0 * sum_missing / sum_analyzed) if sum_analyzed > 0 else 0.0
    return {
        "n_audios": int(len(df)),
        "sum_analyzed_seconds": round(sum_analyzed, 3),
        "sum_missing_seconds": round(sum_missing, 3),
        "weighted_loss_pct": round(weighted_loss, 3),
    }


def run_batch(
    *,
    root: Path,
    out_dir: Path,
    site: str,
    fs: int,
    prefilter: bool,
    max_fwd_delta: Optional[int],
    local_window: int,
    top: int,
    threshold: float,
    split_chunk_seconds: int,
    split_overwrite: bool,
    split_clean: bool,
    patients: Sequence[str],
) -> Tuple[pd.DataFrame, Dict]:
    try:
        layout = SITE_LAYOUTS[site]
    except KeyError as exc:
        valid = ", ".join(sorted(SITE_LAYOUTS))
        raise ValueError(f"Unsupported site {site!r}; expected one of {valid}") from exc

    audios = find_serial_audios(root, patients, layout)
    LOG.info(
        "Found %d serial audio files (site=%s, room=%s).",
        len(audios),
        site,
        layout.room,
    )

    rows: List[Dict] = []
    skipped = 0

    for a in audios:
        LOG.info("[%s | %s | %s] %s", a.patient, a.date, a.room, a.path.name)
        row = analyze_one_audio(
            a,
            out_dir=out_dir,
            fs=fs,
            prefilter=prefilter,
            max_fwd_delta=max_fwd_delta,
            local_window=local_window,
            top=top,
            threshold=threshold,
            decoder_site=site,
            split_chunk_seconds=split_chunk_seconds,
            split_overwrite=split_overwrite,
            split_clean=split_clean,
        )
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["patient", "date", "room", "audio_file"], ignore_index=True
    )
    totals = compute_overall_totals(df)

    # Write summary
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "audio_loss_summary.csv"
    summary_json = summary_dir / "audio_loss_overall.json"

    df.to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "out_dir": str(out_dir),
                "fs_hz": fs,
                "prefilter": prefilter,
                "max_fwd_delta": max_fwd_delta,
                "local_window": local_window,
                "top": top,
                "decoder_site": site,
                "room": layout.room,
                "threshold": threshold,
                "split_chunk_seconds": split_chunk_seconds,
                "split_overwrite": split_overwrite,
                "split_clean": split_clean,
                "patients": list(patients),
                "discovered": len(audios),
                "analyzed": int(len(df)),
                "skipped": int(skipped),
                "totals": totals,
            },
            f,
            indent=2,
        )

    LOG.info("Summary CSV:   %s", summary_csv)
    LOG.info("Summary JSON:  %s", summary_json)
    return df, totals


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch audio-loss report (NBU lounge/sleep + Jamail clinic; split/decode/merge → GAPFILL → analyze)."
    )
    p.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root like /mnt/datalake/data/TRBD-53761",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output dir for work/, raw/, reports/, summary/",
    )
    p.add_argument(
        "--site",
        choices=SITE_CHOICES,
        default=DEFAULT_SITE,
        help="Site preset (nbu_lounge, nbu_sleep, or jamail)",
    )
    p.add_argument(
        "--patients",
        nargs="*",
        default=["TRBD001", "TRBD002"],
        help="Patients to include (default TRBD001 TRBD002)",
    )
    p.add_argument(
        "--fs", type=int, default=44100, help="Audio sample rate Hz (default 44100)"
    )
    p.add_argument(
        "--prefilter",
        action="store_true",
        help="Enable anomaly prefilter before analysis",
    )
    p.add_argument(
        "--max-fwd-delta",
        type=int,
        default=200,
        help="Prefilter MAX_FWD_DELTA; use -1 for no limit",
    )
    p.add_argument(
        "--local-window",
        type=int,
        default=3,
        help="Neighbors per side for local median (analyzer param)",
    )
    p.add_argument(
        "--top", type=int, default=12, help="Top gaps to duplicate into per-audio JSON"
    )
    # Split/Decode params
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decoder threshold (tuned per site preset)",
    )
    p.add_argument(
        "--split-chunk-seconds",
        type=int,
        default=3600,
        help="MP3 split chunk size (seconds)",
    )
    p.add_argument(
        "--split-overwrite",
        action="store_true",
        help="Allow overwriting chunk wavs / pseudo chunk",
    )
    p.add_argument(
        "--split-clean",
        action="store_true",
        help="Delete matching chunks before splitting",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _build_cli()
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s"
    )

    max_delta = None if int(args.max_fwd_delta) < 0 else int(args.max_fwd_delta)

    try:
        run_batch(
            root=args.root,
            out_dir=args.out_dir,
            site=args.site,
            fs=int(args.fs),
            prefilter=bool(args.prefilter),
            max_fwd_delta=max_delta,
            local_window=int(args.local_window),
            top=int(args.top),
            threshold=float(args.threshold),
            split_chunk_seconds=int(args.split_chunk_seconds),
            split_overwrite=bool(args.split_overwrite),
            split_clean=bool(args.split_clean),
            patients=args.patients,
        )
        return 0
    except Exception as e:
        LOG.error("Failed: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
