#!/usr/bin/env python3
import os
import csv
import wave
import contextlib
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
from datetime import datetime
import json
import logging

from scripts.log.logutils import (
    configure_standalone_logging,
    log_context,
)

"""
Clean, MATLAB-style block decoder for frame IDs embedded in audio.
Supports WAV (strict mono) and MP3 (via pydub+ffmpeg). Exports decoded serials to CSV.

Usage
-----
# Single-file decode (writes raw.csv + raw_info.txt)
python wavfileparser.py /path/to/audio.(wav|mp3) \
    --site jamail \
    --threshold 0.5 \
    --outdir /path/to/output_dir

# Batch decode over split WAVs like '*-03-0XX.wav'
# If you also pass a splitter manifest JSON, start/end samples become ABSOLUTE
# offsets relative to the original MP3; otherwise they are per-file (relative).
python wavfileparser.py \
    --decode-split-dir /path/to/splits \
    --outdir /path/to/csv_out \
    --site jamail \
    --pattern "*-03-[0-9][0-9][0-9].wav" \
    --manifest /path/to/<stem>_manifest.json   # optional, enables absolute indices

CSV format
----------
Columns: serial,start_sample,end_sample
- Single-file mode: indices are relative to the provided file.
- Batch mode:
    • without --manifest → per-file (relative) indices
    • with --manifest    → ABSOLUTE indices using each chunk’s start_sample from the manifest

Recent additions
----------------
1) Per-frame sample ranges (start/end indices) recorded to CSV.
2) **Size guards**:
   - Hard stop for MP3 near ≥4GB (pydub limitation).
   - Soft cap for any input via env VIDEOSYNC_DECODE_MAX_BYTES (default: 2 GiB).
3) **Output directory**:
   - Use --outdir to choose where outputs are written. Defaults to the audio file's directory.
   - Output CSV name is fixed to raw.csv; a summary text raw_info.txt is also written (single-file mode).
4) **Manifest support in batch mode**:
   - Pass --manifest pointing to the splitter’s JSON to write absolute sample indices.
"""

logger = logging.getLogger(__name__)

# ---- Size guard thresholds ----
MP3_PYDUB_SIZE_HARD_LIMIT = 4_000_000_000  # ~4GB; pydub/ffmpeg temp WAV header limit
MAX_DECODE_BYTES_DEFAULT = 2 * 1024**3  # 2 GiB soft cap unless overridden by env

# ---- Block presets (from lab MATLAB) ----
BLOCK_PRESETS: Dict[str, Dict[str, object]] = {
    # Jamail site defaults
    "jamail": {
        "flip_signal": True,  # binary_signal = flip(binary_signal)
        "flip_window": True,  # win = flip(win)
        "window_samples": 231,  # size of one serial burst window
        "block_stride": 1100,  # hop size to next window candidate
        # MATLAB defined in 1-based indexing; convert to 0-based in code
        "transition_points_1b": [6, 53, 100, 147, 194],
        "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],  # 8 taps → drop last = 7-bit
    },
    # NBU Sleep room (empirical)
    "nbu_sleep": {
        "flip_signal": True,
        "flip_window": True,
        "window_samples": 231,  # empirically observed at NBU sleep
        "block_stride": 1100,  # keep same unless you have a measured stride
        "transition_points_1b": [6, 53, 100, 147, 194],
        "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],  # 8 taps → drop last = 7-bit
    },
    # NBU Lounge (empirical)
    "nbu_lounge": {
        "flip_signal": True,
        "flip_window": True,
        "window_samples": 231,  # empirically observed at NBU lounge
        "block_stride": 1100,  # adjust if your coworkers logged a different hop
        "transition_points_1b": [6, 53, 100, 147, 194],
        "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],
    },
}


@dataclass
class DecodeStats:
    bytes_total: int
    starts_total: int
    flips: bool
    best_offset: int
    monotonic_span: int


class WavSerialDecoder:
    """Decode frame IDs from mono audio (WAV or MP3) using a fixed-window, MATLAB-style sampler.

    File handling
    -------------
    - `.wav`: loaded via Python's `wave` module; **must be mono** (raises otherwise).
    - `.mp3`: loaded via **pydub + ffmpeg** (single enforced path). Any multi-channel MP3
      is **downmixed to mono** (mean over channels) for decoding.

    Extras
    ------
    - `save_counts_csv(path, counts, site)` writes decoded serials + indices to CSV.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.sample_rate: int = 0
        self.n_channels: int = 0
        self.sampwidth: int = 0
        self.audio: np.ndarray = np.empty(
            (0,), dtype=np.float32
        )  # mono float32 in [-1, 1]
        # holds (start_sample, end_sample) for the last decode_by_block() call, chronological
        self.frame_ranges: List[Tuple[int, int]] = []
        self._read_audio()

    # ---------------------- I/O (WAV + MP3) ----------------------
    def _read_audio(self) -> None:
        """Unified reader: WAV via `wave` (strict mono), MP3 via pydub+ffmpeg.
        Produces self.audio as 1-D float32 mono in [-1, 1].
        """
        ext = os.path.splitext(self.filepath)[1].lower()
        self._size_guard(ext)
        if ext == ".wav":
            self._read_wav_strict()
        elif ext == ".mp3":
            self._read_mp3_with_pydub()
            # Downmix MP3 to mono for the decoder pipeline
            if self.audio.ndim == 2:
                self.audio = self.audio.mean(axis=1).astype(np.float32)
                self.n_channels = 1
            elif self.audio.ndim == 1:
                self.n_channels = 1
            else:
                raise ValueError("Decoded MP3 has unexpected shape.")
        else:
            raise ValueError(
                f"Unsupported format {ext}. Only .wav and .mp3 are supported."
            )

        # Final clamp (safety)
        self.audio = np.clip(self.audio, -1.0, 1.0).astype(np.float32)

    def _size_guard(self, ext: str) -> None:
        """Refuse to decode files that are too large to be handled reliably in-process.
        - Hard stop for MP3 near 4GB (pydub limitation).
        - Soft cap for any input (default 2 GiB). Override with env VIDEOSYNC_DECODE_MAX_BYTES.
        Suggests using --split-minutes.
        """
        try:
            fbytes = os.path.getsize(self.filepath)
        except OSError:
            return
        # Env override
        env = os.getenv("VIDEOSYNC_DECODE_MAX_BYTES")
        try:
            soft_cap = int(env) if env else MAX_DECODE_BYTES_DEFAULT
        except Exception:
            soft_cap = MAX_DECODE_BYTES_DEFAULT

        if ext == ".mp3" and fbytes >= MP3_PYDUB_SIZE_HARD_LIMIT:
            raise RuntimeError(
                "Input MP3 is ~>=4GB; direct decoding is unsafe. "
                "Run splitting first, e.g.: \n"
                "  python wavfileparser.py <file>.mp3 --split-minutes 5\n"
                "This writes <stem>_chunkNNN.wav files ready for decoding."
            )
        if fbytes >= soft_cap:
            raise RuntimeError(
                f"Input file is {fbytes} bytes, exceeding safe decode cap {soft_cap} bytes. "
                "Split into chunks first, e.g.:\n"
                "  python wavfileparser.py <file> --split-minutes 10"
            )

    def _read_wav_strict(self) -> None:
        """Original WAV reader; enforces mono input and converts to float32 in [-1, 1]."""
        with contextlib.closing(wave.open(self.filepath, "rb")) as wf:
            self.sample_rate = wf.getframerate()
            self.n_channels = wf.getnchannels()
            self.sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if self.n_channels != 1:
            raise ValueError(f"Expected mono WAV (1 channel), got {self.n_channels}")

        if self.sampwidth == 1:
            dtype = np.uint8
            data = np.frombuffer(raw, dtype=dtype).astype(np.int16) - 128
            norm = 127.0
        elif self.sampwidth == 2:
            dtype = np.int16
            data = np.frombuffer(raw, dtype=dtype)
            norm = float(np.iinfo(dtype).max)
        elif self.sampwidth == 3:
            # 24-bit little-endian PCM: pack 3 bytes into signed int32
            raw_bytes = np.frombuffer(raw, dtype=np.uint8)
            if raw_bytes.size % 3 != 0:
                raise ValueError("24-bit WAV payload has incomplete frames.")
            triples = raw_bytes.reshape(-1, 3)
            ints = (
                triples[:, 0].astype(np.int32)
                | (triples[:, 1].astype(np.int32) << 8)
                | (triples[:, 2].astype(np.int32) << 16)
            )
            sign_bit = 1 << 23
            ints = np.where(ints & sign_bit, ints - (1 << 24), ints)
            data = ints
            norm = float(sign_bit - 1)
        elif self.sampwidth == 4:
            dtype = np.int32
            data = np.frombuffer(raw, dtype=dtype)
            norm = float(np.iinfo(dtype).max)
        else:
            raise ValueError(f"Unsupported sample width: {self.sampwidth} bytes")

        self.audio = (data.astype(np.float32) / norm).astype(np.float32)

    def _read_mp3_with_pydub(self) -> None:
        """MP3 reader using pydub+ffmpeg (single enforced path). Raises if unavailable.

        Sets:
            self.sample_rate, self.n_channels, self.sampwidth, self.audio (shape (S, C) float32)
        """
        try:
            from pydub import AudioSegment
            from pydub.utils import which
        except Exception as e:
            raise ImportError(
                "MP3 support requires 'pydub'. Install with: pip install pydub"
            ) from e

        ffmpeg_path = which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError(
                "MP3 decoding requires 'ffmpeg' on PATH. Install it (e.g., apt/yum/brew) "
                "and ensure the 'ffmpeg' binary is discoverable."
            )

        try:
            seg = AudioSegment.from_file(self.filepath)  # uses ffmpeg under the hood
        except Exception as e:
            raise RuntimeError(f"ffmpeg failed to decode MP3: {e}") from e

        self.sample_rate = int(seg.frame_rate)
        self.n_channels = int(seg.channels)
        self.sampwidth = int(seg.sample_width)  # bytes per sample per channel

        # Convert to float32 array in [-1, 1). pydub returns interleaved integer samples.
        arr = np.array(seg.get_array_of_samples())
        if self.n_channels > 1:
            arr = arr.reshape(-1, self.n_channels)
        else:
            arr = arr.reshape(-1, 1)

        bits = 8 * self.sampwidth
        # Denominator for signed PCM to map to [-1, 1]
        denom = float((2 ** (bits - 1)) - 1) if bits > 8 else 127.0
        data = (arr.astype(np.float32) / denom).astype(np.float32)
        self.audio = data  # (S, C) float32

    # ---------------------- Helpers ----------------------
    def _get_cfg(self, site: str) -> Dict[str, object]:
        """Return site config with 0-based indices and precomputed 7-bit offsets."""
        cfg = BLOCK_PRESETS.get(site, BLOCK_PRESETS["jamail"]).copy()
        cfg["trans"] = [x - 1 for x in cfg["transition_points_1b"]]
        offs8 = [x - 1 for x in cfg["bit_offsets_1b"]]
        cfg["offs7"] = offs8[:-1]
        cfg["W"] = int(cfg["window_samples"])
        cfg["stride"] = int(cfg["block_stride"])
        return cfg

    def _normalize01(self, sig: np.ndarray) -> Optional[np.ndarray]:
        """Normalize 1-D signal to [0,1]; return None if the signal is flat."""
        s_min = float(sig.min())
        s_max = float(sig.max())
        if s_max <= s_min:
            return None
        return (sig - s_min) / (s_max - s_min)

    def _binarize(self, sig01: np.ndarray, threshold: float) -> np.ndarray:
        """Threshold normalized signal to {0,1} (uint8)."""
        return (sig01 > threshold).astype(np.uint8)

    def _maybe_flip(self, v: np.ndarray, do_flip: bool) -> np.ndarray:
        """Reverse order if `do_flip` is True; no-op otherwise."""
        return v[::-1] if do_flip else v

    def _sample_window(
        self, win: np.ndarray, trans: List[int], offs7: List[int]
    ) -> Optional[List[int]]:
        """Sample 5 bytes x 7 bits from `win`; return list of five values in [0,127].
        Returns None if any tap would be out-of-range.
        """
        bytes5: List[int] = []
        for t in trans:
            try:
                bits = [int(win[t + o]) for o in offs7]
            except IndexError:
                return None
            bits = bits[::-1]  # bit-order flip (MSB..LSB) like MATLAB's flip(...,2)
            bval = 0
            for b in bits:
                bval = (bval << 1) | b
            bytes5.append(bval)
        return bytes5

    def _concat_bytes(self, bytes5: List[int]) -> int:
        """Concatenate as b5‖b4‖b3‖b2‖b1 using 7 bits per byte → 35-bit integer."""
        val = 0
        for b in bytes5[::-1]:
            val = (val << 7) | b
        return val

    def _longest_plus_one_span(self, vals: List[int]) -> int:
        """Length of the longest contiguous run where successive deltas equal +1."""
        if not vals:
            return 0
        best = cur = 1
        for k in range(1, len(vals)):
            if vals[k] - vals[k - 1] == 1:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 1
        return best

    # ---------------------- Block sampler (MATLAB port) ----------------------
    def decode_by_block(
        self,
        site: str = "jamail",
        threshold: float = 0.5,
    ) -> Tuple[List[int], DecodeStats]:
        """Decode frame IDs with the MATLAB block method.

        Steps:
          1) Normalize audio to [0,1]; binarize at `threshold`.
          2) Optionally flip the whole binary stream (site preset).
          3) Scan for low (0) → take window of `W` samples; optionally flip window.
          4) Within the window, sample 5x7 bits at fixed anchors/offsets.
          5) Concatenate as b5‖b4‖b3‖b2‖b1 (7 bits each) to a 35-bit ID.
          6) Advance by fixed stride; repeat. Flip final list to chronological order.

        Also records per-frame sample indices (start inclusive, end exclusive)
        into `self.frame_ranges` in chronological order.
        """
        cfg = self._get_cfg(site)
        sig01 = self._normalize01(self.audio)
        if sig01 is None:
            # Reset frame ranges when nothing decoded
            self.frame_ranges = []
            return [], DecodeStats(0, 0, bool(cfg["flip_signal"]), 0, 0)
        bin_sig = self._binarize(sig01, threshold)
        flipped = bool(cfg["flip_signal"])  # remember if we flipped the stream
        bin_sig = self._maybe_flip(bin_sig, flipped)

        N = bin_sig.size
        frames: List[int] = []
        ranges_tmp: List[Tuple[int, int]] = []
        starts = 0
        i = 0
        while i < N:
            if bin_sig[i] == 1:
                i += 1
                continue
            starts += 1
            if i + cfg["W"] > N:
                break
            win = bin_sig[i : i + cfg["W"]]
            win = self._maybe_flip(win, bool(cfg["flip_window"]))
            bytes5 = self._sample_window(win, cfg["trans"], cfg["offs7"])
            if bytes5 is None:
                break
            frames.append(self._concat_bytes(bytes5))
            ranges_tmp.append((i, i + cfg["W"]))
            i += cfg["stride"]

        # Map ranges back to original orientation, then reverse to chronological to match `frames`
        if flipped:
            mapped = [(N - e, N - s) for (s, e) in ranges_tmp]
        else:
            mapped = ranges_tmp

        frames = frames[::-1]  # restore chronological order
        self.frame_ranges = mapped[::-1]

        span = self._longest_plus_one_span(frames)
        stats = DecodeStats(
            bytes_total=len(frames) * 5,
            starts_total=starts,
            flips=flipped,
            best_offset=0,
            monotonic_span=span,
        )
        return frames, stats

    # ---------------------- High-level API ----------------------
    def parse_counts(
        self, site: str = "jamail", threshold: float = 0.5
    ) -> Tuple[List[int], DecodeStats]:
        """Decode with MATLAB-style block method only."""
        return self.decode_by_block(site=site, threshold=threshold)

    def save_counts_csv(
        self,
        out_path: str | Path,
        counts: List[int],
        site: str = "",
        *,
        offset_samples: int = 0,
    ) -> Path:
        """Save decoded serials to CSV. Returns the output path.

        CSV columns: serial,start_sample,end_sample
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ranges = getattr(self, "frame_ranges", []) or []
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["serial", "start_sample", "end_sample"])
            for i, val in enumerate(counts):
                s, e = ranges[i] if i < len(ranges) else ("", "")
                # apply absolute offset if provided
                s = (s + offset_samples) if s != "" else s
                e = (e + offset_samples) if e != "" else e
                w.writerow([int(val), s, e])
        return out

    # ---------------------- Utilities ----------------------
    def get_metadata(self) -> dict:
        return {
            "filepath": self.filepath,
            "sample_rate": self.sample_rate,
            "channels": self.n_channels,
            "sample_width": self.sampwidth,
            "num_samples": int(self.audio.shape[0]),
        }

    def __repr__(self) -> str:
        dur = self.audio.shape[0] / self.sample_rate if self.sample_rate else 0.0
        return f"<WavSerialDecoder {self.filepath!r}: {self.n_channels}ch, {self.sample_rate}Hz, {dur:.2f}s>"


def decode_to_raw(
    audio: str | Path,
    outdir: str | Path | None = None,
    *,
    site: str = "jamail",
    threshold: float = 0.5,
) -> Tuple[Path, Path, List[int], DecodeStats]:
    """
    Public API: decode an audio file and write:
      - raw.csv : decoded serials with start/end sample indices
      - raw_info.txt : summary (audio path, sample rate, duration (s), processed timestamp)

    Returns
    -------
    (csv_path, txt_path, counts, stats)
    """
    audio_path = Path(audio)
    out_dir = Path(outdir) if outdir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Decoding %s (site=%s, threshold=%.3f)", audio_path.name, site, threshold
    )

    csv_path = out_dir / "raw.csv"
    txt_path = out_dir / "raw_info.txt"

    dec = WavSerialDecoder(str(audio_path))
    counts, stats = dec.parse_counts(site=site, threshold=threshold)
    dec.save_counts_csv(
        csv_path, counts, site=site
    )  # offset default keeps current behavior

    # Summary text
    duration_s = (
        dec.audio.shape[0] / dec.sample_rate if getattr(dec, "sample_rate", 0) else 0.0
    )
    summary = [
        f"audio_input: {audio_path}",
        f"sample_rate_hz: {dec.sample_rate}",
        f"duration_s: {duration_s:.6f}",
        f"processed_at: {datetime.now().isoformat(timespec='seconds')}",
    ]
    txt_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    logger.info(
        "Done: %d frames, sr=%d Hz, dur=%.3fs, span(+1)=%d → %s ; %s",
        len(counts),
        dec.sample_rate,
        duration_s,
        getattr(stats, "monotonic_span", 0),
        csv_path.name,
        txt_path.name,
    )

    return csv_path, txt_path, counts, stats


def _load_manifest_offsets(manifest: str | Path) -> Dict[str, int]:
    """
    Load manifest JSON and return {filename -> start_sample} mapping.
    Expects entries like: {"file": "name.wav", "start_sample": 123, ...} under "segments".
    """
    p = Path(manifest)
    data = json.loads(p.read_text(encoding="utf-8"))
    segs = data.get("segments", []) or []
    return {
        str(seg.get("file")): int(seg.get("start_sample", 0))
        for seg in segs
        if "file" in seg
    }


def decode_split_dir_to_csvs(
    split_dir: str | Path,
    outdir: str | Path,
    *,
    site: str = "jamail",
    threshold: float = 0.5,
    pattern: str = "*-03-[0-9][0-9][0-9].wav",
    manifest: str | Path | None = None,
) -> List[Path]:
    """
    Decode all split audio files in `split_dir` that match `pattern`
    (e.g., '*-03-0XX.wav' / '*-03-XXX.wav') and save one CSV per file
    with the **same base name** into `outdir`.

    The CSVs contain columns: serial,start_sample,end_sample.
    """
    in_dir = Path(split_dir)
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Batch decode: dir=%s, pattern=%s, manifest=%s",
        in_dir,
        pattern,
        manifest if manifest else "-",
    )

    wav_paths = sorted(in_dir.glob(pattern))
    logger.info("Found %d input file(s).", len(wav_paths))

    offsets: Dict[str, int] = _load_manifest_offsets(manifest) if manifest else {}
    outputs: List[Path] = []

    for wav_path in wav_paths:
        # stamp each file’s stem as seg for readability
        with log_context(seg=wav_path.stem, cam="-"):
            dec = WavSerialDecoder(str(wav_path))
            counts, _stats = dec.parse_counts(site=site, threshold=threshold)

            csv_out = out_dir / (wav_path.stem + ".csv")  # same name as input file
            base = wav_path.name
            offset = int(offsets.get(base, 0))
            logger.info("Decoding %s (offset=%d)", base, offset)
            dec.save_counts_csv(csv_out, counts, site=site, offset_samples=offset)
            logger.info("→ %s (%d rows)", csv_out.name, len(counts))

            outputs.append(csv_out)

    logger.info("Batch complete → wrote %d CSV file(s) into %s", len(outputs), out_dir)
    return outputs


# ---------------------- CLI ----------------------
def _main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Decode frame IDs from WAV/MP3 into CSV. "
            "Single-file mode (AUDIO provided) or batch mode (--decode-split-dir)."
        )
    )
    p.add_argument(
        "audio",
        nargs="?",
        help="Path to input audio (.wav or .mp3). Required unless --decode-split-dir is used.",
    )
    p.add_argument(
        "--site", required=True, help="Site preset (jamail|nbu_sleep|nbu_lounge)"
    )
    p.add_argument(
        "--threshold", type=float, default=0.5, help="Binarization threshold in [0,1]"
    )
    p.add_argument(
        "--outdir",
        help=(
            "Output directory. "
            "Single-file mode: where raw.csv/raw_info.txt go. "
            "Batch mode (--decode-split-dir): where per-file CSVs go."
        ),
    )
    p.add_argument(
        "--decode-split-dir",
        type=Path,
        help="Batch mode: directory with split WAVs (e.g., *-03-0XX.wav).",
    )
    p.add_argument(
        "--pattern",
        default="*-03-[0-9][0-9][0-9].wav",
        help="Glob pattern for batch mode (default: '*-03-XXX.wav').",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        help="Path to manifest JSON produced by the splitter; enables absolute sample indices in batch mode.",
    )
    args = p.parse_args()

    # --- Standalone console logging (no-op under driver) ---
    seg_hint = (
        (
            Path(args.decode_split_dir).name
            if args.decode_split_dir
            else Path(args.audio).stem
        )
        if (args.decode_split_dir or args.audio)
        else "-"
    )
    configure_standalone_logging(level="INFO", seg=seg_hint, cam="-")

    if args.decode_split_dir:
        # Batch mode
        if not args.outdir:
            p.error("--outdir is required when using --decode-split-dir")
        with log_context(seg=seg_hint, cam="-"):
            outputs = decode_split_dir_to_csvs(
                split_dir=args.decode_split_dir,
                outdir=args.outdir,
                site=args.site,
                threshold=args.threshold,
                pattern=args.pattern,
                manifest=args.manifest,
            )
            logger.info("Wrote %d CSV file(s) to %s", len(outputs), args.outdir)
        return

    # Single-file mode requires AUDIO
    if not args.audio:
        p.error("AUDIO is required unless --decode-split-dir is used")

    with log_context(seg=Path(args.audio).stem, cam="-"):
        out_csv, out_txt, counts, stats = decode_to_raw(
            args.audio, args.outdir, site=args.site, threshold=args.threshold
        )

        logger.info("Stats: %s", stats)
        if counts:
            logger.info("first 10: %s", counts[:10])
            logger.info("last 10 : %s", counts[-10:])
        else:
            logger.warning("No counts decoded.")
        logger.info("Saved %d rows to %s; summary → %s", len(counts), out_csv, out_txt)


if __name__ == "__main__":
    _main()
