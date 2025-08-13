import os
import csv
import wave
import contextlib
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np

"""
Clean, MATLAB-style block decoder for frame IDs embedded in audio.
Supports WAV (strict mono) and MP3 (via pydub+ffmpeg). Adds CSV export of decoded serials.

Usage
-----
python wav_serial_decoder_with_mp3_and_csv.py /path/to/audio.(wav|mp3) \
    --site jamail \
    --threshold 0.5 \
    --csv out.csv

CSV format
----------
Columns: file,site,sample_rate,channels,index,serial,start_sample,end_sample
Each row corresponds to one decoded serial value in chronological order.

*** Minimal update: ***
- Along with each decoded frame, also record its start/end sample indices (in the
  original, non-flipped signal). Indices are start inclusive, end exclusive.
"""

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
        "window_samples": 536,  # empirically observed at NBU sleep
        "block_stride": 1100,  # keep same unless you have a measured stride
        "transition_points_1b": [6, 53, 100, 147, 194],
        "bit_offsets_1b": [4, 9, 14, 19, 23, 28, 33, 37],  # 8 taps → drop last = 7-bit
    },
    # NBU Lounge (empirical)
    "nbu_lounge": {
        "flip_signal": True,
        "flip_window": True,
        "window_samples": 646,  # empirically observed at NBU lounge
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

        if self.sampwidth == 2:
            dtype = np.int16
            data = np.frombuffer(raw, dtype=dtype)
            norm = float(np.iinfo(dtype).max)
        elif self.sampwidth == 1:
            dtype = np.uint8
            data = np.frombuffer(raw, dtype=dtype).astype(np.int16) - 128
            norm = 127.0
        else:
            # Fallback: treat as int16
            dtype = np.int16
            data = np.frombuffer(raw, dtype=dtype)
            norm = float(np.iinfo(dtype).max)

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

        *** Minimal update: also records per-frame sample indices (start inclusive, end exclusive)
        into `self.frame_ranges` in chronological order. ***
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
        ranges_tmp: List[Tuple[int, int]] = (
            []
        )  # (start_idx_current_space, end_idx_exclusive)
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
        self, out_path: str | Path, counts: List[int], site: str = ""
    ) -> Path:
        """Save decoded serials to CSV. Returns the output path.

        CSV columns: file,site,sample_rate,channels,index,serial,start_sample,end_sample
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = self.get_metadata()
        ranges = getattr(self, "frame_ranges", []) or []
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "file",
                    "site",
                    "sample_rate",
                    "channels",
                    "index",
                    "serial",
                    "start_sample",
                    "end_sample",
                ]
            )
            for i, val in enumerate(counts):
                s, e = ranges[i] if i < len(ranges) else ("", "")
                w.writerow(
                    [
                        meta["filepath"],
                        site,
                        meta["sample_rate"],
                        meta["channels"],
                        i,
                        int(val),
                        s,
                        e,
                    ]
                )
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


def _main() -> None:
    p = argparse.ArgumentParser(
        description="Decode frame IDs from WAV/MP3 and optionally save CSV."
    )
    p.add_argument("audio", help="Path to input audio (.wav or .mp3)")
    p.add_argument(
        "--site", default="jamail", help="Site preset (jamail|nbu_sleep|nbu_lounge)"
    )
    p.add_argument(
        "--threshold", type=float, default=0.5, help="Binarization threshold in [0,1]"
    )
    p.add_argument(
        "--csv", dest="csv_out", help="If set, save decoded serials to this CSV path"
    )
    args = p.parse_args()

    dec = WavSerialDecoder(args.audio)
    counts, stats = dec.parse_counts(site=args.site, threshold=args.threshold)

    # Console preview
    print(stats)
    if counts:
        print("first 10:", counts[:10])
        print("last 10 :", counts[-10:])
        # Optional peek at ranges
        print("ranges first 3:", dec.frame_ranges[:3])
        print("ranges last 3 :", dec.frame_ranges[-3:])
    else:
        print("No counts decoded.")

    if args.csv_out:
        out = dec.save_counts_csv(args.csv_out, counts, site=args.site)
        print(f"Saved {len(counts)} rows to {out}")


if __name__ == "__main__":
    _main()
