import wave
import contextlib
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np

"""
Clean, MATLAB-style block decoder for frame IDs embedded in audio.
- Mono WAV only (raises if not mono)
- Binarize @ 0.5
- Global flip of full binary stream
- For each detected low start, take a fixed window (default 231 samples)
- Flip the window, sample 5x(8 taps), drop last tap → 7-bit per byte
- Concatenate bytes as [b5‖b4‖b3‖b2‖b1] (each 7-bit) → 35-bit counter
- Advance by fixed stride (default 1100 samples)
- Flip final vector to restore chronological order
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
        "transition_points_1b": [
            6,
            53,
            100,
            147,
            194,
        ],
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
    """Decode frame IDs from mono WAV using a fixed-window, MATLAB-style block sampler.
    The algorithm samples hard-coded bit tap positions inside each window and
    concatenates five 7-bit bytes into a 35-bit counter. See `decode_by_block`.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.sample_rate: int = 0
        self.n_channels: int = 0
        self.sampwidth: int = 0
        self.audio: np.ndarray = np.empty((0,), dtype=np.float32)  # mono
        self._read_wav()

    # ---------------------- I/O ----------------------
    def _read_wav(self) -> None:
        """Read the WAV file into `self.audio` as float32 in [-1, 1].
        Enforces mono input (raises if `n_channels != 1`).
        """
        with contextlib.closing(wave.open(self.filepath, "rb")) as wf:
            self.sample_rate = wf.getframerate()
            self.n_channels = wf.getnchannels()
            self.sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if self.n_channels != 1:
            raise ValueError(f"Expected mono audio (1 channel), got {self.n_channels}")

        if self.sampwidth == 2:
            dtype = np.int16
            data = np.frombuffer(raw, dtype=dtype)
            norm = float(np.iinfo(dtype).max)
        elif self.sampwidth == 1:
            dtype = np.uint8
            data = np.frombuffer(raw, dtype=dtype).astype(np.int16) - 128
            norm = 127.0
        else:
            dtype = np.int16
            data = np.frombuffer(raw, dtype=dtype)
            norm = float(np.iinfo(dtype).max)

        self.audio = (data.astype(np.float32) / norm).astype(np.float32)

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
        """
        cfg = self._get_cfg(site)
        sig01 = self._normalize01(self.audio)
        if sig01 is None:
            return [], DecodeStats(0, 0, bool(cfg["flip_signal"]), 0, 0)
        bin_sig = self._binarize(sig01, threshold)
        bin_sig = self._maybe_flip(bin_sig, bool(cfg["flip_signal"]))

        N = bin_sig.size
        frames: List[int] = []
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
            i += cfg["stride"]

        frames = frames[::-1]  # restore chronological order
        span = self._longest_plus_one_span(frames)
        stats = DecodeStats(
            bytes_total=len(frames) * 5,
            starts_total=starts,
            flips=bool(cfg["flip_signal"]),
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


if __name__ == "__main__":
    audio = "/home/auto/CODE/utils/video-sync-nbu/data/jamil_exampe/AUDIO/VideoTest03062025-03.wav"
    site = "jamail"
    decoder = WavSerialDecoder(audio)
    counts, stats = decoder.parse_counts(site=site, threshold=0.5)
    print(stats)
    print(counts[:10])
    print(counts[-10:])
