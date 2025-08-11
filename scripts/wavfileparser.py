import wave
import contextlib
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np


# ---- Site presets from lab findings (flip + nominal window length in samples) ----
SITE_PRESETS: Dict[str, Dict[str, object]] = {
    # Jamail: clean forward decoding, ~231 samples per 5-byte block at 44.1 kHz
    "jamail": {"flip": False, "window_samples": 231},
    # NBU sleep room: decode backwards in MATLAB doc; here we auto-try flip
    "nbu_sleep": {"flip": True, "window_samples": 536},
    # NBU lounge: longer window due to extra idle/spacing
    "nbu_lounge": {"flip": True, "window_samples": 646},
}


def _detect_start_edges(
    signal: np.ndarray,
    fs: int,
    baud: int,
    diff_thresh: float,
    min_gap_bytes: bool = True,
) -> List[int]:
    """
    Detect UART start-bit falling edges from an analog waveform using first-difference.

    We gate edges so we don't re-trigger inside the same byte. If min_gap_bytes=True,
    require at least ~1 byte (10 bits) spacing between accepted edges.
    """
    diff = np.diff(signal)
    falling = np.where(diff < -abs(diff_thresh))[0]
    spb = int(round(fs / baud))  # samples per bit
    min_gap = 10 * spb if min_gap_bytes else spb

    starts: List[int] = []
    last = -min_gap
    for idx in falling:
        if idx - last >= min_gap:
            starts.append(idx)
            last = idx
    return starts


def _decode_uart_byte(
    signal: np.ndarray, start_idx: int, fs: int, baud: int
) -> Optional[int]:
    """
    Decode a single 8N1 UART byte starting from a *falling* start bit edge.
    Returns 0..255 or None if start/stop checks fail.
    """
    spb = int(round(fs / baud))
    mid = start_idx + spb // 2
    if mid >= len(signal) or signal[mid] > 0:  # start bit should be low
        return None

    val = 0
    for bit in range(8):
        sample_point = start_idx + (bit + 1) * spb + spb // 2
        if sample_point >= len(signal):
            return None
        bit_level = 1 if signal[sample_point] > 0 else 0
        val |= bit_level << bit  # LSB first

    stop_mid = start_idx + 9 * spb + spb // 2
    if stop_mid < len(signal) and signal[stop_mid] < 0:  # stop bit should be high
        return None
    return val


@dataclass
class DecodeStats:
    bytes_total: int
    starts_total: int
    flips: bool
    best_offset: int
    monotonic_span: int


class WavSerialDecoder:
    """
    Decode Arduino serial bytes (8N1) embedded in .wav, then assemble 32-bit frame IDs
    from 5x7-bit chunks (lab protocol). Includes lab findings:
      - Site-specific flip option (NBU rooms),
      - Known 5-byte frame, monotonic counter (+1 per frame),
      - Robust start-edge spacing (≥1 byte),
      - Frame-boundary realignment by maximizing monotonic run.
    Also supports the lab's MATLAB **block sampler** path with fixed transition points
    and offsets (no UART), which we call the *block method*.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.sample_rate: int = 0
        self.n_channels: int = 0
        self.sampwidth: int = 0
        # audio as shape (N, C)
        self.audio: np.ndarray = np.empty((0,), dtype=np.float32)
        self._read_wav()

    # ---------------------- I/O ----------------------
    def _read_wav(self) -> None:
        with contextlib.closing(wave.open(self.filepath, "rb")) as wf:
            self.sample_rate = wf.getframerate()
            self.n_channels = wf.getnchannels()
            self.sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Handle common widths; extend here if you encounter 24-bit
        if self.sampwidth == 2:
            dtype = np.int16
            data = np.frombuffer(raw, dtype=dtype)
            norm = float(np.iinfo(dtype).max)
        elif self.sampwidth == 1:
            dtype = np.uint8
            data = np.frombuffer(raw, dtype=dtype).astype(np.int16) - 128  # center
            norm = 127.0
        else:
            # Fallback: interpret as 16-bit
            dtype = np.int16
            data = np.frombuffer(raw, dtype=dtype)
            norm = float(np.iinfo(dtype).max)

        # Expect mono (1 channel) audio for simplicity
        if self.n_channels != 1:
            raise ValueError(f"Expected mono audio (1 channel), got {self.n_channels}")
        data = data.reshape(-1)
        self.audio = (data.astype(np.float32) / norm).astype(np.float32)

    # ---------------------- Core decode ----------------------
    def _decode_bytes_once(
        self,
        baud: int = 9600,
        diff_thresh: Optional[float] = None,
        invert: bool = False,
    ) -> Tuple[List[int], int]:
        sig = self.audio
        sig = -sig if invert else sig
        if diff_thresh is None:
            # Make a small fraction of signal range; diff shrinks amplitude roughly
            diff_thresh = 0.05 * (sig.max() - sig.min())
            if diff_thresh <= 0:
                diff_thresh = 0.01
        starts = _detect_start_edges(
            sig, self.sample_rate, baud, diff_thresh, min_gap_bytes=True
        )
        out: List[int] = []
        for s in starts:
            b = _decode_uart_byte(sig, s, self.sample_rate, baud)
            if b is not None:
                out.append(b)
        return out, len(starts)

    def decode_bytes(
        self,
        baud: int = 9600,
        diff_thresh: Optional[float] = None,
        site: Optional[str] = None,
    ) -> Tuple[List[int], DecodeStats]:
        """
        Try normal and flipped polarity; pick the one yielding more *valid* bytes.
        If a site preset is provided, respect its flip preference as a bias.
        """
        prefer_flip = SITE_PRESETS.get(site, {}).get("flip", None) if site else None

        # Try both polarities
        out_a, starts_a = self._decode_bytes_once(baud, diff_thresh, invert=False)
        out_b, starts_b = self._decode_bytes_once(baud, diff_thresh, invert=True)

        # Choose according to site preference, else larger decoded count
        if prefer_flip is True:
            primary, starts, flipped = (out_b, starts_b, True)
            secondary = out_a
        elif prefer_flip is False:
            primary, starts, flipped = (out_a, starts_a, False)
            secondary = out_b
        else:
            if len(out_b) > len(out_a):
                primary, starts, flipped = (out_b, starts_b, True)
                secondary = out_a
            else:
                primary, starts, flipped = (out_a, starts_a, False)
                secondary = out_b

        # If extremely few bytes, fall back to the other polarity
        chosen = primary if len(primary) >= max(20, len(secondary)) else secondary
        flipped = flipped if chosen is primary else not flipped

        # Align to 5-byte frame boundary and compute stats
        best_offset, span = self._find_best_frame_offset(chosen)
        stats = DecodeStats(
            bytes_total=len(chosen),
            starts_total=starts_a + starts_b,
            flips=flipped,
            best_offset=best_offset,
            monotonic_span=span,
        )
        return chosen, stats

    @staticmethod
    def _find_best_frame_offset(byte_stream: List[int]) -> Tuple[int, int]:
        """
        Slide offsets 0..4 and pick the one that creates the longest *strictly
        monotonic-by-1* run of reconstructed 32-bit counts.
        Returns (best_offset, monotonic_span).
        """

        def counts_from_offset(off: int) -> List[int]:
            counts: List[int] = []
            i = off
            while i + 4 < len(byte_stream):
                val = 0
                for j in range(5):
                    val |= (byte_stream[i + j] & 0x7F) << (7 * j)
                counts.append(val)
                i += 5
            return counts

        best_off, best_span = 0, 0
        for off in range(5):
            cts = counts_from_offset(off)
            if not cts:
                continue
            # longest consecutive +1 span
            span = 1
            curr = 1
            for k in range(1, len(cts)):
                if cts[k] - cts[k - 1] == 1:
                    curr += 1
                    span = max(span, curr)
                else:
                    curr = 1
            if span > best_span:
                best_span, best_off = span, off
        return best_off, best_span

    def reconstruct_counts(self, bytes_out: List[int], offset: int = 0) -> List[int]:
        counts: List[int] = []
        i = offset
        while i + 4 < len(bytes_out):
            val = 0
            for j in range(5):
                val |= (bytes_out[i + j] & 0x7F) << (7 * j)
            counts.append(val)
            i += 5
        return counts

    # ---------------------- High-level API ----------------------
    def parse_counts(
        self,
        baud: int = 9600,
        diff_thresh: Optional[float] = None,
        site: Optional[str] = None,
        method: str = "auto",
    ) -> Tuple[List[int], DecodeStats]:
        """Convenience: decode bytes → realign → counts, with site-specific hints."""
        if method in ("uart", "auto"):
            bytes_out, stats = self.decode_bytes(
                baud=baud,
                diff_thresh=diff_thresh,
                site=site,
            )
            counts_uart = self.reconstruct_counts(bytes_out, offset=stats.best_offset)
        else:
            counts_uart, stats = [], DecodeStats(0, 0, False, 0, 0)

        if method in ("block", "auto"):
            counts_blk, stats_blk = self.decode_by_block(
                site=site,
            )
        else:
            counts_blk, stats_blk = [], DecodeStats(0, 0, False, 0, 0)

        # Choose the better result: prefer longer monotonic span, then length
        cand = [
            (counts_uart, stats),
            (counts_blk, stats_blk),
        ]
        cand.sort(key=lambda t: (t[1].monotonic_span, len(t[0])), reverse=True)
        return cand[0]

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

    # ---------------------- Block sampler (MATLAB port) ----------------------
    def decode_by_block(
        self,
        site: Optional[str] = None,
        threshold: float = 0.5,
    ) -> Tuple[List[int], DecodeStats]:
        """
        Port of the lab MATLAB method using fixed window and bit offsets.
        Works on a binarized channel with optional global flip defined by site preset.
        Steps:
          1) Normalize channel to [0,1] and binarize with threshold
          2) (Optional) flip full signal per site
          3) Scan for low (0) start; on hit, take a window of `window_samples`
          4) Within the window, sample bits at `transition_points + bit_offsets`
          5) Build 5 bytes (MSB/LSB as in MATLAB), concatenate [b5..b1], to integer
          6) Advance index by `block_stride`
        Returns counts and simple stats (monotonic span computed from +1 deltas).
        """
        cfg = self._site_block_cfg(site)
        sig = self.audio
        # normalize to [0,1]
        s_min, s_max = float(sig.min()), float(sig.max())
        if s_max <= s_min:
            return [], DecodeStats(0, 0, False, 0, 0)
        sig01 = (sig - s_min) / (s_max - s_min)
        bin_sig = (sig01 > threshold).astype(np.uint8)

        if cfg["flip_signal"]:
            bin_sig = bin_sig[::-1]

        N = bin_sig.size
        W = cfg["window_samples"]
        stride = cfg["block_stride"]
        trans = cfg["transition_points"]  # already 0-based
        offs = cfg["bit_offsets"]  # already 0-based (8 items)

        frames: List[int] = []
        starts = 0
        i = 0
        while i < N:
            if bin_sig[i] == 1:
                i += 1
                continue
            # start at low
            starts += 1
            if i + W > N:
                break
            win = bin_sig[i : i + W]
            if cfg["flip_window"]:
                win = win[::-1]
            # sample 5 bytes × 8 bits
            bytes5 = []
            for t in trans:
                # MATLAB keeps 8 taps but drops the last bit (1:end-1) → 7-bit payload
                bits = [int(win[t + o]) for o in offs[:-1]]  # drop last offset
                bits = bits[::-1]  # MATLAB used flip(...,2) before conversion
                # convert to integer from bit list (MSB first) → 7-bit value 0..127
                bval = 0
                for b in bits:
                    bval = (bval << 1) | b
                bytes5.append(bval)
            # concat byte5..byte1 into a big integer (like MATLAB strcat order)
            val = 0
            for b in bytes5[::-1]:  # bytes5 = [b1..b5]; want [b5..b1]
                val = (val << 7) | b  # concatenate 5×7-bit → 35-bit counter
            frames.append(val)
            i += stride

        # MATLAB flips the final serial_id vector
        frames = frames[::-1]

        # compute monotonic +1 span
        span = 1
        cur = 1
        for k in range(1, len(frames)):
            if frames[k] - frames[k - 1] == 1:
                cur += 1
                span = max(span, cur)
            else:
                cur = 1

        stats = DecodeStats(
            bytes_total=len(frames) * 5,
            starts_total=starts,
            flips=cfg["flip_signal"],
            best_offset=0,
            monotonic_span=span,
        )
        return frames, stats

    def _site_block_cfg(self, site: Optional[str]) -> Dict[str, object]:
        # Defaults based on Jamail MATLAB code
        if site == "nbu_sleep":
            flip = True
        elif site == "nbu_lounge":
            flip = True
        else:
            flip = (
                True  # Jamail script flipped full signal, then per-window flipped back
            )
        return {
            "flip_signal": flip,
            "flip_window": True,  # matches MATLAB: flip(binary_signal) then flip(window)
            "window_samples": SITE_PRESETS.get(site or "jamail", {}).get(
                "window_samples", 231
            ),
            "block_stride": 1100,
            # Convert MATLAB 1-based to 0-based
            "transition_points": [x - 1 for x in [6, 53, 100, 147, 194]],
            "bit_offsets": [x - 1 for x in [4, 9, 14, 19, 23, 28, 33, 37]],
        }


if __name__ == "__main__":
    dec = WavSerialDecoder(
        "/home/auto/CODE/utils/video-sync-nbu/data/jamil_exampe/AUDIO/VideoTest03062025-03.wav"
    )
    counts, stats = dec.parse_counts(site="jamail", method="block")
    print(stats)
    print(counts[:10])
    print(counts[-10:])
