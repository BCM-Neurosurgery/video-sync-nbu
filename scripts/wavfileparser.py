import wave
import contextlib
import numpy as np
from typing import Optional, List


def _compute_threshold(signal: np.ndarray) -> float:
    """
    Compute an adaptive threshold for edge detection in the waveform.
    Uses the midpoint between the max and min of the signal.

    Args:
        signal: Audio waveform as a 1D numpy array.

    Returns:
        Adaptive threshold value as float.
    """
    return float((signal.max() + signal.min()) / 2)


def _detect_edges(
    signal: np.ndarray, fs: int, baud: int, pos_thresh: float
) -> List[int]:
    """
    Detect rising edge indices in an AC-coupled audio signal by first differencing.

    Args:
        signal: 1D normalized audio waveform.
        fs: Sampling rate in Hz.
        baud: UART baud rate.
        pos_thresh: Threshold for positive peaks in the differential signal.

    Returns:
        List of sample indices corresponding to detected UART start bits.
    """
    diff = np.diff(signal)
    rising = np.where(diff > pos_thresh)[0]
    samples_per_bit = int(fs / baud)
    starts: List[int] = []
    last_idx = -samples_per_bit
    for idx in rising:
        if idx - last_idx >= samples_per_bit:
            starts.append(idx)
            last_idx = idx
    return starts


def _decode_uart_frame(
    signal: np.ndarray, start_idx: int, fs: int, baud: int
) -> Optional[int]:
    """
    Decode a single UART byte frame from an AC-coupled analog waveform.

    Args:
        signal: 1D normalized audio waveform.
        start_idx: Sample index of the detected start bit edge.
        fs: Sampling rate in Hz.
        baud: UART baud rate.

    Returns:
        The decoded byte (0-255), or None if invalid frame.
    """
    samples_per_bit = int(fs / baud)
    byte_val = 0
    start_sample = start_idx + samples_per_bit // 2
    if start_sample >= len(signal) or signal[start_sample] > 0:
        return None
    for bit in range(8):
        sample_point = start_idx + (bit + 1) * samples_per_bit + samples_per_bit // 2
        if sample_point >= len(signal):
            return None
        bit_level = 1 if signal[sample_point] > 0 else 0
        byte_val |= bit_level << bit
    stop_sample = start_idx + 9 * samples_per_bit + samples_per_bit // 2
    if stop_sample < len(signal) and signal[stop_sample] < 0:
        return None
    return byte_val


class WavSerialDecoder:
    """
    Decode a custom UART-like serial stream embedded in an AC-coupled .wav audio file.

    Attributes:
        filepath: Path to the WAV file.
        sample_rate: Sampling frequency.
        n_channels: Number of audio channels.
        sampwidth: Sample width in bytes.
        audio: Mono waveform normalized to [-1, 1].
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.sample_rate: int = 0
        self.n_channels: int = 0
        self.sampwidth: int = 0
        self.audio: np.ndarray = np.array([])
        self._read_wav()

    def _read_wav(self) -> None:
        """
        Load WAV metadata and audio samples, mix to mono if needed, and normalize.
        """
        try:
            with contextlib.closing(wave.open(self.filepath, "rb")) as wf:
                self.sample_rate = wf.getframerate()
                self.n_channels = wf.getnchannels()
                self.sampwidth = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
        except wave.Error as e:
            raise RuntimeError(f"Failed to read WAV file: {e}")
        dtype = np.int16 if self.sampwidth == 2 else np.uint8
        data = np.frombuffer(raw, dtype=dtype)
        if self.n_channels > 1:
            data = data.reshape(-1, self.n_channels).mean(axis=1)
        self.audio = data.astype(np.float32) / np.iinfo(dtype).max

    def decode_serial(
        self, baud: int = 9600, pos_thresh: Optional[float] = None
    ) -> List[int]:
        """
        Decode the embedded serial byte stream from the WAV audio.

        Args:
            baud: UART baud rate (default 9600).
            pos_thresh: Threshold for detecting rising edges; if None, computed adaptively.

        Returns:
            List of decoded byte values.
        """
        if pos_thresh is None:
            pos_thresh = _compute_threshold(self.audio) * 0.5
        starts = _detect_edges(self.audio, self.sample_rate, baud, pos_thresh)
        decoded: List[int] = []
        for idx in starts:
            byte = _decode_uart_frame(self.audio, idx, self.sample_rate, baud)
            if byte is not None:
                decoded.append(byte)
        return decoded

    def reconstruct_counts(self, bytes_out: List[int]) -> List[int]:
        """
        Reconstruct 32-bit counters sent as 5x7-bit chunks from the decoded bytes.

        Args:
            bytes_out: Flat list of decoded bytes.

        Returns:
            List of reconstructed 32-bit integer counters.
        """
        counts: List[int] = []
        for i in range(0, len(bytes_out) - 4, 5):
            val = 0
            for j in range(5):
                val |= (bytes_out[i + j] & 0x7F) << (7 * j)
            counts.append(val)
        return counts

    def parse_counts(
        self, baud: int = 9600, pos_thresh: Optional[float] = None
    ) -> List[int]:
        """
        Convenience method: decode UART bytes and reconstruct 32-bit counters in one call.

        Args:
            baud: UART baud rate (default 9600).
            pos_thresh: Threshold for rising-edge detection; if None, computed adaptively.

        Returns:
            List of reconstructed 32-bit integer counters.
        """
        bytes_out = self.decode_serial(baud=baud, pos_thresh=pos_thresh)
        return self.reconstruct_counts(bytes_out)

    def plot_waveform_interactive(
        self,
        output_html: Optional[str] = None,
        window_size: Optional[int] = None,
        start_sample: int = 0,
    ) -> None:
        """
        Plot the audio waveform as an interactive HTML plot using Plotly.

        Args:
            output_html: Path to save the HTML file. Defaults to '<base>_waveform.html'.
            window_size: Number of samples to plot. If None, plot full audio.
            start_sample: Sample index to start the window.
        """
        import plotly.graph_objs as go
        import plotly.offline as pyo
        import os

        total = len(self.audio)
        if start_sample < 0 or start_sample >= total:
            raise ValueError(f"start_sample {start_sample} out of bounds")
        end = start_sample + window_size if window_size else total
        end = min(end, total)
        window = self.audio[start_sample:end]

        trace = go.Scatter(y=window, mode="lines", name="Waveform")
        layout = go.Layout(
            title=f"Waveform: {os.path.basename(self.filepath)}",
            xaxis={"title": "Sample Index"},
            yaxis={"title": "Amplitude"},
        )
        fig = go.Figure(data=[trace], layout=layout)
        if not output_html:
            base = os.path.splitext(os.path.basename(self.filepath))[0]
            output_html = f"{base}_waveform.html"
        pyo.plot(fig, filename=output_html, auto_open=False)

    def get_metadata(self) -> dict:
        """
        Return basic metadata of the WAV file.
        """
        return {
            "filepath": self.filepath,
            "sample_rate": self.sample_rate,
            "channels": self.n_channels,
            "sample_width": self.sampwidth,
            "num_samples": len(self.audio),
        }

    def __repr__(self) -> str:
        return (
            f"<WavSerialDecoder {self.filepath!r}: "
            f"{self.n_channels}ch, {self.sample_rate}Hz, {len(self.audio)/self.sample_rate:.2f}s>"
        )
