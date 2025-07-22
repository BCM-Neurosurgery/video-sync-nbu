import wave
import numpy as np
import contextlib


class WavFileParser:
    """
    A clean, modular WAV file parser for reading metadata and audio data.

    Attributes:
        filepath (str): Path to the WAV file.
        sample_rate (int): Sample rate of the audio.
        n_channels (int): Number of audio channels.
        n_frames (int): Total number of frames in the audio.
        sampwidth (int): Sample width in bytes.
        duration (float): Duration of the audio in seconds.
        audio_data (np.ndarray | None): Audio data as a NumPy array.
    """

    def __init__(self, filepath: str) -> None:
        """
        Initialize the WavFileParser with the given WAV file path.
        Args:
            filepath (str): Path to the WAV file.
        """
        self.filepath = filepath
        self.sample_rate = None
        self.n_channels = None
        self.n_frames = None
        self.sampwidth = None
        self.duration = None
        self.audio_data = None
        self._parse_metadata()

    def _parse_metadata(self) -> None:
        """
        Parse and set metadata attributes from the WAV file header.
        """
        with contextlib.closing(wave.open(self.filepath, "rb")) as wf:
            self.sample_rate = wf.getframerate()
            self.n_channels = wf.getnchannels()
            self.n_frames = wf.getnframes()
            self.sampwidth = wf.getsampwidth()
            self.duration = self.n_frames / float(self.sample_rate)

    def read_audio(self) -> np.ndarray:
        """
        Read and return the audio data as a NumPy array.
        Returns:
            np.ndarray: Audio samples. Shape is (frames,) for mono or (frames, channels) for multi-channel.
        """
        with contextlib.closing(wave.open(self.filepath, "rb")) as wf:
            frames = wf.readframes(self.n_frames)
            dtype = np.int16 if self.sampwidth == 2 else np.uint8
            audio = np.frombuffer(frames, dtype=dtype)
            if self.n_channels > 1:
                audio = audio.reshape(-1, self.n_channels)
            self.audio_data = audio
        return self.audio_data

    def get_metadata(self) -> dict:
        """
        Get parsed metadata for the WAV file.
        Returns:
            dict: Dictionary containing sample_rate, channels, frames, sample_width, and duration.
        """
        return {
            "sample_rate": self.sample_rate,
            "channels": self.n_channels,
            "frames": self.n_frames,
            "sample_width": self.sampwidth,
            "duration": self.duration,
        }

    def plot_waveform_interactive(
        self, output_html: str = None, window_size: int = None, start_sample: int = 0
    ) -> None:
        """
        Plot the audio waveform as an interactive HTML plot using Plotly.
        Args:
            output_html (str, optional): Path to save the HTML file. If None, saves to '<wavfilename>_waveform.html' in the current directory.
            window_size (int, optional): Number of samples to plot. If None, plot all samples.
            start_sample (int, optional): Starting sample index for the window. Default is 0.
        """
        import plotly.graph_objs as go
        import plotly.offline as pyo
        import os

        audio = self.audio_data if self.audio_data is not None else self.read_audio()
        # Apply windowing with robust validation
        total_samples = audio.shape[0]
        if start_sample < 0 or start_sample >= total_samples:
            raise ValueError(
                f"start_sample {start_sample} is out of bounds (0 to {total_samples-1})"
            )
        if window_size is not None:
            if window_size <= 0:
                raise ValueError(f"window_size must be positive, got {window_size}")
            end_sample = start_sample + window_size
            if end_sample > total_samples:
                end_sample = total_samples
            if audio.ndim == 1:
                audio_window = audio[start_sample:end_sample]
            else:
                audio_window = audio[start_sample:end_sample, :]
        else:
            audio_window = audio[start_sample:]

        if self.n_channels == 1:
            trace = go.Scatter(y=audio_window, mode="lines", name="Mono")
            data = [trace]
        else:
            data = [
                go.Scatter(y=audio_window[:, ch], mode="lines", name=f"Channel {ch+1}")
                for ch in range(self.n_channels)
            ]
        layout = go.Layout(
            title=f"Interactive Waveform: {os.path.basename(self.filepath)}",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )
        fig = go.Figure(data=data, layout=layout)
        if output_html is None:
            base = os.path.splitext(os.path.basename(self.filepath))[0]
            output_html = f"{base}_waveform.html"
        pyo.plot(fig, filename=output_html, auto_open=False)

    def __repr__(self) -> str:
        """
        String representation of the WavFileParser instance.
        Returns:
            str: Human-readable summary of the WAV file.
        """
        return f"<WavFileParser: {self.filepath}, {self.n_channels}ch, {self.sample_rate}Hz, {self.duration:.2f}s>"
