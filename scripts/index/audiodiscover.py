"""
Audio discovery (fast path, WAV header sr; MP3 sr via ffprobe; no duration probing)
==================================================================================

Lightweight, naming-driven discovery of channelized audio files for the A/V pipeline.
This module **does not probe durations** (to stay fast on multi-day recordings), but
it **does determine sampling rate**:
- **WAV**: read the RIFF/WAVE `fmt ` chunk (header-only, O(1) bytes).
- **MP3**: query **ffprobe** for the audio stream sample rate (fast, no full decode).

Validation rules (enforced)
---------------------------
1. Filenames must match ``<prefix>-<chan>.<ext>`` where ``chan`` is two digits and
   ``ext`` ∈ {``wav``, ``mp3``}; otherwise **raise** ``ValueError``.
2. The directory may contain **at most 3** files for a given extension (≤3 ``.wav`` and
   ≤3 ``.mp3``); otherwise **raise**.
3. There must be **exactly one** channel ``-03`` file overall; otherwise **raise**.
4. There must be **at least one** of channels ``-01`` or ``-02``; if neither, **raise**;
   if only one is present, **warn**.

Notes
-----
- We set `Audio.duration = 0.0` as a placeholder (no duration probing).
- `Audio.sample_rate` is filled via a **WAV header read** or **ffprobe (MP3)**.
  If sample-rate detection fails, we raise `AudioGroupDiscoverError`.
"""

import logging
import shutil
import struct
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterable

from scripts.errors import AudioGroupDiscoverError
from scripts.index.common import (
    _DirMixin,
    _filesize_mb,
)
from scripts.index.filepatterns import FilePatterns
from scripts.models import (
    Audio,
    SerialAudio,
    AudioGroup,
)


class AudioDiscoverer(_DirMixin):
    """
    Discover channelized audio files and build an `AudioGroup`
    using filename rules and fast sampling-rate detection.

    Parameters
    ----------
    audio_dir : pathlib.Path
        Directory containing channelized audio files named as
        ``<prefix>-<chan>.<ext>`` where ``chan`` is two digits.
    default_serial_channel : int, default=3
        Channel number treated as the serial channel (e.g., ``-03``).
    log : logging.Logger
        Logger for warnings/errors and progress messages.

    Notes
    -----
    - Duration probing is intentionally **skipped** for performance.
    - Sampling rate is obtained via **WAV header** (RIFF `fmt `) or **ffprobe** (MP3).
    """

    def __init__(
        self,
        audio_dir: Path,
        default_serial_channel: int = 3,
        *,
        log: logging.Logger,
    ):
        self.audio_dir = audio_dir
        self.default_serial_channel = default_serial_channel
        self.log = log

    # ---- small helpers -----------------------------------------------------

    def _collect_candidates(self) -> List[Path]:
        """
        Collect candidate audio files in the target directory.

        Returns
        -------
        list of pathlib.Path
            All files matching ``*.wav`` or ``*.mp3`` in `audio_dir`.

        Raises
        ------
        FileNotFoundError
            If `audio_dir` does not exist.
        """
        self._ensure_exists(self.audio_dir)
        # Case-insensitive extension match (important on case-sensitive filesystems).
        candidates: List[Path] = []
        for p in self.audio_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() in {".wav", ".mp3"}:
                candidates.append(p)
        return sorted(candidates)

    def _validate_candidates(self, candidates: List[Path]) -> None:
        """
        Validate directory contents against naming and count rules.

        Enforced rules
        --------------
        1) Basename must match ``<prefix>-<chan>.<ext>`` (``chan``=2 digits,
           ``ext`` in {wav, mp3}); else **raise**.
        2) At most 3 files **per extension** (``.wav`` or ``.mp3``); else **raise**.
        3) Exactly one channel ``-03`` overall; else **raise**.
        4) At least one of ``-01`` or ``-02`` exists; if neither, **raise**; if only
           one exists, **warn**.

        Parameters
        ----------
        candidates : list of pathlib.Path
            Candidate audio paths to validate.

        Raises
        ------
        ValueError
            If any rule is violated.
        """
        invalid: List[str] = []
        by_ext: Dict[str, List[Path]] = {"wav": [], "mp3": []}
        ch_counts: Dict[int, int] = {}

        for p in candidates:
            parsed = FilePatterns.parse_audio_filename(p)
            if not parsed:
                invalid.append(p.name)
                continue
            ch, ext = parsed
            by_ext.setdefault(ext, []).append(p)
            ch_counts[ch] = ch_counts.get(ch, 0) + 1

        if invalid:
            raise ValueError(
                "Audio files with unexpected name pattern: "
                + ", ".join(sorted(invalid))
            )

        # Rule 1: at most 3 per extension
        for ext, paths in by_ext.items():
            if len(paths) > 3:
                raise ValueError(
                    f"Found {len(paths)} *.{ext} files in {self.audio_dir} "
                    f"(max allowed per extension is 3)."
                )

        # Rule 2: exactly one -03.*
        serial_count = ch_counts.get(self.default_serial_channel, 0)
        if serial_count != 1:
            raise ValueError(
                f"Expected exactly one channel {self.default_serial_channel:02d} "
                f"file (e.g., *-{self.default_serial_channel:02d}.wav/mp3); "
                f"found {serial_count}."
            )

        # Rule 3: at least one of -01 or -02 (warn if only one)
        has_01 = 1 in ch_counts
        has_02 = 2 in ch_counts
        if not (has_01 or has_02):
            raise ValueError(
                "Expected at least one program channel: "
                "found neither -01 nor -02 in AUDIO_DIR."
            )
        if has_01 ^ has_02:  # exactly one present
            missing = "01" if not has_01 else "02"
            self.log.warning(
                "Only one program channel present; missing -%s. "
                "Proceeding with available channel.",
                missing,
            )

    def _choose_best_per_channel(self, candidates: List[Path]) -> Dict[int, Path]:
        """
        Select one file per detected channel, preferring WAV over MP3.

        Parameters
        ----------
        candidates : list of pathlib.Path
            Validated candidate audio files.

        Returns
        -------
        dict[int, pathlib.Path]
            Mapping from channel number to the chosen file path.

        Notes
        -----
        If both ``.wav`` and ``.mp3`` exist for the same channel, the ``.wav`` is chosen.
        """
        chosen: Dict[int, Path] = {}
        for p in candidates:
            parsed = FilePatterns.parse_audio_filename(p)
            if not parsed:
                # Naming validation is handled in _validate_candidates; skip here.
                continue
            ch, ext = parsed
            existing = chosen.get(ch)
            if existing is None:
                chosen[ch] = p
            else:
                if existing.suffix.lower() == ".mp3" and ext == "wav":
                    chosen[ch] = p
        return chosen

    def _infer_shared_extension(self, paths: Iterable[Path]) -> Optional[str]:
        """
        Infer a shared extension across selected channels, if any.

        Parameters
        ----------
        paths : Iterable[pathlib.Path]
            Selected per-channel paths.

        Returns
        -------
        Optional[str]
            The shared extension (``"wav"`` or ``"mp3"``) if all match; otherwise
            ``None``. Logs a warning when mixed.
        """
        exts = {p.suffix.lower().lstrip(".") for p in paths}
        if len(exts) == 1 and exts <= {"wav", "mp3"}:
            return next(iter(exts))
        if exts:
            self.log.warning(
                "Mixed audio extensions detected across channels: %s.",
                ", ".join(sorted(exts)),
            )
        return None

    # ---- sampling-rate helpers ---------------------------------------------

    def _sniff_wav_samplerate(self, p: Path) -> int:
        """
        Read WAV `fmt ` chunk to obtain sample rate (header-only).

        Parameters
        ----------
        p : pathlib.Path
            Path to a WAV file.

        Returns
        -------
        int
            Sample rate in Hz, or ``0`` if parsing fails.
        """
        try:
            with p.open("rb") as f:
                header = f.read(12)
                if (
                    len(header) < 12
                    or header[0:4] != b"RIFF"
                    or header[8:12] != b"WAVE"
                ):
                    return 0
                # Iterate chunks until `fmt ` is found
                while True:
                    chunk_hdr = f.read(8)
                    if len(chunk_hdr) < 8:
                        return 0
                    chunk_id, chunk_size = (
                        chunk_hdr[0:4],
                        struct.unpack("<I", chunk_hdr[4:8])[0],
                    )
                    if chunk_id == b"fmt ":
                        fmt_data = f.read(chunk_size)
                        if len(fmt_data) < 8:
                            return 0
                        # sampleRate is at offset 4..8 in fmt chunk
                        return struct.unpack("<I", fmt_data[4:8])[0]
                    else:
                        # Skip chunk (account for padding to even)
                        f.seek(chunk_size + (chunk_size & 1), 1)
        except Exception:
            return 0

    def _sniff_mp3_samplerate(self, p: Path) -> int:
        """
        Use `ffprobe` to obtain MP3 stream sample rate.

        Parameters
        ----------
        p : pathlib.Path
            Path to an MP3 file.

        Returns
        -------
        int
            Sample rate in Hz, or ``0`` if ffprobe is unavailable or fails.

        Notes
        -----
        - Fast query (no full decode): reads container/stream headers.
        - Falls back to a larger probe window if the first attempt yields nothing.
        """
        if shutil.which("ffprobe") is None:
            self.log.error(
                "ffprobe not found on PATH; cannot sniff MP3 sample rate for %s", p.name
            )
            return 0

        base_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "csv=p=0",
            str(p),
        ]
        try:
            r = subprocess.run(
                base_cmd, capture_output=True, text=True, check=True, timeout=10
            )
            out = r.stdout.strip()
            if out.isdigit():
                return int(out)
        except Exception as e:
            self.log.debug("ffprobe quick sr query failed for %s: %s", p.name, e)

        # Second attempt with larger probe/analyzeduration
        deep_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-probesize",
            "5M",
            "-analyzeduration",
            "5M",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "csv=p=0",
            str(p),
        ]
        try:
            r2 = subprocess.run(
                deep_cmd, capture_output=True, text=True, check=True, timeout=20
            )
            out2 = r2.stdout.strip()
            if out2.isdigit():
                return int(out2)
        except Exception as e:
            self.log.debug("ffprobe deep sr query failed for %s: %s", p.name, e)

        self.log.warning("Could not obtain sample rate via ffprobe for %s", p.name)
        return 0

    def _build_audio_obj(self, ch: int, p: Path) -> Audio:
        """
        Construct an `Audio` object with fast sample rate (no duration probe).

        Parameters
        ----------
        ch : int
            Channel number.
        p : pathlib.Path
            File path.

        Returns
        -------
        Audio
            Object with `duration=0.0` (placeholder) and `sample_rate` from
            WAV header or ffprobe (MP3); falls back to ``0`` if detection fails.
        """
        ext = p.suffix.lower().lstrip(".")
        if ext == "wav":
            sr = self._sniff_wav_samplerate(p)
        elif ext == "mp3":
            sr = self._sniff_mp3_samplerate(p)
        else:
            sr = 0
        if sr <= 0:
            msg = (
                f"Could not determine sample rate for {p.name} (ext={ext}). "
                "Fix the audio file or ensure required tools (e.g. ffprobe) are available."
            )
            self.log.error("%s", msg)
            raise AudioGroupDiscoverError(msg)

        self.log.info("Discovered audio ch%02d: %s (SR=%d Hz)", ch, p, sr)
        return Audio(
            path=p,
            duration=0.0,
            file_size=_filesize_mb(p),
            sample_rate=sr,
            extension=ext,
            channel=ch,
        )

    def _build_serial_audio_obj(self, ch: int, p: Path) -> SerialAudio:
        """
        Construct a `SerialAudio` object (no duration probe, fast SR).

        Parameters
        ----------
        ch : int
            Channel number (typically 3).
        p : pathlib.Path
            File path.

        Returns
        -------
        SerialAudio
            Object mirroring `Audio` fields with placeholder duration and detected
            sample rate when available.
        """
        a = self._build_audio_obj(ch, p)
        return SerialAudio(
            path=a.path,
            duration=a.duration,
            file_size=a.file_size,
            sample_rate=a.sample_rate,
            extension=a.extension,
            channel=a.channel,
        )

    def _build_audio_map(
        self, chosen: Dict[int, Path]
    ) -> Tuple[Dict[int, Audio], Optional[SerialAudio]]:
        """
        Build per-channel `Audio` objects and identify the serial channel.

        Parameters
        ----------
        chosen : dict[int, pathlib.Path]
            Mapping from channel to selected path.

        Returns
        -------
        (dict[int, Audio], Optional[SerialAudio])
            Tuple of the per-channel `Audio` objects and the `SerialAudio`
            (or ``None`` if not present, which should not occur after validation).

        Raises
        ------
        ValueError
            If the expected serial channel is missing (guard after validation).
        """
        audios: Dict[int, Audio] = {}
        serial_audio: Optional[SerialAudio] = None
        for ch, p in sorted(chosen.items(), key=lambda kv: kv[0]):
            if ch == self.default_serial_channel:
                serial_audio = self._build_serial_audio_obj(ch, p)
                audios[ch] = serial_audio
            else:
                audios[ch] = self._build_audio_obj(ch, p)
        if serial_audio is None:
            # Should be impossible due to _validate_candidates, but keep the guard.
            raise ValueError(
                f"Default serial channel {self.default_serial_channel:02d} "
                "not found in AUDIO_DIR."
            )
        return audios, serial_audio

    def _check_equal_durations(self, audios: Dict[int, Audio]) -> None:
        """
        Warn if durations differ across channels.

        Parameters
        ----------
        audios : dict[int, Audio]
            Per-channel audio objects.

        Notes
        -----
        In fast (no-duration-probe) mode, all durations are placeholders (0.0),
        so this check is informational only.
        """
        if not audios:
            return
        items = sorted(audios.items(), key=lambda kv: kv[0])  # sort by channel
        base_dur = items[0][1].duration
        mismatched = [(ch, a.duration) for ch, a in items if a.duration != base_dur]
        if mismatched:
            pretty = ", ".join(
                f"ch{ch:02d}={dur:.6f}s"
                for ch, dur in [(items[0][0], base_dur)] + mismatched
            )
            self.log.warning(
                "Audio durations are not identical across channels: %s", pretty
            )

    # ---- public ------------------------------------------------------------

    def get_audio_group(self) -> AudioGroup:
        """
        Discover and return an `AudioGroup` from `audio_dir` (no duration probing).

        Workflow
        --------
        1) Collect candidates (``*.wav``/``*.mp3``).
        2) Validate naming/count/channel rules.
        3) Choose a single file per channel, preferring WAV.
        4) Infer shared extension (warn if mixed).
        5) Build `Audio`/`SerialAudio` objects with WAV-header or ffprobe (MP3) SR.
        6) Optionally log duration equality (informational in fast mode).

        Returns
        -------
        AudioGroup
            Aggregated audio description for downstream stages.

        Raises
        ------
        FileNotFoundError
            If `audio_dir` is missing.
        ValueError
            If validation fails (see rules in module docstring).
        """
        candidates = self._collect_candidates()
        self._validate_candidates(candidates)

        chosen_by_channel = self._choose_best_per_channel(candidates)

        # Expect either {3,1} or {3,2} or {1,2,3}; other shapes will already have raised earlier.
        shared_ext = self._infer_shared_extension(chosen_by_channel.values())
        audios, serial_audio = self._build_audio_map(chosen_by_channel)

        # Duration check is informational only in fast mode.
        self._check_equal_durations(audios)

        return AudioGroup(
            audios=audios,
            serial_audio=serial_audio,
            shared_extension=shared_ext,
        )
