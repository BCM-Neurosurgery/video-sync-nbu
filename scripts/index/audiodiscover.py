"""
Audio discovery (fast path, no probing)
======================================

Lightweight, naming-driven discovery of channelized audio files for the A/V pipeline.
This module **does not probe** audio durations or sample rates to keep directory
scans fast even when files span many hours/days.

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
- Since we skip probing, created `Audio`/`SerialAudio` objects populate
  ``duration=0.0`` and ``sample_rate=0`` as placeholders.
- Any duration equality checks are therefore **informational only** unless a later
  step fills in accurate metadata.

Examples
--------
Typical directory (either 2 or 3 files sharing the same extension)::

    TRBD002_08062025-01.mp3
    TRBD002_08062025-02.mp3
    TRBD002_08062025-03.mp3

or::

    TRBD002_08062025-01.wav
    TRBD002_08062025-03.wav
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterable

from scripts.index.common import (
    _DirMixin,
    _filesize_mb,
    _safe_glob,
    _format_channels,
)
from scripts.index.filepatterns import FilePatterns
from scripts.models import (
    Audio,
    SerialAudio,
    AudioGroup,
)


class AudioDiscoverer(_DirMixin):
    """
    Discover channelized audio files and build an `AudioGroup` (no probing).

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
    - Discovery enforces filename/structure invariants (see module docstring)
      but intentionally **skips** duration/sample-rate probing for performance.
    - `Audio.duration` and `Audio.sample_rate` are set to placeholder values
      (``0.0`` and ``0``) in this fast mode.
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
        return _safe_glob(self.audio_dir, ("*.wav", "*.mp3"))

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

    def _build_audio_obj(self, ch: int, p: Path) -> Audio:
        """
        Construct an `Audio` object without probing (fast placeholders).

        Parameters
        ----------
        ch : int
            Channel number.
        p : pathlib.Path
            File path.

        Returns
        -------
        Audio
            Object with `duration=0.0` and `sample_rate=0` (not probed). File size
            is filled for convenience.
        """
        ext = p.suffix.lower().lstrip(".")
        # Fast path: skip probes to avoid heavy I/O on long recordings.
        dur, sr = 0.0, 0
        return Audio(
            path=p,
            duration=dur,
            file_size=_filesize_mb(p),
            sample_rate=sr,
            extension=ext,
            channel=ch,
        )

    def _build_serial_audio_obj(self, ch: int, p: Path) -> SerialAudio:
        """
        Construct a `SerialAudio` object without probing (fast placeholders).

        Parameters
        ----------
        ch : int
            Channel number (typically 3).
        p : pathlib.Path
            File path.

        Returns
        -------
        SerialAudio
            Object mirroring `Audio` fields with placeholder duration/sample_rate.
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
        In fast (no-probe) mode, all durations are placeholders (0.0), so this
        check is informational only. Consider running a separate verification step
        that fills accurate metadata before relying on duration equality.
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
        Discover and return an `AudioGroup` from `audio_dir` (no probing).

        Workflow
        --------
        1) Collect candidates (``*.wav``/``*.mp3``).
        2) Validate naming/count/channel rules.
        3) Choose a single file per channel, preferring WAV.
        4) Infer shared extension (warn if mixed).
        5) Build `Audio`/`SerialAudio` objects with placeholder metadata.
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
