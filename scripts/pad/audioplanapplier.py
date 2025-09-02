#!/usr/bin/env python3
"""
AudioPlanApplier
================

Apply a padding EditPlan (JSON) to a single audio file (WAV or MP3 input) and
export the edited audio as **WAV (PCM_16)** with a "-padded.wav" suffix.

This tool ALWAYS writes WAV. If the predicted output would exceed the classic
RIFF/WAV 4 GB limit, it **raises an error** instead of writing a corrupt file.

Plan format (JSON)
------------------
A list of objects, each describing an insertion on the *original* timeline:

[
  {
    "insert_after_sample": <int>,   # 0-based sample index on the original timeline
    "insert_len_samples": <int>,   # number of samples to insert (non-negative)
    "note": "optional diagnostics"
  },
  ...
]

Semantics:
- For each entry, insert `insert_len_samples` samples *after* index `insert_after_sample`.
  Example: if `insert_after_sample == 9`, we insert starting at new position 10
  (i.e., between original samples 9 and 10). Indices are based on the ORIGINAL
  (pre-insertion) audio.
- Insertions are applied in ascending order of `insert_after_sample`.
- The inserted content is **silence** (all zeros) with the same number of channels
  as the source. If you prefer crossfades or cloned content, extend `_make_fill`.

I/O behavior
------------
- Input may be WAV or MP3 (MP3 is decoded to PCM).
- Output is always `<name>-padded.wav` encoded as **PCM_16**.
- Before processing, we compute the predicted output size and raise a clear
  error if it would exceed the RIFF/WAV 4 GB limit.

Dependencies
------------
- Required: numpy
- Required: soundfile (fast WAV I/O). If not available, a clear error is raised.
- Optional: librosa (fallback decoder for MP3/other formats if soundfile cannot read).

Note on encoder/decoder delay
-----------------------------
This tool operates in PCM sample space. If you decode an MP3, apply the plan, and write WAV,
there is no *additional* encoder delay in the output (WAV is PCM). The *relative* placement
of insertions is preserved exactly in the decoded PCM domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
from typing import List, Tuple, Optional, Union
import re
import numpy as np

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - optional dep
    sf = None  # type: ignore

# --- logutils -----------------------------------------------------------
from scripts.log.logutils import configure_standalone_logging

logger = logging.getLogger(__name__)
# -----------------------------------------------------------------------


@dataclass
class EditOp:
    """A single insertion on the original timeline.

    Attributes
    ----------
    insert_after_sample : int
        The 0-based index in the ORIGINAL audio after which the insertion occurs.
    insert_len_samples : int
        Number of samples to insert (non-negative).
    note : Optional[str]
        Optional diagnostic note.
    """

    insert_after_sample: int
    insert_len_samples: int
    note: Optional[str] = None


class AudioPlanApplier:
    """Apply a padding plan to an audio file and save the edited result (WAV).

    Parameters
    ----------
    audio_path : Union[Path, str]
        Path to the input audio file (WAV or MP3).
    plan_path : Union[Path, str]
        Path to the JSON plan (list of edit ops as described above).
    out_dir : Union[Path, str]
        Directory where the edited audio will be saved.

    Notes
    -----
    - Insertions are silence of the same channel count as the source.
    - All indices are assumed to be 0-based and refer to the source's ORIGINAL
      timeline, *before* any insertions are applied.
    - The class does not modify the source file; it writes a new file with
      suffix "-padded.wav".
    """

    def __init__(
        self,
        audio_path: Union[Path, str],
        plan_path: Union[Path, str],
        out_dir: Union[Path, str],
    ):
        self.audio_path = Path(audio_path)
        self.plan_path = Path(plan_path)
        self.out_dir = Path(out_dir)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        if not self.plan_path.exists():
            raise FileNotFoundError(f"Plan JSON not found: {self.plan_path}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------- Public API ------------------------------- #
    def apply(self) -> Path:
        """Apply the plan to the input audio and write the edited WAV."""
        logger.info(
            "Starting plan application: audio=%s plan=%s outdir=%s",
            self.audio_path.name,
            self.plan_path.name,
            self.out_dir.name,
        )
        ops = self._load_plan()
        logger.info("Loaded plan with %d ops", len(ops))
        data, sr = self._load_audio(self.audio_path)
        channels = data.shape[1] if data.ndim == 2 else 1
        logger.info("Audio decoded: shape=%s  sr=%d  ch=%d", data.shape, sr, channels)

        # Preflight size check against WAV 4 GB limit (PCM_16)
        self._preflight_wav_limit(data_len=len(data), channels=channels, ops=ops, sr=sr)

        logger.info("Applying %d insertions…", len(ops))
        edited = self._apply_ops_chunked(data, ops)
        logger.info("Insertions applied. New length = %d samples", len(edited))

        out_path = self._make_out_path()
        logger.info("Writing edited WAV to %s", out_path.name)
        self._write_audio(out_path, edited, sr)
        logger.info("Done.")
        return out_path

    # ------------------------------ I/O helpers ------------------------------ #
    def _load_plan(self) -> List[EditOp]:
        with open(self.plan_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("Plan JSON must be a list of operations")
        ops: List[EditOp] = []
        for i, item in enumerate(raw):
            try:
                ops.append(
                    EditOp(
                        insert_after_sample=int(item["insert_after_sample"]),
                        insert_len_samples=int(item["insert_len_samples"]),
                        note=item.get("note"),
                    )
                )
            except Exception as e:  # pragma: no cover - defensive parsing
                raise ValueError(f"Invalid plan item at index {i}: {item!r} ({e})")
        # Ensure ascending order by anchor index
        ops.sort(key=lambda x: x.insert_after_sample)
        total_insert = sum(max(0, op.insert_len_samples) for op in ops)
        logger.debug(
            "Plan summary: %d ops, total_insert=%d samples", len(ops), total_insert
        )
        return ops

    def _load_audio(self, path: Path) -> Tuple[np.ndarray, int]:
        """Load audio into float32 PCM (frames, channels) and return (data, sr).

        - WAV is read via soundfile.
        - Other formats: try soundfile first (if libsndfile supports them),
          otherwise fall back to librosa/audioread.
        """
        ext = path.suffix.lower()
        if ext == ".wav":
            if sf is None:
                raise RuntimeError("soundfile is required to read WAV files")
            logger.debug("Reading WAV via soundfile: %s", path)
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return data, int(sr)

        # Try soundfile for non-WAV (works if libsndfile has codec support)
        if sf is not None:
            try:
                logger.debug("Reading non-WAV via soundfile: %s", path)
                data, sr = sf.read(path, dtype="float32", always_2d=True)
                return data, int(sr)
            except Exception:
                logger.debug(
                    "soundfile could not decode %s; falling back to librosa", path
                )
                pass

        # Fallback decoder using librosa/audioread (MP3, etc.)
        try:
            import librosa  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Cannot decode non-WAV audio without librosa (and audioread/ffmpeg)."
            ) from e
        logger.debug("Reading via librosa/audioread: %s", path)
        y, sr = librosa.load(str(path), sr=None, mono=False)  # y: (n,) or (ch, n)
        if y.ndim == 1:
            data = y.astype(np.float32)[:, None]  # (n, 1)
        else:
            data = y.T.astype(np.float32)  # (n, ch)
        return data, int(sr)

    # --------------------------- Preflight size check --------------------------- #
    def _preflight_wav_limit(
        self, data_len: int, channels: int, ops: List[EditOp], sr: int
    ) -> None:
        """Raise if predicted WAV output would exceed the 4 GB RIFF/WAV limit.

        Assumes PCM_16 output (2 bytes per sample per channel).
        """
        total_insert = sum(max(0, int(op.insert_len_samples)) for op in ops)
        n_final = int(data_len) + int(total_insert)
        bytes_per_frame = 2 * int(channels)  # PCM_16
        data_bytes = n_final * bytes_per_frame
        MAX_WAV_BYTES = (2**32 - 1) - 44  # conservative header allowance
        if data_bytes > MAX_WAV_BYTES:
            raise RuntimeError(
                (
                    "Predicted WAV output exceeds 4 GB limit: "
                    f"frames={n_final}, channels={channels}, sr={sr}, "
                    f"bytes≈{data_bytes:,} (> {MAX_WAV_BYTES:,}). "
                    "Consider segmenting or using RF64/WAVE64/FLAC."
                )
            )
        logger.info(
            "Preflight OK: final frames=%d, channels=%d, sr=%d, bytes≈%s",
            n_final,
            channels,
            sr,
            f"{data_bytes:,}",
        )

    def _write_audio(self, out_path: Path, data: np.ndarray, sr: int) -> None:
        """Write WAV (PCM_16)."""
        if sf is None:
            raise RuntimeError("soundfile is required to write WAV files")
        if data.ndim == 1:
            data = data[:, None]
        to_write = np.clip(data, -1.0, 1.0)
        sf.write(out_path, to_write, sr, subtype="PCM_16")

    def _make_out_path(self) -> Path:
        """
        If input stem ends with '-NN' (e.g., '-01'/'-02'/'-03'), produce:
            <base>-padded-<NN>.wav
        Else (fallback): <stem>-padded.wav
        """
        stem = self.audio_path.stem  # e.g., "TRBD002_08062025-03"
        m = re.match(r"^(?P<base>.+)-(?P<seg>\d{2})$", stem)
        if m:
            base = m.group("base")
            seg = m.group("seg")
            out_name = f"{base}-padded-{seg}.wav"
        else:
            out_name = f"{stem}-padded.wav"
        return self.out_dir / out_name

    # --------------------------- Core plan application --------------------------- #
    def _apply_ops_chunked(self, data: np.ndarray, ops: List[EditOp]) -> np.ndarray:
        """Apply insertions using a zero-copy chunk assembly strategy.

        This avoids repeatedly reallocating the growing array. Instead, we
        collect source slices and fillers into a list and concatenate once.
        """
        if data.ndim == 1:
            data = data[:, None]
        n, ch = data.shape
        parts: List[np.ndarray] = []
        cursor = 0  # current read head on the ORIGINAL timeline (0..n)

        total_ops = len(ops)
        if total_ops:
            logger.debug("Begin applying %d ops (zero-copy)", total_ops)

        for k, op in enumerate(ops):
            if op.insert_len_samples <= 0:
                continue
            anchor = int(op.insert_after_sample)
            # Convert "after sample" (0..n-1) to an insertion position in [0..n]
            pos = anchor + 1
            # Clamp & monotonicity enforcement
            if pos < cursor:
                logger.warning(
                    "Plan op %d insertion at %d is before current cursor %d. Clamping to %d.",
                    k,
                    pos,
                    cursor,
                    cursor,
                )
                pos = cursor
            if pos > n:
                logger.warning(
                    "Plan op %d insertion at %d beyond EOF %d. Clamping to end.",
                    k,
                    pos,
                    n,
                )
                pos = n
            # Append the next source span, then the filler
            if pos > cursor:
                parts.append(data[cursor:pos])
                cursor = pos
            # Append silence of requested length
            parts.append(self._make_fill(op.insert_len_samples, ch, data.dtype))
            if total_ops >= 100 and (k + 1) % 100 == 0:
                logger.debug("...applied %d/%d ops", k + 1, total_ops)

        # Remainder of the source
        if cursor < n:
            parts.append(data[cursor:n])

        # Concatenate along time axis
        if not parts:
            return data.copy()
        edited = np.concatenate(parts, axis=0)
        return edited

    @staticmethod
    def _make_fill(length: int, channels: int, dtype: np.dtype) -> np.ndarray:
        """Create a silent segment of the given length/channels/dtype."""
        shape = (int(length), int(channels))
        return np.zeros(shape, dtype=dtype)


# ------------------------------ Minimal CLI ------------------------------ #


def _build_cli_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Apply a padding EditPlan (JSON) to an audio file and write <name>-padded.wav",
    )
    p.add_argument("audio", type=Path, help="Input audio file (WAV/MP3)")
    p.add_argument("plan", type=Path, help="Edit plan JSON produced by AudioPadder")
    p.add_argument("outdir", type=Path, help="Output directory for the edited WAV")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logs (default is WARNING for minimal output)",
    )
    return p


def main():
    import sys

    args = _build_cli_parser().parse_args()

    # Use standalone console logging from logutils (won't interfere with driver)
    configure_standalone_logging(
        level="INFO" if args.verbose else "WARNING", seg="-", cam="-"
    )

    try:
        applier = AudioPlanApplier(
            audio_path=args.audio, plan_path=args.plan, out_dir=args.outdir
        )
        out = applier.apply()
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
