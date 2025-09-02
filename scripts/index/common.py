from pathlib import Path
from typing import Tuple, Iterable, List
import wave
from pydub import AudioSegment
from zoneinfo import ZoneInfo


DEFAULT_TZ = ZoneInfo("America/Chicago")


# ---------------------------------------------------------------------------
# Small probes & helpers
# ---------------------------------------------------------------------------
def _filesize_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def _probe_wav(p: Path) -> Tuple[float, int]:
    """Return (duration_sec, sample_rate) for WAV, else (0.0, 0)."""
    try:
        with wave.open(str(p), "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            dur = float(n) / sr if sr else 0.0
            return dur, sr
    except Exception:
        return 0.0, 0


def _probe_mp3(p: Path) -> Tuple[float, int]:
    """Return (duration_sec, sample_rate) for MP3 using pydub if available, else (0.0, 0)."""
    try:
        seg = AudioSegment.from_file(p)
        return len(seg) / 1000.0, seg.frame_rate
    except Exception:
        return 0.0, 0


def _format_channels(channels: Iterable[int]) -> str:
    return ", ".join(f"{ch:02d}" for ch in channels)


def _safe_glob(directory: Path, patterns: Iterable[str]) -> List[Path]:
    """Glob multiple patterns and return a single sorted list."""
    results: List[Path] = []
    for pat in patterns:
        results.extend(directory.glob(pat))
    return sorted({p for p in results if p.is_file()})


class _DirMixin:
    def _ensure_exists(self, directory: Path) -> None:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
