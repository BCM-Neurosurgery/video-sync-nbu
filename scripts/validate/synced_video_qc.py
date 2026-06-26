from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"

MATCHED_WINDOW_RE = re.compile(
    r"Matched window:\s*frames\s*\[(?P<frame0>\d+)\.\.(?P<frame1>\d+)\]"
    r"\s*\(n=(?P<frames>\d+)\),\s*samples\s*\[(?P<s0>\d+)\.\.(?P<s1>\d+)\)"
    r".*?CFR=(?P<fps>[0-9.]+)"
)
ERROR_RE = re.compile(r"\b(ERROR|CRITICAL)\b", re.IGNORECASE)
WARNING_RE = re.compile(r"\bWARNING\b", re.IGNORECASE)
NON_FORWARD_RE = re.compile(r"Non-forward serial pair", re.IGNORECASE)


@dataclass(frozen=True)
class QcThresholds:
    frame_warn: int = 0
    frame_fail: int = 2
    duration_warn_ms: float = 50.0
    duration_fail_ms: float = 100.0
    residual_warn_ms: float = 20.0
    residual_fail_ms: float = 50.0


@dataclass(frozen=True)
class SyncWindow:
    frame0: int
    frame1: int
    expected_frames: int
    sample0: int
    sample1: int
    fps: float

    @property
    def sample_count(self) -> int:
        return self.sample1 - self.sample0

    @property
    def duration_sec(self) -> float:
        return self.expected_frames / self.fps

    @property
    def sample_rate_hz(self) -> float:
        return self.sample_count / self.duration_sec


@dataclass(frozen=True)
class SyncLogInfo:
    path: str
    window: SyncWindow | None
    warnings: list[str]
    errors: list[str]
    non_forward_warnings: list[str]


@dataclass(frozen=True)
class MediaInfo:
    path: str
    video_duration_sec: float | None
    frame_count: int | None
    avg_frame_rate: str | None
    r_frame_rate: str | None
    audio_durations_sec: list[float]
    audio_sample_rates: list[int]


@dataclass(frozen=True)
class ResidualSummary:
    path: str
    anchor_count: int
    max_abs_ms: float
    median_abs_ms: float
    p95_abs_ms: float


def discover_camera_dirs(root: Path) -> list[Path]:
    """Find per-camera sync output directories below an output root."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    camera_dirs = {path.parent for path in root.rglob("sync.log") if path.is_file()}
    return sorted(camera_dirs, key=lambda path: str(path))


def parse_sync_log(path: Path) -> SyncLogInfo:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(MATCHED_WINDOW_RE.finditer(text))
    window = _window_from_match(matches[-1]) if matches else None
    lines = text.splitlines()
    warnings = [line for line in lines if WARNING_RE.search(line)]
    errors = [line for line in lines if ERROR_RE.search(line)]
    non_forward = [line for line in warnings if NON_FORWARD_RE.search(line)]
    return SyncLogInfo(
        path=str(path),
        window=window,
        warnings=warnings,
        errors=errors,
        non_forward_warnings=non_forward,
    )


def probe_media(path: Path) -> MediaInfo:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,sample_rate,channels,duration,"
            "nb_frames,avg_frame_rate,r_frame_rate,width,height:format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")

    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    if not isinstance(video_stream, dict):
        raise RuntimeError(f"No video stream found in {path}")

    return MediaInfo(
        path=str(path),
        video_duration_sec=_positive_float(
            video_stream.get("duration"),
            payload.get("format", {}).get("duration"),
        ),
        frame_count=_nonnegative_int(video_stream.get("nb_frames")),
        avg_frame_rate=_optional_str(video_stream.get("avg_frame_rate")),
        r_frame_rate=_optional_str(video_stream.get("r_frame_rate")),
        audio_durations_sec=[
            value
            for value in (
                _positive_float(stream.get("duration"))
                for stream in streams
                if stream.get("codec_type") == "audio"
            )
            if value is not None
        ],
        audio_sample_rates=[
            value
            for value in (
                _nonnegative_int(stream.get("sample_rate"))
                for stream in streams
                if stream.get("codec_type") == "audio"
            )
            if value is not None
        ],
    )


def find_synced_video(camera_dir: Path) -> Path | None:
    synced_dir = camera_dir / "synced_video"
    candidates = sorted(synced_dir.glob("*_synced.mp4")) if synced_dir.is_dir() else []
    if not candidates:
        candidates = sorted(synced_dir.glob("*.mp4")) if synced_dir.is_dir() else []
    return candidates[0] if candidates else None


def find_anchor_json(camera_dir: Path) -> Path | None:
    preferred = [
        camera_dir / "work" / "gapfilled-filtered-padded-anchors.json",
        camera_dir / "work" / "gapfilled-filtered-anchors.json",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = sorted(camera_dir.rglob("*anchors.json"))
    return candidates[0] if candidates else None


def summarize_anchor_residuals(
    anchors_path: Path,
    window: SyncWindow,
    *,
    sample_rate: float,
) -> ResidualSummary | None:
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    if not isinstance(anchors, list):
        raise ValueError(f"Anchors JSON must contain a list: {anchors_path}")

    residuals = [
        _anchor_residual_ms(anchor, window, sample_rate)
        for anchor in anchors
        if isinstance(anchor, dict)
    ]
    residuals = [value for value in residuals if value is not None]
    if not residuals:
        return None

    abs_values = sorted(abs(value) for value in residuals)
    return ResidualSummary(
        path=str(anchors_path),
        anchor_count=len(abs_values),
        max_abs_ms=abs_values[-1],
        median_abs_ms=median(abs_values),
        p95_abs_ms=_percentile(abs_values, 0.95),
    )


def qc_camera_dir(
    camera_dir: Path,
    *,
    thresholds: QcThresholds = QcThresholds(),
) -> dict[str, Any]:
    segment = camera_dir.parent.name
    camera = camera_dir.name
    sync_log = parse_sync_log(camera_dir / "sync.log")
    video_path = find_synced_video(camera_dir)

    issues: list[dict[str, str]] = []
    if sync_log.window is None:
        issues.append(_issue(STATUS_FAIL, "sync_log", "Matched window not found."))
    if sync_log.errors:
        issues.append(
            _issue(
                STATUS_FAIL, "sync_log", f"{len(sync_log.errors)} error lines found."
            )
        )
    if sync_log.non_forward_warnings:
        issues.append(
            _issue(
                STATUS_FAIL,
                "sync_log",
                f"{len(sync_log.non_forward_warnings)} non-forward serial warning(s).",
            )
        )
    elif sync_log.warnings:
        issues.append(
            _issue(
                STATUS_WARN,
                "sync_log",
                f"{len(sync_log.warnings)} warning line(s) found.",
            )
        )

    media_info: MediaInfo | None = None
    residual_summary: ResidualSummary | None = None
    if video_path is None:
        issues.append(_issue(STATUS_FAIL, "media", "Synced MP4 not found."))
    else:
        media_info = probe_media(video_path)

    if sync_log.window and media_info:
        issues.extend(_window_media_issues(sync_log.window, media_info, thresholds))
        anchors_path = find_anchor_json(camera_dir)
        if anchors_path is not None:
            residual_summary = summarize_anchor_residuals(
                anchors_path,
                sync_log.window,
                sample_rate=sync_log.window.sample_rate_hz,
            )
            if residual_summary is not None:
                issues.extend(_residual_issues(residual_summary, thresholds))

    return {
        "segment": segment,
        "camera": camera,
        "status": _overall_status(issues),
        "camera_dir": str(camera_dir),
        "sync_log": asdict(sync_log),
        "media": asdict(media_info) if media_info else None,
        "anchor_residuals": asdict(residual_summary) if residual_summary else None,
        "issues": issues,
    }


def qc_output_dir(
    root: Path,
    *,
    thresholds: QcThresholds = QcThresholds(),
) -> dict[str, Any]:
    camera_dirs = discover_camera_dirs(root)
    results = [
        qc_camera_dir(
            camera_dir,
            thresholds=thresholds,
        )
        for camera_dir in camera_dirs
    ]
    if not camera_dirs:
        return {
            "root": str(root),
            "status": STATUS_FAIL,
            "counts": {STATUS_PASS: 0, STATUS_WARN: 0, STATUS_FAIL: 1},
            "results": [],
            "issues": [
                _issue(
                    STATUS_FAIL,
                    "discovery",
                    "No per-camera sync.log files found under output root.",
                )
            ],
        }
    counts = {
        status: sum(1 for result in results if result["status"] == status)
        for status in (STATUS_PASS, STATUS_WARN, STATUS_FAIL)
    }
    return {
        "root": str(root),
        "status": _overall_status(
            _issue(
                result["status"], "camera", f"{result['segment']}/{result['camera']}"
            )
            for result in results
        ),
        "counts": counts,
        "results": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QC synced video outputs using sync logs, final MP4 probes, and optional anchor residuals."
    )
    parser.add_argument("out_dir", help="Nested sync output root to scan.")
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    parser.add_argument("--out-json", help="Optional path to write JSON output.")
    parser.add_argument("--frame-fail", type=int, default=2)
    parser.add_argument("--duration-fail-ms", type=float, default=100.0)
    parser.add_argument("--residual-fail-ms", type=float, default=50.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    thresholds = QcThresholds(
        frame_fail=args.frame_fail,
        duration_fail_ms=args.duration_fail_ms,
        residual_fail_ms=args.residual_fail_ms,
    )
    root = Path(args.out_dir)
    try:
        result = qc_output_dir(
            root,
            thresholds=thresholds,
        )
    except FileNotFoundError:
        result = {
            "root": str(root),
            "status": STATUS_FAIL,
            "counts": {STATUS_PASS: 0, STATUS_WARN: 0, STATUS_FAIL: 1},
            "results": [],
            "issues": [_issue(STATUS_FAIL, "discovery", "Output root not found.")],
        }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_summary(result))
    return 1 if result["status"] == STATUS_FAIL else 0


def format_summary(result: dict[str, Any]) -> str:
    counts = result["counts"]
    lines = [
        f"Synced video QC: {result['status']}",
        f"Root: {result['root']}",
        f"PASS={counts[STATUS_PASS]} WARN={counts[STATUS_WARN]} FAIL={counts[STATUS_FAIL]}",
        "",
    ]
    for issue in result.get("issues", []):
        lines.append(f"{issue['status']}  {issue['message']}")
    for item in result["results"]:
        label = f"{item['segment']}/{item['camera']}"
        if not item["issues"]:
            lines.append(f"{item['status']}  {label}")
            continue
        messages = "; ".join(issue["message"] for issue in item["issues"])
        lines.append(f"{item['status']}  {label}  {messages}")
    return "\n".join(lines).rstrip()


def _window_from_match(match: re.Match[str]) -> SyncWindow:
    return SyncWindow(
        frame0=int(match.group("frame0")),
        frame1=int(match.group("frame1")),
        expected_frames=int(match.group("frames")),
        sample0=int(match.group("s0")),
        sample1=int(match.group("s1")),
        fps=float(match.group("fps")),
    )


def _window_media_issues(
    window: SyncWindow, media: MediaInfo, thresholds: QcThresholds
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if media.frame_count is not None:
        delta = media.frame_count - window.expected_frames
        abs_delta = abs(delta)
        if abs_delta > thresholds.frame_fail:
            issues.append(
                _issue(
                    STATUS_FAIL,
                    "frame_count",
                    f"decoded frame count differs by {delta} frame(s).",
                )
            )
        elif abs_delta > thresholds.frame_warn:
            issues.append(
                _issue(
                    STATUS_WARN,
                    "frame_count",
                    f"decoded frame count differs by {delta} frame(s).",
                )
            )

    if media.video_duration_sec is None:
        return issues

    expected_delta_ms = (media.video_duration_sec - window.duration_sec) * 1000.0
    issues.extend(
        _duration_issue("video duration vs sync window", expected_delta_ms, thresholds)
    )
    for audio_duration in media.audio_durations_sec:
        delta_ms = (audio_duration - media.video_duration_sec) * 1000.0
        issues.extend(_duration_issue("audio duration vs video", delta_ms, thresholds))
    return issues


def _residual_issues(
    residuals: ResidualSummary, thresholds: QcThresholds
) -> list[dict[str, str]]:
    if residuals.max_abs_ms > thresholds.residual_fail_ms:
        return [
            _issue(
                STATUS_FAIL,
                "anchor_residual",
                f"max anchor residual is {residuals.max_abs_ms:.1f} ms.",
            )
        ]
    if residuals.max_abs_ms > thresholds.residual_warn_ms:
        return [
            _issue(
                STATUS_WARN,
                "anchor_residual",
                f"max anchor residual is {residuals.max_abs_ms:.1f} ms.",
            )
        ]
    return []


def _duration_issue(
    label: str, delta_ms: float, thresholds: QcThresholds
) -> list[dict[str, str]]:
    abs_delta = abs(delta_ms)
    if abs_delta > thresholds.duration_fail_ms:
        return [
            _issue(STATUS_FAIL, "duration", f"{label} differs by {delta_ms:.1f} ms.")
        ]
    if abs_delta > thresholds.duration_warn_ms:
        return [
            _issue(STATUS_WARN, "duration", f"{label} differs by {delta_ms:.1f} ms.")
        ]
    return []


def _anchor_residual_ms(
    anchor: dict[str, Any], window: SyncWindow, sample_rate: float
) -> float | None:
    frame = _anchor_frame(anchor)
    audio_sample = _nonnegative_int(anchor.get("audio_sample"))
    if frame is None or audio_sample is None:
        return None
    if frame < window.frame0 or frame > window.frame1:
        return None
    frame_seconds = (frame - window.frame0) / window.fps
    audio_seconds = (audio_sample - window.sample0) / float(sample_rate)
    return (frame_seconds - audio_seconds) * 1000.0


def _anchor_frame(anchor: dict[str, Any]) -> int | None:
    for key in ("frame_index", "frame_id_reidx", "frame_id"):
        value = _nonnegative_int(anchor.get(key))
        if value is not None:
            return value
    return None


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return math.nan
    index = min(
        len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * quantile))
    )
    return sorted_values[index]


def _overall_status(issues: Iterable[dict[str, str]]) -> str:
    statuses = {issue["status"] for issue in issues}
    if STATUS_FAIL in statuses:
        return STATUS_FAIL
    if STATUS_WARN in statuses:
        return STATUS_WARN
    return STATUS_PASS


def _issue(status: str, check: str, message: str) -> dict[str, str]:
    return {"status": status, "check": check, "message": message}


def _positive_float(*values: Any) -> float | None:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    return None


def _nonnegative_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


if __name__ == "__main__":
    raise SystemExit(main())
