#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
videoplanapplier.py
===================

Apply a *video padding plan* (from videoplancreater.py) to a source MP4 using a
streaming, O(1)-memory raw-video pipe:

    FFmpeg (decode → raw yuv420p) → Python (insert frames per plan) → FFmpeg (encode)

Why this design
---------------
- Scales to thousands of insertions without gigantic filtergraphs.
- Deterministic, research-friendly (no motion hallucination).
- Memory-efficient (only one frame resident at a time).

Supported policies
------------------
- "dup-prev" (default): duplicate the last emitted frame.
- "black": insert black frames (Y=16, U=128, V=128 in yuv420p) for conspicuous QC.

Notes
-----
- Input videos are assumed to be **video-only** (no audio), as in your pipeline.
- Output video is encoded with libx264, yuv420p, CFR at `target_fps` (from plan or override).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.log.logutils import configure_standalone_logging, log_context
from scripts.parsers.videofileparser import VideoFileParser
from scripts.models import Video

# Module-level logger (handlers/formatting controlled by logutils)
log = logging.getLogger(__name__)


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class PlanOp:
    after_index: int
    insert: int
    frame_id_before: int
    frame_id_after: int


@dataclass(frozen=True)
class VideoPadPlan:
    version: int
    segment_id: str
    cam_serial: str
    source_video: str
    source_fps: float
    target_fps: float
    policy: str
    total_insertions: int
    operations: List[PlanOp]


# -----------------------------
# Helpers
# -----------------------------
def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Please install FFmpeg.")


def _load_plan(plan_json: Path) -> VideoPadPlan:
    data = json.loads(plan_json.read_text(encoding="utf-8"))
    required = [
        "version",
        "segment_id",
        "cam_serial",
        "source_video",
        "source_fps",
        "target_fps",
        "policy",
        "total_insertions",
        "operations",
    ]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key {k!r} in plan JSON: {plan_json}")

    ops: List[PlanOp] = []
    for o in data["operations"]:
        try:
            ops.append(
                PlanOp(
                    after_index=int(o["after_index"]),
                    insert=int(o["insert"]),
                    frame_id_before=int(o["frame_id_before"]),
                    frame_id_after=int(o["frame_id_after"]),
                )
            )
        except Exception as e:
            raise ValueError(f"Invalid operation entry {o!r}: {e}")

    return VideoPadPlan(
        version=int(data["version"]),
        segment_id=str(data["segment_id"]),
        cam_serial=str(data["cam_serial"]),
        source_video=str(data["source_video"]),
        source_fps=float(data["source_fps"]),
        target_fps=float(data["target_fps"]),
        policy=str(data["policy"]),
        total_insertions=int(data["total_insertions"]),
        operations=ops,
    )


def _validate_policy(policy: str) -> None:
    if policy not in {"dup-prev", "black"}:
        raise ValueError(f"Unsupported policy: {policy!r}. Supported: dup-prev, black.")


def _validate_operations(ops: List[PlanOp], frame_count: int) -> None:
    if any(op.insert < 0 for op in ops):
        raise ValueError("Plan contains a negative 'insert' count.")
    # after_index may be equal to last frame (padding at tail is allowed)
    for op in ops:
        if not (0 <= op.after_index <= frame_count - 1):
            raise ValueError(
                f"Operation after_index={op.after_index} out of range for frame_count={frame_count}."
            )
    # ensure non-decreasing (good hygiene; creator already sorts)
    last = -1
    for op in sorted(ops, key=lambda x: x.after_index):
        if op.after_index < last:
            raise ValueError("Operations are not sorted by after_index.")
        last = op.after_index


def _dup_map(ops: Sequence[PlanOp]) -> Dict[int, int]:
    from collections import defaultdict

    m = defaultdict(int)
    for op in ops:
        # treat plan.after_index as i, and insert AFTER i-1
        m[int(op.after_index) - 1] += int(op.insert)
    return dict(m)


def _spawn_decoder(video: Path) -> subprocess.Popen:
    # Decode to raw yuv420p frames, one frame per output (no vsync fiddling)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "0",
        "-",  # stdout raw frames
    ]
    log.debug("Decoder cmd: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _spawn_encoder(
    out_path: Path, width: int, height: int, target_fps: float, crf: int, preset: str
) -> subprocess.Popen:
    # With raw input, specify -r BEFORE -i to interpret incoming frames at target_fps (CFR).
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{target_fps:.9f}",
        "-i",
        "-",  # stdin raw frames
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        str(preset),
        "-crf",
        str(crf),
        str(out_path),
    ]
    log.debug("Encoder cmd: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes or raise EOFError."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            # EOF mid-frame
            raise EOFError(
                f"Unexpected EOF while reading frame bytes (wanted {n}, got {got})."
            )
        view[got : got + len(chunk)] = chunk
        got += len(chunk)
    return bytes(buf)


def _make_black_frame(width: int, height: int) -> bytes:
    """TV-range black in yuv420p: Y=16, U=128, V=128."""
    y = bytes([16]) * (width * height)
    u = bytes([128]) * ((width // 2) * (height // 2))
    v = bytes([128]) * ((width // 2) * (height // 2))
    return y + u + v


def _close_process(proc: subprocess.Popen, *, name: str) -> None:
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
        proc.wait()
    if proc.returncode != 0:
        try:
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        except Exception:
            err = ""
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}. {err}")


# -----------------------------
# Public API
# -----------------------------
def apply_video_padding_plan(
    plan_json: Path,
    video: Video,
    out_dir: Path,
    *,
    crf: int = 18,
    preset: str = "veryfast",
    override_target_fps: Optional[float] = None,
    override_policy: Optional[str] = None,
    progress_every: int = 2000,
) -> Tuple[Path, Video]:
    """
    Apply a videopad plan to `video`, write the padded MP4 to `out_dir`
    under the same filename as the source, and return:
      (output_path, NEW Video with updated metadata and padded CamJson arrays).

    Returns
    -------
    (Path, Video)
        Output MP4 path and an updated, immutable Video object whose
        companion_json arrays (fixed_serials, fixed_frame_ids, fixed_reidx_frame_ids)
        are padded to match the inserted frames.
    """
    _require_ffmpeg()
    plan = _load_plan(Path(plan_json))

    # Introspect from Video object
    video_path = Path(video.path)
    res_str = str(getattr(video, "resolution", "") or "")
    try:
        src_w, src_h = [int(x) for x in res_str.lower().split("x", 1)]
    except Exception as e:
        raise RuntimeError(
            f"Invalid video.resolution='{res_str}' for {video_path.name}"
        ) from e
    src_fps = float(getattr(video, "frame_rate", 0.0) or 0.0)
    src_frames = int(getattr(video, "frame_count", 0) or 0)
    if src_fps <= 0 or src_frames <= 0:
        raise RuntimeError(
            f"Video meta incomplete (fps={src_fps}, frames={src_frames}) for {video_path.name}"
        )

    # Sanity checks and setup
    target_fps = (
        float(override_target_fps) if override_target_fps else float(plan.target_fps)
    )
    policy = str(override_policy) if override_policy else str(plan.policy)
    _validate_policy(policy)
    _validate_operations(plan.operations, frame_count=src_frames)

    dup_after = _dup_map(plan.operations)  # {src_index: insert_count}
    expected_out_frames = src_frames + plan.total_insertions

    # Log context
    adapter = logging.LoggerAdapter(
        log, extra={"seg": plan.segment_id, "cam": plan.cam_serial}
    )
    with log_context(seg=plan.segment_id, cam=plan.cam_serial):
        # Warn on FPS mismatch (plan vs actual)
        if not math.isclose(src_fps, float(plan.source_fps), rel_tol=0, abs_tol=1e-3):
            adapter.warning(
                "Source FPS (video=%.6f) differs from plan.source_fps (%.6f).",
                src_fps,
                float(plan.source_fps),
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / video_path.name

        adapter.info(
            "Applying padding (policy=%s): src=%s (%dx%d @ %.6f, frames=%d) → tgt_fps=%.6f, total_insertions=%d, expected_out_frames=%d",
            policy,
            video_path.name,
            src_w,
            src_h,
            src_fps,
            src_frames,
            target_fps,
            plan.total_insertions,
            expected_out_frames,
        )

        # Spawn processes
        dec = _spawn_decoder(video_path)
        enc = _spawn_encoder(out_path, src_w, src_h, target_fps, crf=crf, preset=preset)
        if not dec.stdout or not enc.stdin:
            dec.kill()
            enc.kill()
            raise RuntimeError("Failed to connect FFmpeg pipes.")

        frame_size = (src_w * src_h * 3) // 2  # yuv420p
        black_frame = _make_black_frame(src_w, src_h) if policy == "black" else None

        # Stream frames
        written = 0
        try:
            for idx in range(src_frames):
                # Read one source frame
                frame = _read_exact(dec.stdout, frame_size)
                # Emit the source frame
                enc.stdin.write(frame)
                written += 1

                # Insert per plan after this index
                ins = dup_after.get(idx, 0)
                if ins > 0:
                    if policy == "dup-prev":
                        # repeat the same frame
                        for _ in range(ins):
                            enc.stdin.write(frame)
                        written += ins
                    elif policy == "black":
                        assert black_frame is not None
                        for _ in range(ins):
                            enc.stdin.write(black_frame)
                        written += ins

                if progress_every and (idx + 1) % progress_every == 0:
                    adapter.info(
                        "Progress: %d/%d input frames processed (output so far: %d)",
                        idx + 1,
                        src_frames,
                        written,
                    )

        finally:
            # Close encoder stdin to flush & finalize
            try:
                enc.stdin.close()
            except Exception:
                pass

            # Drain/close decoder stdout
            try:
                if dec.stdout:
                    dec.stdout.close()
            except Exception:
                pass

        # Wait for processes and check status
        _close_process(dec, name="Decoder (ffmpeg)")
        _close_process(enc, name="Encoder (ffmpeg)")

        # Validate output frame count
        try:
            vout = VideoFileParser(str(out_path))
            out_frames = int(vout.frame_count)
            if out_frames != expected_out_frames:
                adapter.warning(
                    "Output frame count (%d) differs from expected (%d). Check rounding / plan indices.",
                    out_frames,
                    expected_out_frames,
                )
            else:
                adapter.info("Output frame count verified: %d", out_frames)
        except Exception as e:
            adapter.warning("Could not parse output for verification: %s", e)
            out_frames = expected_out_frames  # fallback

        # ------------------------------------------------------------------
        # Construct NEW CamJson and NEW Video (no in-place mutation)
        # ------------------------------------------------------------------
        cj = getattr(video, "companion_json", None)
        if cj is None:
            raise RuntimeError(
                "Video has no companion_json; cannot construct updated Video."
            )

        serials = list(getattr(cj, "fixed_serials", None) or [])
        fids_u = list(getattr(cj, "fixed_frame_ids", None) or [])
        fids_r = list(getattr(cj, "fixed_reidx_frame_ids", None) or [])
        if not (len(serials) == len(fids_u) == len(fids_r) == src_frames):
            raise RuntimeError(
                f"CamJson array lengths ({len(serials)}/{len(fids_u)}/{len(fids_r)}) "
                f"!= src_frames ({src_frames}); cannot construct updated arrays."
            )

        padded_serials: list[int] = []
        padded_fids_u: list[int] = []
        padded_fids_r: list[int] = []
        for idx in range(src_frames):
            # original frame
            padded_serials.append(int(serials[idx]))
            padded_fids_u.append(int(fids_u[idx]))
            padded_fids_r.append(int(fids_r[idx]))
            # insertions after this index
            ins = dup_after.get(idx, 0)
            if ins > 0:
                last_u = int(fids_u[idx])
                last_r = int(fids_r[idx])
                for k in range(1, ins + 1):
                    padded_serials.append(0)  # mark synthetic frames
                    padded_fids_u.append(last_u + k)  # keep +1 progression
                    padded_fids_r.append(last_r + k)

        new_cj = replace(
            cj,
            fixed_serials=padded_serials,
            fixed_frame_ids=padded_fids_u,
            fixed_reidx_frame_ids=padded_fids_r,
        )

        # Use the actual encoded fps (target_fps) in the new Video
        new_duration = out_frames / float(target_fps)
        new_video = replace(
            video,
            path=out_path,
            frame_rate=float(target_fps),
            frame_count=int(out_frames),
            duration=float(new_duration),
            companion_json=new_cj,
        )

        adapter.info("Output written: %s (new Video constructed)", out_path)
        return out_path, new_video


# -----------------------------
# CLI
# -----------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Apply a video padding plan JSON to an MP4 via a streaming raw-video pipe."
    )
    ap.add_argument("plan_json", help="Path to <video_stem>-videopad.json")
    ap.add_argument("video", help="Path to source MP4 (video-only)")
    ap.add_argument(
        "--outdir",
        required=True,
        help="Directory where the padded MP4 will be written (same filename as input).",
    )
    ap.add_argument(
        "--policy",
        choices=["dup-prev", "black"],
        default=None,
        help="Override plan.policy (default: use the plan's policy).",
    )
    ap.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Override plan.target_fps (float). If omitted, use plan value.",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF quality (lower is higher quality). Default: 18.",
    )
    ap.add_argument(
        "--preset",
        default="veryfast",
        help="x264 preset, e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Log progress every N input frames (0 to disable). Default: 2000.",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    # Standalone console logger (no-op if driver already configured root)
    configure_standalone_logging(args.log_level, seg="-", cam="-")
    try:
        out = apply_video_padding_plan(
            Path(args.plan_json),
            Path(args.video),
            Path(args.outdir),
            crf=int(args.crf),
            preset=str(args.preset),
            override_target_fps=float(args.target_fps) if args.target_fps else None,
            override_policy=str(args.policy) if args.policy else None,
            progress_every=(
                int(args.progress_every) if args.progress_every is not None else 2000
            ),
        )
        log.info("Done: %s", out)
        return 0
    except Exception as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
