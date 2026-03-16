#!/usr/bin/env python3
"""Build manifest of NBU videos for FPS/filename remediation.

Scans datalake video directories, probes each MP4 for metadata,
pairs with companion JSON files for correct timestamps, and classifies
each file for re-encoding, remuxing, or copying.

Usage:
    python -m scripts.datalake.resave_videos.build_manifest \
        --roots /mnt/datalake/data/TRBD-53761 /mnt/datalake/data/AA-56119 \
        --out-root /mnt/new-datalake/NBU-video-recover \
        --output manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.index.common import DEFAULT_TZ
from scripts.index.filepatterns import FilePatterns
from scripts.parsers.jsonfileparser import JsonParser
from scripts.parsers.videofileparser import VideoFileParser

log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────

TARGET_FPS = 30.0
FPS_TOLERANCE = 1.0  # |fps - TARGET_FPS| > tolerance → re-encode
UTC = ZoneInfo("UTC")
_RE_TIMESTAMP_TAIL = re.compile(r"_(\d{8}_\d{6})$")
_RE_FIXED_SUFFIX = re.compile(r"_fixed\.mp4$", re.IGNORECASE)


# ── Data types ──────────────────────────────────────────────────────


@dataclass
class DirInfo:
    """Metadata about a discovered video directory."""

    root: Path
    root_name: str
    patient: str
    visit_date: str
    site: str
    path: Path


@dataclass
class VideoRecord:
    """One row of the output manifest."""

    src_path: str
    root_name: str
    patient: str
    visit_date: str
    site: str
    segment_id: str
    cam_serial: str
    filename_timestamp: str
    json_timestamp: str
    timestamp_mismatch: bool
    current_fps: float
    needs_reencode: bool
    has_companion_json: bool
    duration_sec: float
    frame_count: int
    action: str  # reencode | remux | copy | skip
    skip_reason: str
    dst_path: str


# ── Discovery ───────────────────────────────────────────────────────


LAYOUTS = ("nbu", "clinic")


def _discover_nbu(roots: list[Path]) -> list[DirInfo]:
    """Walk <root>/<patient>/NBU/<date>/video/<site>/."""
    dirs: list[DirInfo] = []
    for root in roots:
        if not root.is_dir():
            log.warning("Root not found: %s", root)
            continue
        for patient_dir in sorted(root.iterdir()):
            if not patient_dir.is_dir():
                continue
            nbu = patient_dir / "NBU"
            if not nbu.is_dir():
                continue
            for date_dir in sorted(nbu.iterdir()):
                if not date_dir.is_dir():
                    continue
                video_dir = date_dir / "video"
                if not video_dir.is_dir():
                    continue
                for site_dir in sorted(video_dir.iterdir()):
                    if not site_dir.is_dir():
                        continue
                    if not any(site_dir.glob("*.mp4")):
                        continue
                    dirs.append(
                        DirInfo(
                            root=root,
                            root_name=root.name,
                            patient=patient_dir.name,
                            visit_date=date_dir.name,
                            site=site_dir.name,
                            path=site_dir,
                        )
                    )
    return dirs


def _discover_clinic(roots: list[Path]) -> list[DirInfo]:
    """Walk <root>/<patient>/clinic/<date>/video/FLIR/."""
    dirs: list[DirInfo] = []
    for root in roots:
        if not root.is_dir():
            log.warning("Root not found: %s", root)
            continue
        for patient_dir in sorted(root.iterdir()):
            if not patient_dir.is_dir():
                continue
            clinic = patient_dir / "clinic"
            if not clinic.is_dir():
                continue
            for date_dir in sorted(clinic.iterdir()):
                if not date_dir.is_dir():
                    continue
                flir_dir = date_dir / "video" / "FLIR"
                if not flir_dir.is_dir():
                    continue
                if not any(flir_dir.glob("*.mp4")):
                    continue
                dirs.append(
                    DirInfo(
                        root=root,
                        root_name=root.name,
                        patient=patient_dir.name,
                        visit_date=date_dir.name,
                        site="FLIR",
                        path=flir_dir,
                    )
                )
    return dirs


def discover_video_dirs(roots: list[Path], layout: str = "nbu") -> list[DirInfo]:
    """Discover video directories using the given layout."""
    if layout == "nbu":
        dirs = _discover_nbu(roots)
    elif layout == "clinic":
        dirs = _discover_clinic(roots)
    else:
        raise ValueError(f"Unknown layout: {layout!r}. Choose from: {LAYOUTS}")
    log.info("Discovered %d video directories (layout=%s)", len(dirs), layout)
    return dirs


# ── JSON handling ───────────────────────────────────────────────────


def build_json_map(video_dir: Path) -> dict[str, Path]:
    """Map segment_id -> json_path. Falls back to sibling dir for *_recovered."""
    result: dict[str, Path] = {}
    for jp in video_dir.glob("*.json"):
        seg_id = FilePatterns.parse_json_filename(jp)
        if seg_id:
            result[seg_id] = jp
    # For _recovered dirs with no JSONs, check the original sibling
    if not result and video_dir.name.endswith("_recovered"):
        sibling = video_dir.parent / video_dir.name.removesuffix("_recovered")
        if sibling.is_dir():
            for jp in sibling.glob("*.json"):
                seg_id = FilePatterns.parse_json_filename(jp)
                if seg_id:
                    result[seg_id] = jp
    return result


def _json_start_to_chicago(json_path: Path) -> str | None:
    """Extract real_times[0] from JSON, convert UTC -> America/Chicago timestamp."""
    try:
        utc_dt = JsonParser(json_path).get_start_realtime()
        if utc_dt is None:
            return None
        chicago = utc_dt.replace(tzinfo=UTC).astimezone(DEFAULT_TZ)
        return chicago.strftime("%Y%m%d_%H%M%S")
    except Exception as e:
        log.debug("JSON parse failed %s: %s", json_path, e)
        return None


def precompute_json_timestamps(json_map: dict[str, Path]) -> dict[str, str]:
    """Batch-extract corrected timestamps from all JSONs in a directory."""
    return {
        seg_id: ts
        for seg_id, path in json_map.items()
        if (ts := _json_start_to_chicago(path)) is not None
    }


# ── Video probing ───────────────────────────────────────────────────


def probe_video(path: Path) -> dict:
    """Probe video via ffprobe. Returns {fps, duration, frame_count} or {error}."""
    try:
        vfp = VideoFileParser(path)
        return {
            "fps": vfp.fps,
            "duration": vfp.duration,
            "frame_count": vfp.frame_count,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Classification & path helpers ───────────────────────────────────


def _extract_prefix(segment_id: str) -> str | None:
    """Extract prefix before the _YYYYMMDD_HHMMSS tail."""
    m = _RE_TIMESTAMP_TAIL.search(segment_id)
    return segment_id[: m.start()] if m else None


def _build_dst_path(
    src: Path,
    dir_info: DirInfo,
    out_root: Path,
    segment_id: str,
    cam_serial: str,
    correct_ts: str | None,
) -> str:
    """Build output path preserving directory structure, with corrected filename."""
    prefix = _extract_prefix(segment_id)
    if correct_ts and prefix:
        name = f"{prefix}_{correct_ts}.{cam_serial}.mp4"
    else:
        name = src.name
    rel = src.parent.relative_to(dir_info.root)
    return str(out_root / dir_info.root_name / rel / name)


def _classify(has_json: bool, fps: float, ts_mismatch: bool) -> tuple[str, str]:
    """Return (action, skip_reason) for a video file."""
    if not has_json:
        return "skip", "no_companion_json"
    if abs(fps - TARGET_FPS) > FPS_TOLERANCE:
        return "reencode", ""
    if ts_mismatch:
        return "remux", ""
    return "copy", ""


# ── Per-file / per-directory processing ─────────────────────────────


def _make_record(dir_info: DirInfo, mp4: Path, **kwargs) -> VideoRecord:
    """Build a VideoRecord with directory-level defaults, overridden by kwargs."""
    defaults = dict(
        src_path=str(mp4),
        root_name=dir_info.root_name,
        patient=dir_info.patient,
        visit_date=dir_info.visit_date,
        site=dir_info.site,
        segment_id="",
        cam_serial="",
        filename_timestamp="",
        json_timestamp="",
        timestamp_mismatch=False,
        current_fps=0.0,
        needs_reencode=False,
        has_companion_json=False,
        duration_sec=0.0,
        frame_count=0,
        action="skip",
        skip_reason="",
        dst_path="",
    )
    defaults.update(kwargs)
    return VideoRecord(**defaults)


def _process_file(
    mp4: Path,
    json_ts: dict[str, str],
    dir_info: DirInfo,
    out_root: Path,
) -> VideoRecord:
    """Process a single MP4 into a manifest record."""
    parsed = FilePatterns.parse_video_filename(mp4)
    # Handle *_fixed.mp4 from previous manual recovery — strip suffix and re-parse
    if not parsed and _RE_FIXED_SUFFIX.search(mp4.name):
        clean = mp4.with_name(_RE_FIXED_SUFFIX.sub(".mp4", mp4.name))
        parsed = FilePatterns.parse_video_filename(clean)
    if not parsed:
        return _make_record(dir_info, mp4, skip_reason="unparseable_filename")

    seg_id, serial = parsed
    fn_dt = FilePatterns.parse_tail_datetime(seg_id)
    fn_ts = fn_dt.strftime("%Y%m%d_%H%M%S") if fn_dt else ""

    j_ts = json_ts.get(seg_id, "")
    has_json = bool(j_ts)

    probe = probe_video(mp4)
    if "error" in probe:
        return _make_record(
            dir_info,
            mp4,
            segment_id=seg_id,
            cam_serial=serial,
            filename_timestamp=fn_ts,
            json_timestamp=j_ts,
            has_companion_json=has_json,
            skip_reason=f"ffprobe_failed: {probe['error']}",
        )

    fps = probe["fps"]
    ts_mismatch = has_json and j_ts != fn_ts
    action, reason = _classify(has_json, fps, ts_mismatch)
    dst = (
        _build_dst_path(mp4, dir_info, out_root, seg_id, serial, j_ts)
        if action != "skip"
        else ""
    )

    return _make_record(
        dir_info,
        mp4,
        segment_id=seg_id,
        cam_serial=serial,
        filename_timestamp=fn_ts,
        json_timestamp=j_ts,
        timestamp_mismatch=ts_mismatch,
        current_fps=fps,
        needs_reencode=abs(fps - TARGET_FPS) > FPS_TOLERANCE,
        has_companion_json=has_json,
        duration_sec=probe["duration"],
        frame_count=probe["frame_count"],
        action=action,
        skip_reason=reason,
        dst_path=dst,
    )


def process_directory(
    dir_info: DirInfo, out_root: Path, cache: dict[str, VideoRecord] | None = None
) -> list[VideoRecord]:
    """Process all MP4s in one video directory, using cache to skip ffprobe."""
    mp4s = sorted(dir_info.path.glob("*.mp4"))

    # Fast path: if every file is cached, skip all JSON parsing and probing
    if cache and all(str(mp4) in cache for mp4 in mp4s):
        return [cache[str(mp4)] for mp4 in mp4s]

    # Some files need processing — parse JSONs for timestamp lookup
    json_map = build_json_map(dir_info.path)
    json_ts = precompute_json_timestamps(json_map)
    results: list[VideoRecord] = []
    for mp4 in mp4s:
        cached = cache.get(str(mp4)) if cache else None
        if cached is not None:
            results.append(cached)
        else:
            results.append(_process_file(mp4, json_ts, dir_info, out_root))
    return results


# ── Output ──────────────────────────────────────────────────────────


def _write_csv(records: list[VideoRecord], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(VideoRecord)]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def _row_to_record(row: dict) -> VideoRecord:
    """Convert a CSV row dict back into a VideoRecord."""
    return VideoRecord(
        src_path=row["src_path"],
        root_name=row["root_name"],
        patient=row["patient"],
        visit_date=row["visit_date"],
        site=row["site"],
        segment_id=row["segment_id"],
        cam_serial=row["cam_serial"],
        filename_timestamp=row["filename_timestamp"],
        json_timestamp=row["json_timestamp"],
        timestamp_mismatch=row["timestamp_mismatch"] == "True",
        current_fps=float(row["current_fps"] or 0),
        needs_reencode=row["needs_reencode"] == "True",
        has_companion_json=row["has_companion_json"] == "True",
        duration_sec=float(row["duration_sec"] or 0),
        frame_count=int(row["frame_count"] or 0),
        action=row["action"],
        skip_reason=row["skip_reason"],
        dst_path=row["dst_path"],
    )


def _load_cache(output: Path) -> dict[str, VideoRecord]:
    """Load previous manifest + skipped CSVs as probe cache keyed by src_path.

    Skips 'unparseable_filename' entries so they get re-processed (code changes
    may now parse them, e.g. *_fixed.mp4 support).
    """
    cache: dict[str, VideoRecord] = {}
    for suffix in ("", "_skipped"):
        p = output.with_name(f"{output.stem}{suffix}{output.suffix}")
        if not p.exists():
            continue
        with open(p) as fh:
            for row in csv.DictReader(fh):
                if row.get("skip_reason") == "unparseable_filename":
                    continue
                cache[row["src_path"]] = _row_to_record(row)
        log.info("Cache: loaded %d entries from %s", len(cache), p.name)
    return cache


def _print_summary(records: list[VideoRecord]) -> None:
    by_action: dict[str, list[VideoRecord]] = {}
    for r in records:
        by_action.setdefault(r.action, []).append(r)

    total_hrs = sum(r.duration_sec for r in records if r.duration_sec > 0) / 3600

    print(f"\n{'=' * 50}")
    print(f"Total: {len(records)} files, {total_hrs:.1f} hours")
    for action in ("reencode", "remux", "copy", "skip"):
        items = by_action.get(action, [])
        hrs = sum(r.duration_sec for r in items if r.duration_sec > 0) / 3600
        print(f"  {action:10s}: {len(items):6d} files  ({hrs:.1f} h)")

    skip_reasons: dict[str, int] = {}
    for r in by_action.get("skip", []):
        skip_reasons[r.skip_reason] = skip_reasons.get(r.skip_reason, 0) + 1
    if skip_reasons:
        print("Skip breakdown:")
        for reason, n in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {n}")


# ── Orchestration ───────────────────────────────────────────────────


def build_manifest(
    roots: list[Path],
    out_root: Path,
    output: Path,
    workers: int,
    use_cache: bool = False,
    layout: str = "nbu",
) -> None:
    """Discover -> scan -> classify -> write manifest."""
    dirs = discover_video_dirs(roots, layout=layout)
    if not dirs:
        log.error("No video directories found")
        sys.exit(1)

    cache: dict[str, VideoRecord] | None = None
    if use_cache:
        cache = _load_cache(output)
        if cache:
            log.info("Probe cache: %d files (will skip ffprobe for these)", len(cache))

    all_records: list[VideoRecord] = []
    done = 0
    total = len(dirs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process_directory, d, out_root, cache): d for d in dirs}
        for fut in as_completed(futs):
            d = futs[fut]
            done += 1
            try:
                recs = fut.result()
                all_records.extend(recs)
                log.info(
                    "[%d/%d] %s/%s/%s — %d files",
                    done,
                    total,
                    d.patient,
                    d.visit_date,
                    d.site,
                    len(recs),
                )
            except Exception:
                log.exception("[%d/%d] FAILED %s", done, total, d.path)

    manifest = [r for r in all_records if r.action != "skip"]
    skipped = [r for r in all_records if r.action == "skip"]

    _write_csv(manifest, output)
    log.info("Manifest: %d records -> %s", len(manifest), output)

    skip_path = output.with_name(f"{output.stem}_skipped{output.suffix}")
    _write_csv(skipped, skip_path)
    log.info("Skipped: %d records -> %s", len(skipped), skip_path)

    # Check for dst_path collisions (e.g., DST fall-back producing duplicate names)
    dst_counts: dict[str, list[str]] = {}
    for r in manifest:
        dst_counts.setdefault(r.dst_path, []).append(r.src_path)
    collisions = {dst: srcs for dst, srcs in dst_counts.items() if len(srcs) > 1}
    if collisions:
        log.warning("%d destination path collisions detected!", len(collisions))
        collision_path = output.with_name(f"{output.stem}_collisions{output.suffix}")
        with open(collision_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dst_path", "src_paths"])
            for dst, srcs in sorted(collisions.items()):
                w.writerow([dst, "; ".join(srcs)])
        log.warning("Collisions written to %s", collision_path)

    _print_summary(all_records)


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build manifest for NBU video FPS/filename remediation"
    )
    ap.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        required=True,
        help="Datalake root directories to scan",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root for fixed videos",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("manifest.csv"),
        help="Manifest CSV output path (default: manifest.csv)",
    )
    ap.add_argument(
        "--layout",
        default="nbu",
        choices=LAYOUTS,
        help="Directory layout to scan (default: nbu)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel directory workers (default: 8)",
    )
    ap.add_argument(
        "--cache",
        action="store_true",
        help="Reuse probe results from previous manifest/skipped CSVs at --output path",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    build_manifest(
        args.roots, args.out_root, args.output, args.workers, args.cache, args.layout
    )


if __name__ == "__main__":
    main()
