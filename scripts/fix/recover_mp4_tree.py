#!/usr/bin/env python3
"""
recover_mp4_tree.py — Recover corrupted MP4s across the TRBD datalake tree.

Scans:
  <root>/<patient>/NBU/<YYYY-MM-DD>/video/<site>

For each discovered video directory:
  - Detect corrupted MP4s (missing moov atom)
  - Check whether corresponding *_fixed.mp4 already exists in sibling
    <site>_recovered
  - Recover only missing outputs by invoking scripts.fix.recover_mp4

Writes one aggregated run log (JSON + CSV) with per-directory and total stats.

Examples:
  # Dry run (scan + quantify only)
  python -m scripts.fix.recover_mp4_tree

  # Execute recovery
  python -m scripts.fix.recover_mp4_tree --run

  # Limit scope
  python -m scripts.fix.recover_mp4_tree --patients TRBD001 --sites lounge --run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.fix.recover_mp4 import scan_directory
from scripts.index.filepatterns import FilePatterns
from scripts.log.logutils import configure_standalone_logging

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_list(values: list[str]) -> list[str]:
    """Split comma-separated and space-separated values, de-duplicated."""
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        for part in item.split(","):
            token = part.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def _cam_from_path(path: Path) -> str:
    parsed = FilePatterns.parse_video_filename(path)
    if parsed:
        return parsed[1]
    parts = path.stem.rsplit(".", 1)
    return parts[-1] if len(parts) >= 2 else ""


def _discover_video_dirs(
    root: Path,
    patients: list[str],
    sites: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for patient in patients:
        nbu_dir = root / patient / "NBU"
        if not nbu_dir.is_dir():
            log.warning("Missing NBU directory: %s", nbu_dir)
            continue

        for date_dir in sorted(p for p in nbu_dir.iterdir() if p.is_dir()):
            video_root = date_dir / "video"
            if not video_root.is_dir():
                continue

            for site in sites:
                video_dir = video_root / site
                if not video_dir.is_dir():
                    continue
                records.append(
                    {
                        "patient": patient,
                        "date": date_dir.name,
                        "site": site,
                        "video_dir": video_dir,
                        "output_dir": video_root / f"{site}_recovered",
                    }
                )
    return records


def _report_status_counts(report_path: Path) -> dict[str, int]:
    """Read a per-directory recovery_report.json and count file statuses."""
    if not report_path.exists():
        return {}

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        files = payload.get("files", [])
        counts = Counter(str(entry.get("status", "unknown")) for entry in files)
        return {key: int(value) for key, value in counts.items()}
    except Exception as exc:
        log.warning("Could not parse report %s: %s", report_path, exc)
        return {}


def _invoke_recover_mp4(
    python_exec: str,
    input_dir: Path,
    output_dir: Path,
    cam: str,
    log_level: str,
) -> tuple[int, str]:
    cmd = [
        python_exec,
        "-m",
        "scripts.fix.recover_mp4",
        str(input_dir),
        str(output_dir),
        "--run",
        "--log-level",
        log_level,
    ]
    if cam:
        cmd.extend(["--cam", cam])

    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr_lines = result.stderr.strip().splitlines()
    stdout_lines = result.stdout.strip().splitlines()
    tail = ""
    if stderr_lines:
        tail = stderr_lines[-1]
    elif stdout_lines:
        tail = stdout_lines[-1]
    return result.returncode, tail


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("~/mnt/datalake/TRBD-53761").expanduser(),
        help="Root datalake path (default: ~/mnt/datalake/TRBD-53761)",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["TRBD001", "TRBD002"],
        help="Patient IDs; supports comma-separated and/or space-separated values",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["lounge", "sleep"],
        help="Site names; supports comma-separated and/or space-separated values",
    )
    parser.add_argument(
        "--cam",
        type=str,
        default="",
        help="Optional camera serial filter forwarded to recover_mp4",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute recovery; default is scan-only dry run",
    )
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke recover_mp4 (default: current Python)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for aggregated logs (default: <root>/recovery_logs)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    configure_standalone_logging(level=args.log_level)

    root = args.root.expanduser().resolve()
    patients = _normalize_list(args.patients)
    sites = _normalize_list(args.sites)
    log_dir = (
        args.log_dir.expanduser().resolve() if args.log_dir else root / "recovery_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    run_started_at = _now_iso()
    discovered = _discover_video_dirs(root, patients, sites)
    log.info("Discovered %d video directories.", len(discovered))

    dir_records: list[dict[str, Any]] = []
    summary_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()

    for entry in discovered:
        patient = entry["patient"]
        date = entry["date"]
        site = entry["site"]
        video_dir: Path = entry["video_dir"]
        output_dir: Path = entry["output_dir"]

        try:
            corrupted, good, _ = scan_directory(video_dir)
            total_mp4s = len(corrupted) + len(good)
        except Exception as exc:
            summary_counts["dirs_scan_failed"] += 1
            dir_records.append(
                {
                    "patient": patient,
                    "date": date,
                    "site": site,
                    "video_dir": str(video_dir),
                    "output_dir": str(output_dir),
                    "total_mp4s": 0,
                    "good_mp4s": 0,
                    "corrupted_mp4s": 0,
                    "targeted_corrupted_mp4s": 0,
                    "already_fixed_outputs": 0,
                    "pending_outputs_before": 0,
                    "recover_invocation_rc": None,
                    "recover_invocation_note": str(exc),
                    "status": "scan_failed",
                    "report_recovered": 0,
                    "report_failed": 0,
                    "report_empty_stub": 0,
                    "report_no_reference": 0,
                    "report_pending": 0,
                    "report_recovered_status": 0,
                    "report_newly_recovered_est": 0,
                }
            )
            continue

        already_fixed = 0
        pending_outputs = 0
        targeted_corrupted = (
            [p for p in corrupted if _cam_from_path(p) == args.cam]
            if args.cam
            else corrupted
        )

        for src in targeted_corrupted:
            fixed_path = output_dir / f"{src.stem}_fixed.mp4"
            if fixed_path.exists():
                already_fixed += 1
            else:
                pending_outputs += 1

        record: dict[str, Any] = {
            "patient": patient,
            "date": date,
            "site": site,
            "video_dir": str(video_dir),
            "output_dir": str(output_dir),
            "total_mp4s": total_mp4s,
            "good_mp4s": len(good),
            "corrupted_mp4s": len(corrupted),
            "targeted_corrupted_mp4s": len(targeted_corrupted),
            "already_fixed_outputs": already_fixed,
            "pending_outputs_before": pending_outputs,
            "recover_invocation_rc": None,
            "recover_invocation_note": "",
            "status": "",
            "report_recovered": 0,
            "report_failed": 0,
            "report_empty_stub": 0,
            "report_no_reference": 0,
            "report_pending": 0,
            "report_recovered_status": 0,
            "report_newly_recovered_est": 0,
        }

        file_counts["total_mp4s"] += total_mp4s
        file_counts["good_mp4s"] += len(good)
        file_counts["corrupted_mp4s"] += len(corrupted)
        file_counts["targeted_corrupted_mp4s"] += len(targeted_corrupted)
        file_counts["already_fixed_outputs"] += already_fixed
        file_counts["pending_outputs_before"] += pending_outputs

        if total_mp4s == 0:
            record["status"] = "no_mp4"
            summary_counts["dirs_no_mp4"] += 1
            dir_records.append(record)
            continue

        if not corrupted:
            record["status"] = "clean"
            summary_counts["dirs_clean"] += 1
            dir_records.append(record)
            continue

        summary_counts["dirs_with_any_corruption"] += 1

        if not targeted_corrupted:
            record["status"] = "no_target_corruption"
            summary_counts["dirs_no_target_corruption"] += 1
            dir_records.append(record)
            continue

        summary_counts["dirs_with_target_corruption"] += 1

        if pending_outputs == 0:
            record["status"] = "already_recovered"
            summary_counts["dirs_already_recovered"] += 1
            dir_records.append(record)
            continue

        if not args.run:
            record["status"] = "needs_recovery"
            summary_counts["dirs_needing_recovery"] += 1
            dir_records.append(record)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        rc, tail = _invoke_recover_mp4(
            python_exec=args.python_exec,
            input_dir=video_dir,
            output_dir=output_dir,
            cam=args.cam,
            log_level=args.log_level,
        )
        record["recover_invocation_rc"] = rc
        record["recover_invocation_note"] = tail

        if rc != 0:
            record["status"] = "recover_invocation_failed"
            summary_counts["dirs_recover_invocation_failed"] += 1
            dir_records.append(record)
            continue

        report_path = output_dir / "recovery_report.json"
        report_counts = _report_status_counts(report_path)
        if not report_path.exists():
            record["status"] = "recovered_no_report"
            summary_counts["dirs_recovered_no_report"] += 1
            dir_records.append(record)
            continue

        record["status"] = "recovered"
        record["report_recovered_status"] = report_counts.get("recovered", 0)
        record["report_newly_recovered_est"] = max(
            record["report_recovered_status"] - already_fixed, 0
        )
        record["report_recovered"] = record["report_newly_recovered_est"]
        record["report_failed"] = report_counts.get("failed", 0)
        record["report_empty_stub"] = report_counts.get("empty_stub", 0)
        record["report_no_reference"] = report_counts.get("no_reference", 0)
        record["report_pending"] = report_counts.get("pending", 0)

        file_counts["recovered"] += record["report_recovered"]
        file_counts["recovered_status"] += record["report_recovered_status"]
        file_counts["failed"] += record["report_failed"]
        file_counts["empty_stub"] += record["report_empty_stub"]
        file_counts["no_reference"] += record["report_no_reference"]
        file_counts["pending_after_run"] += record["report_pending"]
        summary_counts["dirs_recovered"] += 1
        dir_records.append(record)

    run_finished_at = _now_iso()
    run_id = _run_stamp()

    aggregate = {
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "mode": "run" if args.run else "dry_run",
        "root": str(root),
        "patients": patients,
        "sites": sites,
        "cam_filter": args.cam or None,
        "summary": {
            "video_dirs_discovered": len(discovered),
            "dirs_with_any_corruption": summary_counts.get(
                "dirs_with_any_corruption", 0
            ),
            "dirs_with_target_corruption": summary_counts.get(
                "dirs_with_target_corruption", 0
            ),
            "dirs_clean": summary_counts.get("dirs_clean", 0),
            "dirs_no_mp4": summary_counts.get("dirs_no_mp4", 0),
            "dirs_no_target_corruption": summary_counts.get(
                "dirs_no_target_corruption", 0
            ),
            "dirs_already_recovered": summary_counts.get("dirs_already_recovered", 0),
            "dirs_needing_recovery": summary_counts.get("dirs_needing_recovery", 0),
            "dirs_recovered": summary_counts.get("dirs_recovered", 0),
            "dirs_recovered_no_report": summary_counts.get(
                "dirs_recovered_no_report", 0
            ),
            "dirs_recover_invocation_failed": summary_counts.get(
                "dirs_recover_invocation_failed", 0
            ),
            "dirs_scan_failed": summary_counts.get("dirs_scan_failed", 0),
            "files_total_mp4s": file_counts.get("total_mp4s", 0),
            "files_good_mp4s": file_counts.get("good_mp4s", 0),
            "files_corrupted_mp4s": file_counts.get("corrupted_mp4s", 0),
            "files_targeted_corrupted_mp4s": file_counts.get(
                "targeted_corrupted_mp4s", 0
            ),
            "files_already_fixed_outputs": file_counts.get("already_fixed_outputs", 0),
            "files_pending_outputs_before": file_counts.get(
                "pending_outputs_before", 0
            ),
            "files_recovered": file_counts.get("recovered", 0),
            "files_recovered_status": file_counts.get("recovered_status", 0),
            "files_failed": file_counts.get("failed", 0),
            "files_empty_stub": file_counts.get("empty_stub", 0),
            "files_no_reference": file_counts.get("no_reference", 0),
            "files_pending_after_run": file_counts.get("pending_after_run", 0),
        },
        "directories": dir_records,
    }

    json_path = log_dir / f"recover_mp4_tree_{run_id}.json"
    json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    csv_path = log_dir / f"recover_mp4_tree_{run_id}.csv"
    fieldnames = [
        "patient",
        "date",
        "site",
        "video_dir",
        "output_dir",
        "status",
        "total_mp4s",
        "good_mp4s",
        "corrupted_mp4s",
        "targeted_corrupted_mp4s",
        "already_fixed_outputs",
        "pending_outputs_before",
        "recover_invocation_rc",
        "recover_invocation_note",
        "report_recovered",
        "report_recovered_status",
        "report_newly_recovered_est",
        "report_failed",
        "report_empty_stub",
        "report_no_reference",
        "report_pending",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dir_records)

    log.info("Wrote aggregate JSON: %s", json_path)
    log.info("Wrote aggregate CSV: %s", csv_path)
    log.info("Mode: %s", aggregate["mode"])
    log.info(
        "Directories: total=%d target_corrupted=%d clean=%d no_mp4=%d already_recovered=%d need_recovery=%d recovered=%d failed=%d",
        aggregate["summary"]["video_dirs_discovered"],
        aggregate["summary"]["dirs_with_target_corruption"],
        aggregate["summary"]["dirs_clean"],
        aggregate["summary"]["dirs_no_mp4"],
        aggregate["summary"]["dirs_already_recovered"],
        aggregate["summary"]["dirs_needing_recovery"],
        aggregate["summary"]["dirs_recovered"],
        aggregate["summary"]["dirs_recover_invocation_failed"],
    )
    log.info(
        "Files: total=%d corrupted=%d already_fixed=%d pending_before=%d recovered=%d failed=%d",
        aggregate["summary"]["files_total_mp4s"],
        aggregate["summary"]["files_corrupted_mp4s"],
        aggregate["summary"]["files_already_fixed_outputs"],
        aggregate["summary"]["files_pending_outputs_before"],
        aggregate["summary"]["files_recovered"],
        aggregate["summary"]["files_failed"],
    )


if __name__ == "__main__":
    main()
