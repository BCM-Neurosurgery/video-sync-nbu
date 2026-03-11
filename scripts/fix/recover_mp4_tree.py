#!/usr/bin/env python3
"""
recover_mp4_tree.py — Recover corrupted MP4s across the TRBD datalake tree.

Scans:
  <root>/<patient>/NBU/<YYYY-MM-DD>/video/<site>

For each discovered video directory:
  - Detect corrupted MP4s (missing moov atom)
  - Check whether corresponding *_fixed.mp4 already exists in sibling
    <site>_recovered
  - Recover only missing outputs via scripts.fix.recover_mp4 public API

Writes one aggregated run log (JSON + CSV) with per-directory and total stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.fix.recover_mp4 import plan_recovery, run_recovery
from scripts.log.logutils import configure_standalone_logging

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_list(values: list[str]) -> list[str]:
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


def _discover_patients(root: Path) -> list[str]:
    """Auto-discover patient directories under a root.

    A subdirectory is considered a patient if it contains an NBU/ folder.
    """
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "NBU").is_dir())


def _discover_video_dirs(
    roots: list[Path],
    patients: list[str],
    sites: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for root in roots:
        if not root.is_dir():
            log.warning("Root directory does not exist: %s", root)
            continue

        # Auto-discover patients if none specified, otherwise filter
        available = _discover_patients(root)
        if patients:
            use_patients = [p for p in patients if p in available]
        else:
            use_patients = available

        for patient in use_patients:
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


def _scan_failed_record(
    *,
    patient: str,
    date: str,
    site: str,
    video_dir: Path,
    output_dir: Path,
    error: str,
) -> dict[str, Any]:
    return {
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
        "recover_api_rc": None,
        "recover_api_note": error,
        "status": "scan_failed",
        "report_recovered": 0,
        "report_failed": 0,
        "report_empty_stub": 0,
        "report_no_reference": 0,
        "report_pending": 0,
        "report_recovered_status": 0,
        "report_newly_recovered": 0,
        "report_recovered_frames": 0,
        "report_recovered_duration_s": 0,
    }


def _sum_file_records(records: list[dict]) -> tuple[int, float]:
    """Sum recovered_frames and recovered_duration_s across file records."""
    frames = sum(int(r.get("recovered_frames", 0)) for r in records)
    dur = sum(float(r.get("recovered_duration_s", 0)) for r in records)
    return frames, round(dur, 2)


def _make_record(
    *,
    patient: str,
    date: str,
    site: str,
    video_dir: Path,
    output_dir: Path,
    total_mp4s: int,
    good_mp4s: int,
    corrupted_mp4s: int,
    targeted_corrupted_mp4s: int,
    already_fixed_outputs: int,
    pending_outputs_before: int,
) -> dict[str, Any]:
    return {
        "patient": patient,
        "date": date,
        "site": site,
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "total_mp4s": total_mp4s,
        "good_mp4s": good_mp4s,
        "corrupted_mp4s": corrupted_mp4s,
        "targeted_corrupted_mp4s": targeted_corrupted_mp4s,
        "already_fixed_outputs": already_fixed_outputs,
        "pending_outputs_before": pending_outputs_before,
        "recover_api_rc": None,
        "recover_api_note": "",
        "status": "",
        "report_recovered": 0,
        "report_failed": 0,
        "report_empty_stub": 0,
        "report_no_reference": 0,
        "report_pending": 0,
        "report_recovered_status": 0,
        "report_newly_recovered": 0,
        "report_recovered_frames": 0,
        "report_recovered_duration_s": 0,
    }


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            Path("~/mnt/datalake/TRBD-53761").expanduser(),
            Path("~/mnt/datalake/AA-56119").expanduser(),
        ],
        help="Root datalake paths (default: ~/mnt/datalake/TRBD-53761 ~/mnt/datalake/AA-56119)",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=[],
        help="Patient IDs to process; if omitted, auto-discovers all patients "
        "under each root. Supports comma-separated and/or space-separated values",
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
        "--ref-dir",
        type=Path,
        default=None,
        help="External directory with good MP4s to use as references "
        "(for when a video directory has no uncorrupted files)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute recovery; default is scan-only dry run",
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

    roots = [r.expanduser().resolve() for r in args.roots]
    patients = _normalize_list(args.patients)
    sites = _normalize_list(args.sites)
    log_dir = (
        args.log_dir.expanduser().resolve()
        if args.log_dir
        else roots[0] / "recovery_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    run_started_at = _now_iso()
    discovered = _discover_video_dirs(roots, patients, sites)
    log.info("Discovered %d video directories.", len(discovered))

    dir_records: list[dict[str, Any]] = []
    summary_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()

    total_dirs = len(discovered)
    for idx, entry in enumerate(discovered, start=1):
        patient = entry["patient"]
        date = entry["date"]
        site = entry["site"]
        video_dir: Path = entry["video_dir"]
        output_dir: Path = entry["output_dir"]

        log.info(
            "[%d/%d] Scanning %s | patient=%s date=%s site=%s",
            idx,
            total_dirs,
            video_dir,
            patient,
            date,
            site,
        )

        ref_dir = args.ref_dir.expanduser().resolve() if args.ref_dir else None

        try:
            plan = plan_recovery(
                input_dir=video_dir,
                output_dir=output_dir,
                cam=args.cam,
                ref_dir=ref_dir,
            )
        except Exception as exc:
            log.error(
                "[%d/%d] Scan failed for %s: %s",
                idx,
                total_dirs,
                video_dir,
                exc,
            )
            summary_counts["dirs_scan_failed"] += 1
            dir_records.append(
                _scan_failed_record(
                    patient=patient,
                    date=date,
                    site=site,
                    video_dir=video_dir,
                    output_dir=output_dir,
                    error=str(exc),
                )
            )
            continue

        scan = plan["scan"]
        plan_counts = plan["counts"]
        plan_status = str(plan["status"])

        total_mp4s = int(scan["total_mp4s"])
        good_count = int(scan["good_files"])
        corrupted_count = int(scan["corrupted_files"])
        targeted_corrupted_count = int(plan["targeted_corrupted_files"])
        already_fixed = int(plan_counts["already_done"])
        pending_outputs = int(plan_counts["pending"])
        no_reference_count = int(plan_counts["no_reference"])
        empty_stub_count = int(plan_counts["empty_stub"])

        log.info(
            "[%d/%d] Scan done: total=%d good=%d corrupted=%d",
            idx,
            total_dirs,
            total_mp4s,
            good_count,
            corrupted_count,
        )

        record = _make_record(
            patient=patient,
            date=date,
            site=site,
            video_dir=video_dir,
            output_dir=output_dir,
            total_mp4s=total_mp4s,
            good_mp4s=good_count,
            corrupted_mp4s=corrupted_count,
            targeted_corrupted_mp4s=targeted_corrupted_count,
            already_fixed_outputs=already_fixed,
            pending_outputs_before=pending_outputs,
        )

        # Estimate durations using reference file duration as proxy
        ref_dur = float(plan.get("ref_duration_s", 0))
        est_good_dur = good_count * ref_dur
        est_corrupted_dur = (corrupted_count - empty_stub_count) * ref_dur

        file_counts["total_mp4s"] += total_mp4s
        file_counts["good_mp4s"] += good_count
        file_counts["corrupted_mp4s"] += corrupted_count
        file_counts["targeted_corrupted_mp4s"] += targeted_corrupted_count
        file_counts["already_fixed_outputs"] += already_fixed
        file_counts["pending_outputs_before"] += pending_outputs
        file_counts["est_good_duration_s"] += est_good_dur
        file_counts["est_corrupted_duration_s"] += est_corrupted_dur
        file_counts["no_reference_detected"] += no_reference_count
        file_counts["empty_stub_detected"] += empty_stub_count

        if plan_status == "no_mp4":
            record["status"] = "no_mp4"
            log.info("[%d/%d] Status: no_mp4", idx, total_dirs)
            summary_counts["dirs_no_mp4"] += 1
            dir_records.append(record)
            continue

        if plan_status == "no_corrupted":
            record["status"] = "clean"
            log.info("[%d/%d] Status: clean (no corrupted files)", idx, total_dirs)
            summary_counts["dirs_clean"] += 1
            dir_records.append(record)
            continue

        summary_counts["dirs_with_any_corruption"] += 1

        if plan_status == "no_target_corruption":
            record["status"] = "no_target_corruption"
            log.info(
                "[%d/%d] Status: no_target_corruption%s",
                idx,
                total_dirs,
                f" (cam={args.cam})" if args.cam else "",
            )
            summary_counts["dirs_no_target_corruption"] += 1
            dir_records.append(record)
            continue

        summary_counts["dirs_with_target_corruption"] += 1
        log.info(
            "[%d/%d] Targeted corrupted=%d | already_fixed=%d | pending=%d",
            idx,
            total_dirs,
            targeted_corrupted_count,
            already_fixed,
            pending_outputs,
        )

        if plan_status == "no_reference_files":
            record["status"] = "no_reference_files"
            record["recover_api_rc"] = 1
            record["recover_api_note"] = str(plan.get("error", "no_reference_files"))
            log.warning(
                "[%d/%d] Status: no_reference_files (%s)",
                idx,
                total_dirs,
                record["recover_api_note"],
            )
            summary_counts["dirs_no_reference_files"] += 1
            dir_records.append(record)
            continue

        if pending_outputs == 0:
            record["status"] = "already_recovered"
            log.info("[%d/%d] Status: already_recovered", idx, total_dirs)
            summary_counts["dirs_already_recovered"] += 1
            dir_records.append(record)
            continue

        if not args.run:
            record["status"] = "needs_recovery"
            log.info("[%d/%d] Status: needs_recovery (dry-run)", idx, total_dirs)
            summary_counts["dirs_needing_recovery"] += 1
            dir_records.append(record)
            continue

        try:
            recover_result = run_recovery(
                input_dir=video_dir,
                output_dir=output_dir,
                run=True,
                cam=args.cam,
                write_report=True,
                plan=plan,
            )
        except Exception as exc:
            record["recover_api_rc"] = 1
            record["recover_api_note"] = str(exc)
            record["status"] = "recover_api_failed"
            log.error(
                "[%d/%d] Status: recover_api_failed (%s)",
                idx,
                total_dirs,
                exc,
            )
            summary_counts["dirs_recover_api_failed"] += 1
            dir_records.append(record)
            continue

        if recover_result.get("status") == "no_reference_files":
            record["recover_api_rc"] = 1
            record["recover_api_note"] = str(
                recover_result.get("error", "no_reference_files")
            )
            record["status"] = "recover_api_failed"
            log.error(
                "[%d/%d] Status: recover_api_failed (%s)",
                idx,
                total_dirs,
                record["recover_api_note"],
            )
            summary_counts["dirs_recover_api_failed"] += 1
            dir_records.append(record)
            continue

        record["recover_api_rc"] = 0
        record["recover_api_note"] = str(recover_result.get("status", "ok"))

        if not recover_result.get("report_written", False):
            record["status"] = "recovered_no_report"
            log.warning("[%d/%d] Status: recovered_no_report", idx, total_dirs)
            summary_counts["dirs_recovered_no_report"] += 1
            dir_records.append(record)
            continue

        counts = recover_result.get("counts", {})
        recovered_new = int(counts.get("recovered", 0))
        recovered_already = int(counts.get("already_done", 0))

        record["status"] = "recovered"
        record["report_recovered"] = recovered_new
        record["report_recovered_status"] = recovered_new + recovered_already
        record["report_newly_recovered"] = recovered_new
        record["report_failed"] = int(counts.get("failed", 0))
        record["report_empty_stub"] = int(counts.get("empty_stub", 0))
        record["report_no_reference"] = int(counts.get("no_reference", 0))
        record["report_pending"] = int(counts.get("pending", 0))

        file_records = recover_result.get("records", [])
        dir_frames, dir_dur = _sum_file_records(file_records)
        record["report_recovered_frames"] = dir_frames
        record["report_recovered_duration_s"] = dir_dur

        file_counts["recovered"] += record["report_recovered"]
        file_counts["recovered_status"] += record["report_recovered_status"]
        file_counts["failed"] += record["report_failed"]
        file_counts["empty_stub"] += record["report_empty_stub"]
        file_counts["no_reference"] += record["report_no_reference"]
        file_counts["pending_after_run"] += record["report_pending"]
        file_counts["recovered_frames"] += dir_frames
        file_counts["recovered_duration_s"] += dir_dur
        summary_counts["dirs_recovered"] += 1
        log.info(
            "[%d/%d] Status: recovered | new=%d failed=%d empty=%d no_ref=%d",
            idx,
            total_dirs,
            record["report_recovered"],
            record["report_failed"],
            record["report_empty_stub"],
            record["report_no_reference"],
        )
        dir_records.append(record)

    run_finished_at = _now_iso()
    run_id = _run_stamp()

    aggregate = {
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "mode": "run" if args.run else "dry_run",
        "roots": [str(r) for r in roots],
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
            "dirs_no_reference_files": summary_counts.get("dirs_no_reference_files", 0),
            "dirs_already_recovered": summary_counts.get("dirs_already_recovered", 0),
            "dirs_needing_recovery": summary_counts.get("dirs_needing_recovery", 0),
            "dirs_recovered": summary_counts.get("dirs_recovered", 0),
            "dirs_recovered_no_report": summary_counts.get(
                "dirs_recovered_no_report", 0
            ),
            "dirs_recover_api_failed": summary_counts.get("dirs_recover_api_failed", 0),
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
            "files_no_reference_detected": file_counts.get("no_reference_detected", 0),
            "files_empty_stub_detected": file_counts.get("empty_stub_detected", 0),
            "files_recovered": file_counts.get("recovered", 0),
            "files_recovered_status": file_counts.get("recovered_status", 0),
            "files_failed": file_counts.get("failed", 0),
            "files_empty_stub": file_counts.get("empty_stub", 0),
            "files_no_reference": file_counts.get("no_reference", 0),
            "files_pending_after_run": file_counts.get("pending_after_run", 0),
            "files_recovered_frames": file_counts.get("recovered_frames", 0),
            "files_recovered_duration_s": round(
                file_counts.get("recovered_duration_s", 0), 2
            ),
            "files_recovered_duration_hours": round(
                file_counts.get("recovered_duration_s", 0) / 3600, 2
            ),
            "est_total_duration_hours": round(
                (
                    file_counts.get("est_good_duration_s", 0)
                    + file_counts.get("est_corrupted_duration_s", 0)
                )
                / 3600,
                2,
            ),
            "est_corrupted_duration_hours": round(
                file_counts.get("est_corrupted_duration_s", 0) / 3600, 2
            ),
            "est_corrupted_pct": round(
                file_counts.get("est_corrupted_duration_s", 0)
                / max(
                    file_counts.get("est_good_duration_s", 0)
                    + file_counts.get("est_corrupted_duration_s", 0),
                    1,
                )
                * 100,
                1,
            ),
            "recovered_pct_of_corrupted": round(
                file_counts.get("recovered_duration_s", 0)
                / max(file_counts.get("est_corrupted_duration_s", 0), 1)
                * 100,
                1,
            ),
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
        "recover_api_rc",
        "recover_api_note",
        "report_recovered",
        "report_recovered_status",
        "report_newly_recovered",
        "report_failed",
        "report_empty_stub",
        "report_no_reference",
        "report_pending",
        "report_recovered_frames",
        "report_recovered_duration_s",
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
        aggregate["summary"]["dirs_recover_api_failed"],
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
    s = aggregate["summary"]
    log.info(
        "Video: est_total=%.1fh est_corrupted=%.1fh (%.1f%%) recovered=%.1fh (%.1f%% of corrupted)",
        s["est_total_duration_hours"],
        s["est_corrupted_duration_hours"],
        s["est_corrupted_pct"],
        s["files_recovered_duration_hours"],
        s["recovered_pct_of_corrupted"],
    )


if __name__ == "__main__":
    main()
