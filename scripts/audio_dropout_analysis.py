#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dropout_analysis.py

Minimal dropout checker for an audio serial-index CSV.

CSV columns (header required):
    serial,start_sample,end_sample

What it does
------------
- Loads rows, sorts by start_sample.
- Computes inter-block deltas = start[i] - start[i-1].
- Uses a robust baseline = median(deltas) by default (or expected 1470 if you prefer).
- Flags "audio loss" where delta > baseline + tolerance.
- Sums missing samples and reports:
    * whether loss exists
    * how many gaps
    * where (gap index, serials, sample positions)
    * percentage of audio lost
- Writes a human-readable TXT report next to the CSV:
    <name>-dropout-analysis.txt

Defaults (per your setup)
-------------------------
- audio_fs = 44100 Hz
- serial_rate = 30 Hz  → expected delta ≈ 1470 samples

Usage (library)
---------------
from dropout_analysis import AudioDropoutAnalysis
report = AudioDropoutAnalysis("index.csv").analyze()  # saves TXT automatically

CLI
---
$ python dropout_analysis.py /path/to/index.csv
$ python dropout_analysis.py /path/to/index.csv --no-median --tolerance 200
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import csv
import statistics
import argparse
import json


@dataclass
class Row:
    serial: int
    start: int
    end: int
    file_order: int  # 0-based


class AudioDropoutAnalysis:
    """
    Simple audio dropout analysis.

    Parameters
    ----------
    csv_path : str | Path
        CSV with columns: serial,start_sample,end_sample
    audio_fs : int, default 44100
        Audio sampling rate.
    serial_rate : float, default 30.0
        Serial transmission rate (blocks per second).
    use_median_baseline : bool, default True
        If True, baseline = median observed delta; else use expected (≈ audio_fs/serial_rate).
    tolerance : Optional[int], default None
        Absolute tolerance in samples. If None, set to max(6*MAD, 50),
        where MAD = median(|delta - baseline|).
    sort_by_start : bool, default True
        Sort rows by start_sample before analysis.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        audio_fs: int = 44100,
        serial_rate: float = 30.0,
        use_median_baseline: bool = True,
        tolerance: Optional[int] = None,
        sort_by_start: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.audio_fs = int(audio_fs)
        self.serial_rate = float(serial_rate)
        self.use_median_baseline = bool(use_median_baseline)
        self.user_tolerance = tolerance
        self.sort_by_start = bool(sort_by_start)

    # ---------------------------- Public API -----------------------------

    def analyze(self, save_report: bool = True) -> Dict[str, Any]:
        rows = self._read_rows()
        if self.sort_by_start:
            rows.sort(key=lambda r: r.start)

        if len(rows) < 2:
            report = {
                "has_data_loss": False,
                "num_loss_gaps": 0,
                "total_missing_samples": 0,
                "pct_audio_lost": 0.0,
                "baseline_delta": None,
                "tolerance": (
                    self.user_tolerance if self.user_tolerance is not None else 0
                ),
                "expected_delta_from_rates": round(self.audio_fs / self.serial_rate),
                "gaps": [],
                "stats": {"rows": len(rows)},
            }
            if save_report:
                path = self._write_txt_report(report)
                report["report_path"] = str(path)
            return report

        # Inter-block deltas
        deltas = [rows[i].start - rows[i - 1].start for i in range(1, len(rows))]

        expected = round(self.audio_fs / self.serial_rate)
        baseline = statistics.median(deltas) if self.use_median_baseline else expected

        mad = statistics.median([abs(d - baseline) for d in deltas]) if deltas else 0
        tol = (
            self.user_tolerance
            if self.user_tolerance is not None
            else max(int(6 * mad), 50)
        )

        # Detect loss gaps: oversized delta
        gaps: List[Dict[str, Any]] = []
        total_missing = 0
        for i, delta in enumerate(
            deltas, start=1
        ):  # i is the right index of the gap (between i-1 and i)
            if delta > (baseline + tol):
                missing = delta - baseline
                total_missing += missing
                prev, curr = rows[i - 1], rows[i]
                gaps.append(
                    {
                        "gap_index": i,  # between rows i-1 and i (1-based gap index)
                        "prev_serial": prev.serial,
                        "curr_serial": curr.serial,
                        "prev_start_sample": prev.start,
                        "curr_start_sample": curr.start,
                        "delta_samples": int(delta),
                        "estimated_missing_samples": int(missing),
                        "approx_missing_ms": round(1000.0 * missing / self.audio_fs, 3),
                    }
                )

        # Denominator for % lost: duration covered by rows (inclusive span)
        total_span = max(1, rows[-1].end - rows[0].start)
        pct_lost = 100.0 * total_missing / total_span

        report = {
            "has_data_loss": len(gaps) > 0,
            "num_loss_gaps": len(gaps),
            "total_missing_samples": int(total_missing),
            "pct_audio_lost": pct_lost,
            "baseline_delta": int(baseline),
            "tolerance": int(tol),
            "expected_delta_from_rates": int(expected),
            "gaps": gaps,
            "stats": {
                "rows": len(rows),
                "min_delta": int(min(deltas)),
                "max_delta": int(max(deltas)),
                "median_delta": int(statistics.median(deltas)),
                "mad_delta": int(mad),
            },
        }

        if save_report:
            path = self._write_txt_report(report)
            report["report_path"] = str(path)
        return report

    # -------------------------- Helpers ---------------------------------

    def _read_rows(self) -> List[Row]:
        out: List[Row] = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            need = {"serial", "start_sample", "end_sample"}
            if not rdr.fieldnames or not need.issubset(set(rdr.fieldnames)):
                raise ValueError(f"CSV must have columns: {sorted(need)}")
            for idx, r in enumerate(rdr):
                try:
                    out.append(
                        Row(
                            serial=int(r["serial"]),
                            start=int(r["start_sample"]),
                            end=int(r["end_sample"]),
                            file_order=idx,
                        )
                    )
                except Exception:
                    continue
        if not out:
            raise ValueError("No valid rows parsed.")
        return out

    def _write_txt_report(self, report: Dict[str, Any]) -> Path:
        out_path = self.csv_path.with_suffix("")  # drop .csv
        out_path = out_path.with_name(out_path.name + "-dropout-analysis.txt")
        lines: List[str] = []

        lines.append(f"Dropout Analysis Report")
        lines.append(f"Source CSV         : {self.csv_path.name}")
        lines.append(f"Audio fs (Hz)      : {self.audio_fs}")
        lines.append(f"Serial rate (Hz)   : {self.serial_rate}")
        lines.append(
            f"Expected Δ (rates) : {report['expected_delta_from_rates']} samples"
        )
        lines.append(f"Baseline Δ used    : {report['baseline_delta']} samples")
        lines.append(f"Tolerance          : ±{report['tolerance']} samples")
        lines.append("")
        lines.append(f"Rows               : {report['stats']['rows']}")
        lines.append(
            f"Min/Median/Max Δ   : {report['stats']['min_delta']}/"
            f"{report['stats']['median_delta']}/{report['stats']['max_delta']} samples"
        )
        lines.append(f"MAD(Δ)             : {report['stats']['mad_delta']} samples")
        lines.append("")
        lines.append(f"Has data loss      : {report['has_data_loss']}")
        lines.append(f"Loss gaps          : {report['num_loss_gaps']}")
        lines.append(
            f"Missing samples    : {report['total_missing_samples']} "
            f"(~{round(report['total_missing_samples']/self.audio_fs, 3)} s)"
        )
        lines.append(f"% audio lost       : {report['pct_audio_lost']:.6f}%")
        lines.append("")

        if report["gaps"]:
            lines.append(
                "GAP DETAILS (between row i-1 and i in start_sample-sorted order):"
            )
            lines.append(
                " idx | prev_serial -> curr_serial | prev_start -> curr_start | Δsamples | ~missing(samp) | ~missing(ms)"
            )
            lines.append(
                "-----+----------------------------+---------------------------+---------+---------------+--------------"
            )
            for g in report["gaps"]:
                lines.append(
                    f"{g['gap_index']:>4} | "
                    f"{g['prev_serial']:>11} -> {g['curr_serial']:<11} | "
                    f"{g['prev_start_sample']:>10} -> {g['curr_start_sample']:<10} | "
                    f"{g['delta_samples']:>7} | "
                    f"{g['estimated_missing_samples']:>13} | "
                    f"{g['approx_missing_ms']:>12}"
                )
        else:
            lines.append("No oversized gaps detected (within tolerance).")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path


# ------------------------------ CLI ------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect audio dropouts from serial block start spacing."
    )
    p.add_argument("csv", help="CSV path with columns serial,start_sample,end_sample")
    p.add_argument(
        "--audio-fs",
        type=int,
        default=44100,
        help="Audio sampling rate (default: 44100)",
    )
    p.add_argument(
        "--serial-rate", type=float, default=30.0, help="Serial rate Hz (default: 30)"
    )
    p.add_argument(
        "--no-median",
        action="store_true",
        help="Use expected delta instead of median baseline",
    )
    p.add_argument(
        "--tolerance",
        type=int,
        default=None,
        help="Absolute tolerance in samples (default: 6*MAD, min 50)",
    )
    p.add_argument(
        "--no-sort",
        action="store_true",
        help="Keep file order (default sorts by start_sample)",
    )
    p.add_argument(
        "--json", action="store_true", help="Also print JSON report to stdout"
    )
    return p


def _main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    analyzer = AudioDropoutAnalysis(
        args.csv,
        audio_fs=args.audio_fs,
        serial_rate=args.serial_rate,
        use_median_baseline=not args.no_median,
        tolerance=args.tolerance,
        sort_by_start=not args.no_sort,
    )
    report = analyzer.analyze(save_report=True)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Report written to: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
