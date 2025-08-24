#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dropout_analysis.py  —  minimal dropout checker using pandas

CSV columns (header required):
    serial,start_sample,end_sample

Finds oversized gaps in start_sample spacings (baseline = median by default),
estimates missing samples, loss %, and saves a neatly formatted TXT report:
<csv>-dropout-analysis.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import json
import pandas as pd


class AudioDropoutAnalysis:
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

    def analyze(self, save_report: bool = True) -> Dict[str, Any]:
        df = pd.read_csv(
            self.csv_path,
            dtype={"serial": "int64", "start_sample": "int64", "end_sample": "int64"},
        )
        if self.sort_by_start:
            df = df.sort_values("start_sample", kind="stable").reset_index(drop=True)

        if len(df) < 2:
            report = {
                "has_data_loss": False,
                "num_loss_gaps": 0,
                "total_missing_samples": 0,
                "pct_audio_lost": 0.0,
                "baseline_delta": None,
                "tolerance": self.user_tolerance or 0,
                "expected_delta_from_rates": int(
                    round(self.audio_fs / self.serial_rate)
                ),
                "gaps": [],
                "stats": {"rows": int(len(df))},
            }
            if save_report:
                report["report_path"] = str(
                    self._write_txt_report(report, gaps_df=None)
                )
            return report

        expected = int(round(self.audio_fs / self.serial_rate))  # 44100/30 = 1470
        deltas = df["start_sample"].diff()
        baseline = int(deltas.median()) if self.use_median_baseline else expected
        mad = int((deltas - baseline).abs().median())
        tol = int(
            self.user_tolerance if self.user_tolerance is not None else max(6 * mad, 50)
        )

        # Oversized gaps
        mask = deltas > (baseline + tol)
        gaps_idx = df.index[mask].tolist()  # right side of the gap (i)
        gaps: List[Dict[str, Any]] = []
        for i in gaps_idx:
            prev = df.loc[i - 1]
            curr = df.loc[i]
            delta = int(curr["start_sample"] - prev["start_sample"])
            missing = max(0, delta - baseline)
            gaps.append(
                {
                    "gap_index": int(i),  # between rows i-1 and i after sorting
                    "prev_serial": int(prev["serial"]),
                    "curr_serial": int(curr["serial"]),
                    "prev_start_sample": int(prev["start_sample"]),
                    "curr_start_sample": int(curr["start_sample"]),
                    "delta_samples": int(delta),
                    "estimated_missing_samples": int(missing),
                    "approx_missing_ms": round(1000.0 * missing / self.audio_fs, 3),
                }
            )

        total_missing = sum(g["estimated_missing_samples"] for g in gaps)
        total_span = max(1, int(df.iloc[-1]["end_sample"] - df.iloc[0]["start_sample"]))
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
                "rows": int(len(df)),
                "min_delta": int(deltas.iloc[1:].min()),
                "median_delta": int(deltas.iloc[1:].median()),
                "max_delta": int(deltas.iloc[1:].max()),
                "mad_delta": int(mad),
            },
        }

        gaps_df = pd.DataFrame(gaps)
        if save_report:
            report["report_path"] = str(
                self._write_txt_report(
                    report, gaps_df=gaps_df if not gaps_df.empty else None
                )
            )
        return report

    # -------------------------- report writer --------------------------

    def _write_txt_report(
        self, report: Dict[str, Any], gaps_df: Optional[pd.DataFrame]
    ) -> Path:
        out_path = self.csv_path.with_suffix("")
        out_path = out_path.with_name(out_path.name + "-dropout-analysis.txt")

        lines = []
        lines.append("Dropout Analysis Report")
        lines.append(f"Source CSV         : {self.csv_path.name}")
        lines.append(f"Audio fs (Hz)      : {self.audio_fs}")
        lines.append(f"Serial rate (Hz)   : {self.serial_rate}")
        lines.append(
            f"Expected Δ (rates) : {report['expected_delta_from_rates']} samples"
        )
        lines.append(f"Baseline Δ used    : {report['baseline_delta']}")
        lines.append(f"Tolerance          : ±{report['tolerance']} samples")
        lines.append("")
        lines.append(f"Rows               : {report['stats']['rows']}")
        lines.append(
            "Min/Median/Max Δ   : "
            f"{report['stats']['min_delta']}/{report['stats']['median_delta']}/{report['stats']['max_delta']} samples"
        )
        lines.append(f"MAD(Δ)             : {report['stats']['mad_delta']} samples")
        lines.append("")
        lines.append(f"Has data loss      : {report['has_data_loss']}")
        lines.append(f"Loss gaps          : {report['num_loss_gaps']}")
        lines.append(
            f"Missing samples    : {report['total_missing_samples']:,} "
            f"(~{round(report['total_missing_samples']/self.audio_fs, 3)} s)"
        )
        lines.append(f"% audio lost       : {report['pct_audio_lost']:.6f}%")
        lines.append("")

        if gaps_df is not None:
            # Nicely formatted text table via pandas
            fmt = {
                "gap_index": "{:d}".format,
                "prev_serial": "{:d}".format,
                "curr_serial": "{:d}".format,
                "prev_start_sample": "{:,}".format,
                "curr_start_sample": "{:,}".format,
                "delta_samples": "{:,}".format,
                "estimated_missing_samples": "{:,}".format,
                "approx_missing_ms": "{:.3f}".format,
            }
            table = gaps_df[
                [
                    "gap_index",
                    "prev_serial",
                    "curr_serial",
                    "prev_start_sample",
                    "curr_start_sample",
                    "delta_samples",
                    "estimated_missing_samples",
                    "approx_missing_ms",
                ]
            ].to_string(index=False, formatters=fmt)
            lines.append(
                "GAP DETAILS (between row i-1 and i in start_sample-sorted order):"
            )
            lines.append(table)
        else:
            lines.append("No oversized gaps detected (within tolerance).")

        Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return Path(out_path)


# ------------------------------- CLI -----------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect audio dropouts (pandas-based).")
    p.add_argument("csv", help="CSV path with columns serial,start_sample,end_sample")
    p.add_argument("--audio-fs", type=int, default=44100)
    p.add_argument("--serial-rate", type=float, default=30.0)
    p.add_argument(
        "--no-median",
        action="store_true",
        help="Use expected delta (fs/rate) instead of median baseline",
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


def _main(argv: Optional[list[str]] = None) -> int:
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
