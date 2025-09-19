#!/usr/bin/env python3
"""
json_analysis.py — JSON front end for serial discontinuity diagnostics.

Public API:
    analyze_json_serials(path, key, expect_step=1, top=5, hist_cols=2,
                         out_dir=None) -> tuple[Analysis, pathlib.Path]

CLI:
    json-serial-analysis <path.json> --key SERIALS_KEY --out-dir OUTDIR
                          [--expect-step 1] [--top 5] [--hist-cols 2]

Notes
-----
- The JSON is expected to be a dict at the top level.
- `--key` must point to a top-level key whose value is a list of integers
  (floats that are whole numbers are accepted).
- The report filename is:
    OUTDIR / f"{<json_stem>}.{key}.txt"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from scripts.analysis.serial_analysis import (
    analyze,
    summarize_text,
    Analysis,
)


# -----------------------------
# Helpers
# -----------------------------
def _to_int_list(seq: Iterable[Any]) -> List[int]:
    """
    Convert an iterable of numeric-like values to a list[int].
    Accepts ints and floats that are integral (e.g., 3.0).
    Raises ValueError if any element is non-numeric or if fewer than 2 ints result.
    """
    out: List[int] = []
    for x in seq:
        if isinstance(x, bool):
            raise ValueError("Boolean values are not valid serials")
        if isinstance(x, int):
            out.append(int(x))
        elif isinstance(x, float):
            if x.is_integer():
                out.append(int(x))
            else:
                raise ValueError(f"Non-integer float encountered: {x}")
        else:
            raise ValueError(f"Non-numeric value encountered: {x!r}")
    if len(out) < 2:
        raise ValueError("Need at least 2 integer values for analysis")
    return out


# -----------------------------
# Public API
# -----------------------------
def analyze_json_serials(
    path: str,
    key: str,
    *,
    expect_step: int = 1,
    top: int = 5,
    hist_cols: int = 2,
    out_dir: Optional[str] = None,
) -> Tuple[Analysis, Path]:
    """Analyze a JSON list of integers at a given top-level key and write a text report.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    key : str
        Top-level key in the JSON dict whose value is a list of integers.
    expect_step : int, default 1
        Expected increment per step (E). Steps are diff = ids[i+1] - ids[i].
    top : int, default 5
        How many top forward/drops to list in the text report.
    hist_cols : int, default 2
        Histogram entries per row in the text report (1 = one per line).
    out_dir : str or None, default None
        Output directory for the text report. If None, uses the JSON's directory.

    Returns
    -------
    (Analysis, Path)
        The structured analysis and the absolute path to the written text report.

    Raises
    ------
    FileNotFoundError
        If the JSON path does not exist.
    ValueError
        If the JSON is not a dict, the key is missing, the value is not a list,
        or the list does not contain >=2 integer-like values.
    OSError
        If writing out files fails due to I/O errors.
    """
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    # Load JSON
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object/dict")

    if key not in data:
        raise ValueError(f"Key not found in JSON: {key}")

    seq = data[key]
    if not isinstance(seq, list):
        raise ValueError(f"JSON[{key!r}] must be a list, got {type(seq).__name__}")

    ids = _to_int_list(seq)

    # Analyze
    result = analyze(ids, expect_step=expect_step, top_k=top)

    # Render text report
    report_text = summarize_text(
        result, include_tops=True, hist_cols=max(1, int(hist_cols))
    )

    # Output path
    out_dir_path = Path(out_dir) if out_dir is not None else json_path.parent
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_text_path = (out_dir_path / f"{json_path.stem}.{key}.txt").resolve()

    # Write text output
    try:
        out_text_path.write_text(
            "Source → JSON[{key}]\n{body}\n".format(key=key, body=report_text),
            encoding="utf-8",
        )
    except OSError as e:
        raise OSError(f"Failed to write text report: {out_text_path}") from e

    return result, out_text_path


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="json-serial-analysis",
        description=(
            "Analyze discontinuities (ok, duplicate, forward, drop) "
            "in a JSON list of integers at a given top-level key."
        ),
    )
    p.add_argument("path", help="Path to the JSON file")
    p.add_argument(
        "--key",
        required=True,
        help="Top-level JSON key containing the integer list (e.g., 'serials')",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write the text report into",
    )
    p.add_argument(
        "--expect-step",
        type=int,
        default=1,
        help="Expected increment per step (default: 1)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top forward/drops to list (default: 5)",
    )
    p.add_argument(
        "--hist-cols",
        type=int,
        default=2,
        help="Histogram entries per row (1 = one per line, default: 2)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    try:
        res, outp = analyze_json_serials(
            path=args.path,
            key=args.key,
            expect_step=args.expect_step,
            top=args.top,
            hist_cols=args.hist_cols,
            out_dir=args.out_dir,
        )
        logging.info(
            "Done. Values=%d Steps=%d ok=%d (%.2f%%) MissingIDs=%d → %s",
            res.n_values,
            res.total_steps,
            res.ok_steps,
            100.0 * res.ok_ratio,
            res.total_missing_ids,
            str(outp),
        )
        return 0
    except (FileNotFoundError, ValueError, OSError) as e:
        logging.error(str(e))
        return 2
    except Exception as e:
        logging.exception("Unexpected error: %s", e)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
