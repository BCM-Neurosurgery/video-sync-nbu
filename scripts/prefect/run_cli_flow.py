from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import yaml

from scripts.prefect.flows import TimeSyncRunConfig, time_sync_flow


def load_runs_from_yaml(config_path: Path) -> List[TimeSyncRunConfig]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}

    runs_payload = payload.get("runs")
    if not runs_payload:
        raise ValueError(
            f"No runs defined in {config_path}. Add at least one entry under 'runs:'."
        )

    return [TimeSyncRunConfig.from_mapping(item) for item in runs_payload]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute cli_emu_time runs via Prefect using a YAML manifest."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("prefect/time_sync_runs.yml"),
        help="Path to a YAML file describing the runs to execute.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        runs = load_runs_from_yaml(args.config)
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"[prefect-run] {exc}", file=sys.stderr)
        return 1

    results: List[str] = []
    for idx, config in enumerate(runs, start=1):
        out_dir = time_sync_flow(config)
        results.append(out_dir)
        print(f"[prefect-run] Run {idx} outputs -> {out_dir}")
    if not results:
        print("[prefect-run] No runs executed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
