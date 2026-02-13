from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts.webui.db import get_conn, tx
from scripts.webui.models import utc_now_iso
from scripts.webui.runner import Runner, RunnerConfig, build_cli_cmd, tail_log_sse
from scripts.validate.validate_audio_dir import validate_audio_dir
from scripts.validate.validate_audio_dir import validate_audio_dir_progress
from scripts.validate.validate_out_dir import discover_synced_pairs, validate_out_dir
from scripts.validate.validate_video_dir import (
    discover_from_video_dir,
    validate_video_dir,
    validate_video_dir_progress,
)
from scripts.time.find_audio_abs_time import (
    OutputLayout,
    compute_audio_start_records,
    _record_to_payload,
)
from scripts.index.common import DEFAULT_TZ


ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
try:
    templates.env.globals["static_v"] = int(
        (ROOT / "static" / "app.css").stat().st_mtime
    )
except Exception:
    templates.env.globals["static_v"] = 1


def _default_args() -> Dict[str, Any]:
    return {
        "audio_dir": "",
        "video_dir": "",
        "out_dir": "",
        "site": "nbu_lounge",
        "segments": [],
        "cameras": [],
        "target_pairs": [],
        "manual_target_pairs": [],
        "range_config": {},
        "selection_mode": "segments",
        "log_level": "INFO",
        "skip_decode": False,
        "overwrite_clips": False,
        "split": False,
        "split_overwrite": False,
        "split_clean": False,
        "split_chunk_seconds": 3600,
    }


def _normalize_target_pairs(raw: Optional[List[str]]) -> List[str]:
    pairs: List[str] = []
    seen: set[str] = set()
    for item in raw or []:
        s = str(item).strip()
        if not s:
            continue
        if "::" in s:
            seg, cam = s.split("::", 1)
        elif ":" in s:
            seg, cam = s.split(":", 1)
        else:
            continue
        seg = seg.strip()
        cam = cam.strip()
        if not seg or not cam:
            continue
        key = f"{seg}::{cam}"
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _sort_segment_ids(segment_ids: List[str]) -> List[str]:
    def key(seg: str) -> tuple:
        match = re.search(r"(\d{8})_(\d{6})", seg)
        if match:
            try:
                return (0, int(match.group(1)), int(match.group(2)), seg)
            except ValueError:
                pass
        return (1, seg)

    return sorted(segment_ids, key=key)


def _segment_range_title(args: Dict[str, Any]) -> str:
    segs: List[str] = []
    raw_pairs = args.get("target_pairs")
    if isinstance(raw_pairs, list) and raw_pairs:
        seen: set[str] = set()
        for item in raw_pairs:
            s = str(item).strip()
            if not s:
                continue
            if "::" in s:
                seg = s.split("::", 1)[0]
            elif ":" in s:
                seg = s.split(":", 1)[0]
            else:
                continue
            seg = seg.strip()
            if seg and seg not in seen:
                seen.add(seg)
                segs.append(seg)
    else:
        raw = args.get("segments")
        if isinstance(raw, list):
            segs = [str(s).strip() for s in raw if str(s).strip()]

    segs = _sort_segment_ids(segs)
    if not segs:
        return "All segments"
    if len(segs) == 1:
        return segs[0]
    first = segs[0]
    last = segs[-1]
    prefix = first.rsplit("_", 1)[0] if "_" in first else ""
    if prefix and last.startswith(prefix + "_"):
        last = last[len(prefix) + 1 :]
    return f"{first} → {last}"


RANGE_MODE_VALUES = {"manual", "time", "sample"}
TIME_DISPLAY_FMT = "%Y-%m-%d %H:%M:%S"
WEBUI_DEFAULT_TIME_ZONE = "America/Chicago"


def _normalize_range_mode(raw: object, *, default: str = "manual") -> str:
    s = str(raw or "").strip().lower()
    return s if s in RANGE_MODE_VALUES else default


def _parse_int_or_none(raw: object) -> Optional[int]:
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _normalize_time_text(raw: object) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    s = s.replace("T", " ")
    try:
        return datetime.fromisoformat(s).strftime(TIME_DISPLAY_FMT)
    except Exception:
        pass
    for fmt in (TIME_DISPLAY_FMT, "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).strftime(TIME_DISPLAY_FMT)
        except Exception:
            continue
    return ""


def _parse_time_text(raw: object) -> Optional[datetime]:
    normalized = _normalize_time_text(raw)
    if not normalized:
        return None
    try:
        return datetime.strptime(normalized, TIME_DISPLAY_FMT)
    except Exception:
        return None


def _extract_available_pair_map(video_result: Mapping[str, Any]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    available_pairs = video_result.get("available_pairs")
    if not isinstance(available_pairs, list):
        return out
    for item in available_pairs:
        if isinstance(item, dict):
            seg = item.get("segment")
            cam = item.get("camera")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            seg = item[0]
            cam = item[1]
        else:
            continue
        if seg and cam:
            out[f"{seg}::{cam}"] = True
    return out


def _extract_missing_json_map(video_result: Mapping[str, Any]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    missing_json_segments = video_result.get("missing_json_segments")
    if not isinstance(missing_json_segments, list):
        return out
    for seg in missing_json_segments:
        if seg:
            out[str(seg)] = True
    return out


def _extract_synced_pair_map(out_result: Mapping[str, Any]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    synced_pairs = out_result.get("synced_pairs")
    if not isinstance(synced_pairs, list):
        return out
    for item in synced_pairs:
        if isinstance(item, dict):
            seg = item.get("segment")
            cam = item.get("camera")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            seg = item[0]
            cam = item[1]
        else:
            continue
        if seg and cam:
            out[f"{seg}::{cam}"] = True
    return out


def _load_serial_sample_map(serial_csv: Path) -> Dict[int, Tuple[int, int]]:
    mapping: Dict[int, Tuple[int, int]] = {}
    if not serial_csv.exists() or not serial_csv.is_file():
        return mapping
    try:
        with serial_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    serial = int(str(row.get("serial", "")).strip())
                    start = int(str(row.get("start_sample", "")).strip())
                    end = int(str(row.get("end_sample", "")).strip())
                except Exception:
                    continue
                mapping[serial] = (start, end)
    except Exception:
        return {}
    return mapping


def _build_range_catalog(
    *,
    video_dir: Path,
    out_dir: Path,
    available_pair_map: Mapping[str, bool],
) -> Dict[str, Any]:
    pairs: Dict[str, Dict[str, Any]] = {}
    pairs_by_camera: Dict[str, List[str]] = {}
    camera_time_bounds: Dict[str, Dict[str, str]] = {}
    camera_sample_bounds: Dict[str, Dict[str, int]] = {}

    serial_csv = out_dir / "audio_decoded" / "raw-gapfilled-filtered.csv"
    serial_sample_map = _load_serial_sample_map(serial_csv)
    sample_ready = bool(serial_sample_map)

    segment_ids = sorted({k.split("::", 1)[0] for k in available_pair_map.keys()})
    for seg_id in segment_ids:
        json_path = video_dir / f"{seg_id}.json"
        if not json_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        real_times = payload.get("real_times") or []
        time_start = _normalize_time_text(real_times[0]) if real_times else ""
        time_end = _normalize_time_text(real_times[-1]) if real_times else ""

        serials = [str(s).strip() for s in (payload.get("serials") or [])]
        chunk_serial_data = payload.get("chunk_serial_data") or []
        if not isinstance(chunk_serial_data, list):
            chunk_serial_data = []

        for cam_idx, cam in enumerate(serials):
            key = f"{seg_id}::{cam}"
            if key not in available_pair_map:
                continue

            pair_info: Dict[str, Any] = {
                "time_start": time_start,
                "time_end": time_end,
                "sample_start": None,
                "sample_end": None,
            }

            if sample_ready:
                sample_min: Optional[int] = None
                sample_max: Optional[int] = None
                for row in chunk_serial_data:
                    if not isinstance(row, list) or cam_idx >= len(row):
                        continue
                    try:
                        serial_val = int(row[cam_idx])
                    except Exception:
                        continue
                    rng = serial_sample_map.get(serial_val)
                    if rng is None:
                        continue
                    s0, s1 = rng
                    sample_min = s0 if sample_min is None else min(sample_min, s0)
                    sample_max = s1 if sample_max is None else max(sample_max, s1)
                if sample_min is not None and sample_max is not None:
                    pair_info["sample_start"] = int(sample_min)
                    pair_info["sample_end"] = int(sample_max)

            pairs[key] = pair_info
            pairs_by_camera.setdefault(cam, []).append(key)

            if time_start and time_end:
                cur = camera_time_bounds.get(cam)
                if cur is None:
                    camera_time_bounds[cam] = {"start": time_start, "end": time_end}
                else:
                    if time_start < cur["start"]:
                        cur["start"] = time_start
                    if time_end > cur["end"]:
                        cur["end"] = time_end

            s0 = pair_info.get("sample_start")
            s1 = pair_info.get("sample_end")
            if isinstance(s0, int) and isinstance(s1, int):
                cur_s = camera_sample_bounds.get(cam)
                if cur_s is None:
                    camera_sample_bounds[cam] = {"start": s0, "end": s1}
                else:
                    if s0 < cur_s["start"]:
                        cur_s["start"] = s0
                    if s1 > cur_s["end"]:
                        cur_s["end"] = s1

    for cam, keys in pairs_by_camera.items():
        keys.sort()

    global_time_bounds: Optional[Dict[str, str]] = None
    if camera_time_bounds:
        starts = [v["start"] for v in camera_time_bounds.values()]
        ends = [v["end"] for v in camera_time_bounds.values()]
        global_time_bounds = {"start": min(starts), "end": max(ends)}

    global_sample_bounds: Optional[Dict[str, int]] = None
    if camera_sample_bounds:
        starts_s = [v["start"] for v in camera_sample_bounds.values()]
        ends_s = [v["end"] for v in camera_sample_bounds.values()]
        global_sample_bounds = {"start": min(starts_s), "end": max(ends_s)}

    return {
        "pairs": pairs,
        "pairs_by_camera": pairs_by_camera,
        "camera_time_bounds": camera_time_bounds,
        "camera_sample_bounds": camera_sample_bounds,
        "global_time_bounds": global_time_bounds,
        "global_sample_bounds": global_sample_bounds,
        "sample_ready": sample_ready,
    }


def _default_range_config(cameras: List[str]) -> Dict[str, Any]:
    return {
        "cameras": {
            str(cam): {
                "enabled": True,
                "mode": "manual",
                "time_start": "",
                "time_end": "",
                "time_zone": WEBUI_DEFAULT_TIME_ZONE,
                "sample_start": "",
                "sample_end": "",
            }
            for cam in cameras
        },
    }


def _coerce_range_config(raw: object, cameras: List[str]) -> Dict[str, Any]:
    cfg = _default_range_config(cameras)
    if not isinstance(raw, dict):
        return cfg

    cameras_raw = raw.get("cameras")
    if isinstance(cameras_raw, dict):
        for cam in cameras:
            cam_raw = cameras_raw.get(cam)
            if not isinstance(cam_raw, dict):
                continue
            cur = cfg["cameras"][cam]
            cur["enabled"] = bool(cam_raw.get("enabled", True))
            cur["mode"] = _normalize_range_mode(cam_raw.get("mode"))
            cur["time_start"] = _normalize_time_text(cam_raw.get("time_start"))
            cur["time_end"] = _normalize_time_text(cam_raw.get("time_end"))
            cur["time_zone"] = (
                str(cam_raw.get("time_zone") or WEBUI_DEFAULT_TIME_ZONE).strip()
                or WEBUI_DEFAULT_TIME_ZONE
            )
            cs0 = _parse_int_or_none(cam_raw.get("sample_start"))
            cs1 = _parse_int_or_none(cam_raw.get("sample_end"))
            cur["sample_start"] = "" if cs0 is None else str(cs0)
            cur["sample_end"] = "" if cs1 is None else str(cs1)

    return cfg


def _resolve_camera_rule(range_config: Dict[str, Any], cam: str) -> Dict[str, Any]:
    cam_rules = range_config.get("cameras") or {}
    base = dict(cam_rules.get(str(cam)) or {})
    return {
        "mode": _normalize_range_mode(base.get("mode")),
        "time_start": _normalize_time_text(base.get("time_start")),
        "time_end": _normalize_time_text(base.get("time_end")),
        "time_zone": (
            str(base.get("time_zone") or WEBUI_DEFAULT_TIME_ZONE).strip()
            or WEBUI_DEFAULT_TIME_ZONE
        ),
        "sample_start": _parse_int_or_none(base.get("sample_start")),
        "sample_end": _parse_int_or_none(base.get("sample_end")),
        "enabled": bool(base.get("enabled", True)),
    }


def _pair_matches_rule(pair_meta: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    mode = rule.get("mode")
    if mode == "time":
        t0 = _parse_time_text(rule.get("time_start"))
        t1 = _parse_time_text(rule.get("time_end"))
        p0 = _parse_time_text(pair_meta.get("time_start"))
        p1 = _parse_time_text(pair_meta.get("time_end"))
        if not t0 or not t1 or not p0 or not p1:
            return False
        if t1 < t0:
            return False
        return (p1 >= t0) and (p0 <= t1)
    if mode == "sample":
        s0 = _parse_int_or_none(rule.get("sample_start"))
        s1 = _parse_int_or_none(rule.get("sample_end"))
        p0 = _parse_int_or_none(pair_meta.get("sample_start"))
        p1 = _parse_int_or_none(pair_meta.get("sample_end"))
        if s0 is None or s1 is None or p0 is None or p1 is None:
            return False
        if s1 < s0:
            return False
        return (p1 >= s0) and (p0 <= s1)
    return False


def _compute_effective_target_pairs(
    *,
    manual_pairs: List[str],
    cameras: List[str],
    available_pair_map: Mapping[str, bool],
    range_config: Dict[str, Any],
    range_catalog: Dict[str, Any],
) -> Tuple[List[str], Optional[str]]:
    manual_set = {p for p in manual_pairs if p in available_pair_map}
    pairs_by_camera: Dict[str, List[str]] = range_catalog.get("pairs_by_camera") or {}
    pair_meta_map: Dict[str, Dict[str, Any]] = range_catalog.get("pairs") or {}

    selected: List[str] = []
    seen: set[str] = set()

    for cam in cameras:
        rule = _resolve_camera_rule(range_config, cam)
        if not bool(rule.get("enabled", True)):
            continue
        mode = rule.get("mode")
        if mode == "manual":
            for pair in manual_pairs:
                if pair in seen:
                    continue
                if pair not in manual_set:
                    continue
                if pair.endswith(f"::{cam}"):
                    seen.add(pair)
                    selected.append(pair)
            continue

        if mode == "time":
            if not rule.get("time_start") or not rule.get("time_end"):
                return [], f"Camera {cam}: both time start/end are required."
        elif mode == "sample":
            s0 = rule.get("sample_start")
            s1 = rule.get("sample_end")
            if s0 is None or s1 is None:
                return [], f"Camera {cam}: both sample start/end are required."
            if int(s1) < int(s0):
                return [], f"Camera {cam}: sample end must be >= sample start."
        else:
            return [], f"Camera {cam}: invalid range mode '{mode}'."

        for pair in pairs_by_camera.get(cam, []):
            if pair in seen or pair not in available_pair_map:
                continue
            meta = pair_meta_map.get(pair) or {}
            if _pair_matches_rule(meta, rule):
                seen.add(pair)
                selected.append(pair)

    if not selected:
        return [], (
            "No segment/camera pairs matched the current selection rules. "
            "Adjust manual picks or range values."
        )
    return selected, None


def _build_sync_run_groups(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected_pairs = _normalize_target_pairs(data.get("target_pairs"))
    if not selected_pairs:
        return []

    cameras = [str(c) for c in (data.get("cameras_all") or []) if str(c).strip()]
    if not cameras:
        cameras = sorted({p.split("::", 1)[1] for p in selected_pairs if "::" in p})
    range_config = _coerce_range_config(data.get("range_config"), cameras)

    grouped: Dict[Tuple[str, str, str, str], List[str]] = {}
    order: List[Tuple[str, str, str, str]] = []

    for pair in sorted(selected_pairs):
        cam = pair.split("::", 1)[1] if "::" in pair else ""
        rule = _resolve_camera_rule(range_config, cam)
        mode = rule.get("mode", "manual")
        if mode == "time":
            key = (
                "time",
                str(rule.get("time_start") or ""),
                str(rule.get("time_end") or ""),
                str(rule.get("time_zone") or WEBUI_DEFAULT_TIME_ZONE),
            )
        elif mode == "sample":
            key = (
                "sample",
                str(rule.get("sample_start") or ""),
                str(rule.get("sample_end") or ""),
                "",
            )
        else:
            key = ("manual", "", "", "")

        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(pair)

    runs: List[Dict[str, Any]] = []
    for key in order:
        mode, a, b, c = key
        entry: Dict[str, Any] = {
            "mode": mode,
            "target_pairs": grouped[key],
            "time_start": "",
            "time_end": "",
            "time_zone": "",
            "audio_sample_start": None,
            "audio_sample_end": None,
        }
        if mode == "time":
            entry["time_start"] = a
            entry["time_end"] = b
            entry["time_zone"] = c or WEBUI_DEFAULT_TIME_ZONE
        elif mode == "sample":
            entry["audio_sample_start"] = _parse_int_or_none(a)
            entry["audio_sample_end"] = _parse_int_or_none(b)
        runs.append(entry)
    return runs


def _draft_create(*, mode: str = "sync") -> int:
    conn = get_conn()
    now = utc_now_iso()
    data = {"mode": mode}
    if mode == "audio_timestamp":
        data["site"] = "nbu_lounge"
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO drafts(created_at, updated_at, data_json) VALUES(?, ?, ?)",
            (now, now, json.dumps(data)),
        )
        return int(cur.lastrowid)


def _draft_get(draft_id: int) -> Dict[str, Any]:
    conn = get_conn()
    row = conn.execute(
        "SELECT data_json FROM drafts WHERE id=?", (draft_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Draft not found")
    try:
        return json.loads(row["data_json"])
    except Exception:
        return {}


def _draft_set(draft_id: int, data: Dict[str, Any]) -> None:
    conn = get_conn()
    now = utc_now_iso()
    with tx(conn):
        conn.execute(
            "UPDATE drafts SET updated_at=?, data_json=? WHERE id=?",
            (now, json.dumps(data), draft_id),
        )


def _draft_delete(draft_id: int) -> None:
    conn = get_conn()
    with tx(conn):
        conn.execute("DELETE FROM drafts WHERE id=?", (draft_id,))


def _wizard_flow_context(data: Dict[str, Any], *, step: int) -> Dict[str, Any]:
    mode = str(data.get("mode") or "sync")
    if mode == "audio_timestamp":
        flow_steps = ["Audio", "Video", "Output", "Select", "Summary"]
        subtitles = {
            1: "Step 1/5 — Select audio folder",
            2: "Step 2/5 — Select video folder",
            3: "Step 3/5 — Select output folder",
            4: "Step 4/5 — Configure output",
            5: "Step 5/5 — Summary",
        }
        return {
            "mode": mode,
            "base_layout": "layout_timestamp.html",
            "flow_title": "Find audio timestamp",
            "flow_subtitle": subtitles.get(step),
            "flow_nav": "timestamp_new",
            "cancel_url": "/",
            "flow_steps": flow_steps,
            "select_hint": (
                "Select segment-camera pairs to use as reference anchors. "
                "-- = missing video. Click a segment or camera header to toggle its row/column."
            ),
            "show_reuse_audio": False,
        }
    return {
        "mode": mode,
        "base_layout": "layout_nbu.html",
        "flow_title": "New run",
        "flow_subtitle": None,
        "flow_nav": "new",
        "cancel_url": "/runs",
        "flow_steps": ["Audio", "Video", "Output", "Select", "Summary"],
        "select_hint": None,
        "show_reuse_audio": True,
    }


def _default_timestamp_output_path(data: Dict[str, Any]) -> str:
    out_dir = str(data.get("out_dir", "")).strip()
    if not out_dir:
        return ""
    try:
        return str(OutputLayout(Path(out_dir)).default_metadata_path)
    except Exception:
        return ""


def _resolve_timestamp_output_path(
    data: Dict[str, Any], *, override: Optional[str] = None
) -> str:
    raw = (override or "").strip()
    if not raw:
        raw = str(data.get("timestamp_output_path") or "").strip()
    if raw:
        try:
            return str(Path(raw).expanduser())
        except Exception:
            return raw
    return _default_timestamp_output_path(data)


def _create_timestamp_run(
    args: Dict[str, Any], *, output_path: Optional[str] = None
) -> int:
    now = utc_now_iso()
    conn = get_conn()
    with tx(conn):
        cur = conn.execute(
            """
            INSERT INTO timestamp_runs(created_at, started_at, status, args_json, output_path)
            VALUES(?, ?, ?, ?, ?)
            """,
            (now, now, "running", json.dumps(args), output_path),
        )
        return int(cur.lastrowid)


def _update_timestamp_run(
    run_id: int,
    *,
    status: Optional[str] = None,
    finished_at: Optional[str] = None,
    output_path: Optional[str] = None,
    records_count: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    fields = []
    values: List[Any] = []
    if status is not None:
        fields.append("status=?")
        values.append(status)
    if finished_at is not None:
        fields.append("finished_at=?")
        values.append(finished_at)
    if output_path is not None:
        fields.append("output_path=?")
        values.append(output_path)
    if records_count is not None:
        fields.append("records_count=?")
        values.append(records_count)
    if error is not None:
        fields.append("error=?")
        values.append(error)
    if not fields:
        return
    values.append(run_id)
    conn = get_conn()
    with tx(conn):
        conn.execute(
            f"UPDATE timestamp_runs SET {', '.join(fields)} WHERE id=?", values
        )


def _start_audio_timestamp_job(draft_id: int, *, run_id: int) -> None:
    def worker() -> None:
        data = _draft_get(draft_id)
        try:
            audio_dir = Path(str(data.get("audio_dir", "")).strip()).expanduser()
            video_dir = Path(str(data.get("video_dir", "")).strip()).expanduser()
            out_dir = Path(str(data.get("out_dir", "")).strip()).expanduser()
            site = str(data.get("site") or "nbu_lounge")

            records = compute_audio_start_records(
                audio_dir=audio_dir,
                video_dir=video_dir,
                out_dir=out_dir,
                local_tz=DEFAULT_TZ,
                site=site,
            )
            payload = [_record_to_payload(rec) for rec in records]
            output_path_raw = _resolve_timestamp_output_path(data)
            if not output_path_raw:
                raise RuntimeError("Output JSON path is required.")
            output_path = Path(output_path_raw)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            data = _draft_get(draft_id)
            data["timestamp_output_path"] = str(output_path)
            _draft_set(draft_id, data)
            _update_timestamp_run(
                run_id,
                status="succeeded",
                finished_at=utc_now_iso(),
                output_path=str(output_path),
                records_count=len(payload),
            )
        except Exception as exc:
            _update_timestamp_run(
                run_id,
                status="failed",
                finished_at=utc_now_iso(),
                error=str(exc),
            )

    threading.Thread(
        target=worker, name=f"audio-timestamp-{draft_id}", daemon=True
    ).start()


def create_app() -> FastAPI:
    app = FastAPI(title="video-sync-nbu Web UI")
    app.mount(
        "/static",
        StaticFiles(directory=str(ROOT / "static")),
        name="static",
    )

    max_parallel = int(os.environ.get("VSYNC_WEBUI_MAX_PARALLEL", "1"))
    runner = Runner(cfg=RunnerConfig(max_parallel=max_parallel))
    runner.start()

    audio_val_lock = threading.Lock()
    audio_val_state: Dict[int, Dict[str, Any]] = {}
    audio_check_names = [
        "Find audio files",
        "Naming pattern",
        "Serial channel present",
        "Program channel present",
        "Sample rate detected",
        "Duration detected",
    ]

    video_val_lock = threading.Lock()
    video_val_state: Dict[int, Dict[str, Any]] = {}
    video_check_names = [
        "Segment JSON present",
        "Camera MP4 present",
        "Segments discovered",
        "Companion JSON per segment",
        "Cameras discovered",
    ]

    out_val_lock = threading.Lock()
    out_val_state: Dict[int, Dict[str, Any]] = {}
    out_check_names = [
        "Directory empty",
        "Audio metadata present",
    ]

    @app.get("/api/video-index")
    def api_video_index(video_dir: str) -> Dict[str, Any]:
        """
        Return discovered segment IDs and camera serials for a given video directory.
        """
        p = Path(video_dir).expanduser()
        try:
            data = discover_from_video_dir(p)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to scan video_dir: {e}"
            )
        return {"video_dir": str(p), **data}

    @app.get("/api/audio-metadata")
    def api_audio_metadata(out_dir: str) -> Dict[str, Any]:
        out_dir = (out_dir or "").strip()
        if not out_dir:
            raise HTTPException(status_code=400, detail="out_dir is required")
        base = Path(out_dir).expanduser()
        path = base / "audio_metadata" / "audio_abs_start.json"
        if not path.exists():
            return {
                "ok": False,
                "path": str(path),
                "error": "audio_abs_start.json not found",
            }
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return {
                "ok": False,
                "path": str(path),
                "error": f"Failed to read JSON: {exc}",
            }
        max_chars = 200000
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
            truncated = True
        return {"ok": True, "path": str(path), "text": text, "truncated": truncated}

    @app.get("/api/pick-dir")
    def api_pick_dir(initial: str = "") -> Dict[str, Any]:
        """
        Open a native directory picker on the machine running the web server.

        Note: this cannot open a picker on a remote client's machine; it opens locally.
        """
        initial = (initial or "").strip()

        # Prefer platform-native pickers where possible (tkinter is flaky on some macOS setups).
        if sys.platform == "darwin":
            script_lines = [
                "try",
                'set promptText to "Select folder"',
            ]
            if initial:
                # If a file path is provided, use its parent directory as the initial folder.
                p = Path(initial).expanduser()
                if p.is_file():
                    p = p.parent
                initial_dir = str(p)
                # Escape quotes for AppleScript.
                initial_dir = initial_dir.replace('"', '\\"')
                script_lines.append(
                    f'set chosen to (choose folder with prompt promptText default location (POSIX file "{initial_dir}"))'
                )
            else:
                script_lines.append(
                    "set chosen to (choose folder with prompt promptText)"
                )
            script_lines += [
                "POSIX path of chosen",
                "on error number -128",
                '""',
                "end try",
            ]
            script = "\n".join(script_lines)
            try:
                r = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
                chosen = (r.stdout or "").strip()
                if chosen:
                    return {"path": chosen, "canceled": False}
                return {"path": "", "canceled": True}
            except Exception:
                # Fall back to tkinter.
                pass

        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return {
                "path": "",
                "canceled": False,
                "error": "Directory picker requires tkinter (not available in this Python environment).",
            }

        try:
            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass
            chosen = filedialog.askdirectory(
                initialdir=initial or str(Path.cwd()),
                title="Select folder",
            )
            root.destroy()
        except Exception:
            try:
                root.destroy()  # type: ignore[name-defined]
            except Exception:
                pass
            return {
                "path": "",
                "canceled": False,
                "error": "Directory picker failed to open.",
            }

        if chosen:
            return {"path": str(chosen), "canceled": False}
        return {"path": "", "canceled": True}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("home.html", {"request": request})

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request) -> HTMLResponse:
        conn = get_conn()
        items = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 200").fetchall()
        runs_view: List[Dict[str, Any]] = []
        for row in items:
            d = dict(row)
            try:
                args = json.loads(d.get("args_json") or "{}")
            except Exception:
                args = {}
            d["display_title"] = _segment_range_title(args)
            runs_view.append(d)
        return templates.TemplateResponse(
            "runs.html",
            {"request": request, "runs": runs_view},
        )

    @app.get("/audio-timestamp/runs", response_class=HTMLResponse)
    def audio_timestamp_runs(request: Request) -> HTMLResponse:
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM timestamp_runs ORDER BY id DESC LIMIT 200"
        ).fetchall()
        runs_view: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            try:
                args = json.loads(d.get("args_json") or "{}")
            except Exception:
                args = {}
            d["site"] = args.get("site", "")
            runs_view.append(d)
        return templates.TemplateResponse(
            "audio_timestamp_runs.html",
            {"request": request, "runs": runs_view},
        )

    @app.get("/api/runs")
    def api_runs() -> List[Dict[str, Any]]:
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, status, exit_code FROM runs ORDER BY id DESC LIMIT 200"
        ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/audio-timestamp/runs")
    def api_audio_timestamp_runs() -> List[Dict[str, Any]]:
        conn = get_conn()
        rows = conn.execute(
            """
            SELECT id, status, finished_at, records_count
            FROM timestamp_runs
            ORDER BY id DESC
            LIMIT 200
            """
        ).fetchall()
        return [dict(r) for r in rows]

    @app.post("/runs/clear")
    def runs_clear() -> RedirectResponse:
        """
        Clear run history (DB rows + log files) for all runs that are not running.
        """
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, status, log_path FROM runs WHERE status != 'running'"
        ).fetchall()
        for r in rows:
            try:
                p = Path(r["log_path"])
                if p.exists():
                    p.unlink()
            except Exception:
                # Best-effort deletion; DB clear still proceeds.
                pass
        with tx(conn):
            conn.execute("DELETE FROM runs WHERE status != 'running'")
        return RedirectResponse(url="/runs", status_code=303)

    @app.get("/runs/new", response_class=HTMLResponse)
    def runs_new() -> RedirectResponse:
        draft_id = _draft_create(mode="sync")
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @app.get("/tools/audio-timestamp")
    def tools_audio_timestamp() -> RedirectResponse:
        draft_id = _draft_create(mode="audio_timestamp")
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @app.get("/tools/audio-timestamp/new")
    def tools_audio_timestamp_new() -> RedirectResponse:
        draft_id = _draft_create(mode="audio_timestamp")
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @app.get("/api/drafts/{draft_id}/validate-audio")
    def api_validate_audio(draft_id: int, audio_dir: str) -> Dict[str, Any]:
        """
        Validate audio_dir and persist the result into the run-creation draft.
        """
        data = _draft_get(draft_id)
        if data.get("audio_dir") != audio_dir.strip():
            # Changing audio dir invalidates downstream choices.
            data.pop("video_dir", None)
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("target_pairs", None)
            data.pop("manual_target_pairs", None)
            data.pop("range_config", None)
            data.pop("out_dir", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)
        data["audio_dir"] = audio_dir.strip()

        result = validate_audio_dir(
            audio_dir, logger=logging.getLogger("webui.validate.audio")
        )
        data["audio_ok"] = bool(result.get("ok"))
        data["audio_result"] = result
        data["audio_error"] = result.get("error")
        _draft_set(draft_id, data)
        return result

    @app.get("/api/drafts/{draft_id}/validate-audio/start")
    def api_validate_audio_start(draft_id: int, audio_dir: str) -> Dict[str, Any]:
        """
        Start async audio validation so the UI can show per-check progress.
        """
        data = _draft_get(draft_id)
        audio_dir = audio_dir.strip()
        if data.get("audio_dir") != audio_dir:
            # Changing audio dir invalidates downstream choices.
            data.pop("video_dir", None)
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("target_pairs", None)
            data.pop("manual_target_pairs", None)
            data.pop("range_config", None)
            data.pop("out_dir", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)
            data.pop("timestamp_output_path", None)
            data.pop("timestamp_run_id", None)
        data["audio_dir"] = audio_dir
        _draft_set(draft_id, data)

        with audio_val_lock:
            st = audio_val_state.get(draft_id)
            if st and st.get("running") and st.get("audio_dir") == audio_dir:
                return st.get("result") or {
                    "audio_dir": audio_dir,
                    "ok": False,
                    "running": True,
                    "files": [],
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in audio_check_names
                    ],
                    "error": None,
                }

            seq = int(st.get("seq", 0)) + 1 if st else 1
            audio_val_state[draft_id] = {
                "seq": seq,
                "audio_dir": audio_dir,
                "running": True,
                "result": {
                    "audio_dir": audio_dir,
                    "ok": False,
                    "running": True,
                    "files": [],
                    "nested_files": [],
                    "nested_count": 0,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in audio_check_names
                    ],
                    "error": None,
                },
            }

        def run_validation(local_seq: int, local_audio_dir: str) -> None:
            def on_progress(payload: Dict[str, Any]) -> None:
                # Deep-copy into state to avoid sharing mutable dicts across threads.
                snap = json.loads(json.dumps(payload))
                with audio_val_lock:
                    cur = audio_val_state.get(draft_id)
                    if not cur or int(cur.get("seq", 0)) != local_seq:
                        return
                    cur["result"] = snap
                    cur["running"] = bool(snap.get("running", False))

            result = validate_audio_dir_progress(
                local_audio_dir,
                logger=logging.getLogger("webui.validate.audio"),
                on_progress=on_progress,
            )
            result["running"] = False
            on_progress(result)

            with audio_val_lock:
                cur = audio_val_state.get(draft_id)
                if not cur or int(cur.get("seq", 0)) != local_seq:
                    return
                cur["running"] = False
                cur["result"] = json.loads(json.dumps(result))

            # Persist the final result to the draft.
            data2 = _draft_get(draft_id)
            if data2.get("audio_dir") != local_audio_dir:
                return
            data2["audio_ok"] = bool(result.get("ok"))
            data2["audio_result"] = result
            data2["audio_error"] = result.get("error")
            _draft_set(draft_id, data2)

        threading.Thread(
            target=run_validation,
            args=(seq, audio_dir),
            name=f"webui-audio-validate-{draft_id}",
            daemon=True,
        ).start()

        with audio_val_lock:
            return audio_val_state[draft_id]["result"]

    @app.get("/api/drafts/{draft_id}/validate-audio/status")
    def api_validate_audio_status(draft_id: int) -> Dict[str, Any]:
        with audio_val_lock:
            st = audio_val_state.get(draft_id)
            if st and st.get("result") is not None:
                return st["result"]
        data = _draft_get(draft_id)
        if data.get("audio_result"):
            return data["audio_result"]
        return {
            "audio_dir": data.get("audio_dir", ""),
            "ok": False,
            "running": False,
            "files": [],
            "checks": [
                {"name": n, "status": "pending", "message": ""}
                for n in audio_check_names
            ],
            "error": None,
        }

    @app.get("/api/drafts/{draft_id}/validate-video")
    def api_validate_video(draft_id: int, video_dir: str) -> Dict[str, Any]:
        """
        Validate video_dir and persist the result into the run-creation draft.
        """
        data = _draft_get(draft_id)
        if data.get("video_dir") != video_dir.strip():
            # Changing video dir invalidates downstream choices.
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("all_segments", None)
            data.pop("all_cameras", None)
            data.pop("target_pairs", None)
            data.pop("manual_target_pairs", None)
            data.pop("range_config", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)
        data["video_dir"] = video_dir.strip()

        result = validate_video_dir(video_dir)
        data["video_ok"] = bool(result.get("ok"))
        data["video_result"] = result
        data["video_error"] = result.get("error")
        if data["video_ok"]:
            try:
                idx = discover_from_video_dir(Path(video_dir).expanduser())
                data["segments_all"] = idx["segments"]
                data["cameras_all"] = idx["cameras"]
            except Exception as e:
                data["video_ok"] = False
                data["video_error"] = str(e)
                data["video_result"] = {**result, "ok": False, "error": str(e)}
        _draft_set(draft_id, data)
        return data.get("video_result") or result

    @app.get("/api/drafts/{draft_id}/validate-video/start")
    def api_validate_video_start(draft_id: int, video_dir: str) -> Dict[str, Any]:
        """
        Start async video validation so the UI can show per-check progress.
        """
        data = _draft_get(draft_id)
        video_dir = video_dir.strip()
        if data.get("video_dir") != video_dir:
            # Changing video dir invalidates downstream choices.
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("all_segments", None)
            data.pop("all_cameras", None)
            data.pop("target_pairs", None)
            data.pop("manual_target_pairs", None)
            data.pop("range_config", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)
        data["video_dir"] = video_dir
        _draft_set(draft_id, data)

        with video_val_lock:
            st = video_val_state.get(draft_id)
            if st and st.get("running") and st.get("video_dir") == video_dir:
                return st.get("result") or {
                    "video_dir": video_dir,
                    "ok": False,
                    "running": True,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in video_check_names
                    ],
                    "error": None,
                }

            seq = int(st.get("seq", 0)) + 1 if st else 1
            video_val_state[draft_id] = {
                "seq": seq,
                "video_dir": video_dir,
                "running": True,
                "result": {
                    "video_dir": video_dir,
                    "ok": False,
                    "running": True,
                    "segments_count": 0,
                    "cameras_count": 0,
                    "segments_first": None,
                    "segments_last": None,
                    "segments_preview": [],
                    "cameras": [],
                    "cameras_preview": [],
                    "json_count": 0,
                    "mp4_count": 0,
                    "nested_files": [],
                    "nested_count": 0,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in video_check_names
                    ],
                    "error": None,
                },
            }

        def run_validation(local_seq: int, local_video_dir: str) -> None:
            def on_progress(payload: Dict[str, Any]) -> None:
                snap = json.loads(json.dumps(payload))
                with video_val_lock:
                    cur = video_val_state.get(draft_id)
                    if not cur or int(cur.get("seq", 0)) != local_seq:
                        return
                    cur["result"] = snap
                    cur["running"] = bool(snap.get("running", False))

            result = validate_video_dir_progress(
                local_video_dir, on_progress=on_progress
            )
            result["running"] = False
            on_progress(result)

            with video_val_lock:
                cur = video_val_state.get(draft_id)
                if not cur or int(cur.get("seq", 0)) != local_seq:
                    return
                cur["running"] = False
                cur["result"] = json.loads(json.dumps(result))

            # Persist the final result to the draft.
            data2 = _draft_get(draft_id)
            if data2.get("video_dir") != local_video_dir:
                return
            data2["video_ok"] = bool(result.get("ok"))
            data2["video_result"] = result
            data2["video_error"] = result.get("error")
            if data2["video_ok"]:
                try:
                    idx = discover_from_video_dir(Path(local_video_dir).expanduser())
                    data2["segments_all"] = idx["segments"]
                    data2["cameras_all"] = idx["cameras"]
                except Exception as e:
                    data2["video_ok"] = False
                    data2["video_error"] = str(e)
                    data2["video_result"] = {**result, "ok": False, "error": str(e)}
            _draft_set(draft_id, data2)

        threading.Thread(
            target=run_validation,
            args=(seq, video_dir),
            name=f"webui-video-validate-{draft_id}",
            daemon=True,
        ).start()

        with video_val_lock:
            return video_val_state[draft_id]["result"]

    @app.get("/api/drafts/{draft_id}/validate-video/status")
    def api_validate_video_status(draft_id: int) -> Dict[str, Any]:
        with video_val_lock:
            st = video_val_state.get(draft_id)
            if st and st.get("result") is not None:
                return st["result"]
        data = _draft_get(draft_id)
        if data.get("video_result"):
            return data["video_result"]
        return {
            "video_dir": data.get("video_dir", ""),
            "ok": False,
            "running": False,
            "segments_count": 0,
            "cameras_count": 0,
            "segments_first": None,
            "segments_last": None,
            "segments_preview": [],
            "cameras": [],
            "cameras_preview": [],
            "json_count": 0,
            "mp4_count": 0,
            "nested_files": [],
            "nested_count": 0,
            "checks": [
                {"name": n, "status": "pending", "message": ""}
                for n in video_check_names
            ],
            "error": None,
        }

    @app.get("/api/drafts/{draft_id}/validate-out")
    def api_validate_out(draft_id: int, out_dir: str) -> Dict[str, Any]:
        """
        Validate out_dir and persist the result into the run-creation draft.
        """
        data = _draft_get(draft_id)
        out_dir = out_dir.strip()
        if data.get("out_dir") != out_dir:
            data.pop("target_pairs", None)
            data.pop("manual_target_pairs", None)
            data.pop("range_config", None)
        data["out_dir"] = out_dir

        segments = data.get("segments_all")
        if not isinstance(segments, list):
            segments = None

        result = validate_out_dir(out_dir, segments=segments)
        data["out_ok"] = bool(result.get("ok"))
        data["out_result"] = result
        data["out_error"] = result.get("error")
        _draft_set(draft_id, data)
        return result

    @app.get("/api/drafts/{draft_id}/validate-out/start")
    def api_validate_out_start(draft_id: int, out_dir: str) -> Dict[str, Any]:
        """
        Start async out_dir validation so the UI can show per-check progress.
        """
        data = _draft_get(draft_id)
        out_dir = out_dir.strip()
        if data.get("out_dir") != out_dir:
            data.pop("target_pairs", None)
            data.pop("manual_target_pairs", None)
            data.pop("range_config", None)
        data["out_dir"] = out_dir
        _draft_set(draft_id, data)

        segments = data.get("segments_all")
        if not isinstance(segments, list):
            segments = None

        with out_val_lock:
            st = out_val_state.get(draft_id)
            if st and st.get("running") and st.get("out_dir") == out_dir:
                return st.get("result") or {
                    "out_dir": out_dir,
                    "ok": False,
                    "running": True,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in out_check_names
                    ],
                    "error": None,
                }

            seq = int(st.get("seq", 0)) + 1 if st else 1
            out_val_state[draft_id] = {
                "seq": seq,
                "out_dir": out_dir,
                "running": True,
                "result": {
                    "out_dir": out_dir,
                    "ok": False,
                    "running": True,
                    "exists": False,
                    "will_create": False,
                    "empty": None,
                    "entries_count": 0,
                    "audio_decoded": {
                        "exists": False,
                        "raw_csv": False,
                        "filtered_csv": False,
                    },
                    "can_reuse_audio_decoded": False,
                    "split_decoded_count": 0,
                    "serial_chunks_count": 0,
                    "synced_segments_count": 0,
                    "synced_segments_preview": [],
                    "total_segments": None,
                    "checks": [
                        {"name": n, "status": "pending", "message": ""}
                        for n in out_check_names
                    ],
                    "error": None,
                },
            }

        def run_validation(local_seq: int, local_out_dir: str) -> None:
            def on_progress(payload: Dict[str, Any]) -> None:
                snap = json.loads(json.dumps(payload))
                with out_val_lock:
                    cur = out_val_state.get(draft_id)
                    if not cur or int(cur.get("seq", 0)) != local_seq:
                        return
                    cur["result"] = snap
                    cur["running"] = bool(snap.get("running", False))

            from scripts.validate.validate_out_dir import validate_out_dir_progress

            result = validate_out_dir_progress(
                local_out_dir, segments=segments, on_progress=on_progress
            )
            result["running"] = False
            on_progress(result)

            with out_val_lock:
                cur = out_val_state.get(draft_id)
                if not cur or int(cur.get("seq", 0)) != local_seq:
                    return
                cur["running"] = False
                cur["result"] = json.loads(json.dumps(result))

            data2 = _draft_get(draft_id)
            if data2.get("out_dir") != local_out_dir:
                return
            data2["out_ok"] = bool(result.get("ok"))
            data2["out_result"] = result
            data2["out_error"] = result.get("error")
            _draft_set(draft_id, data2)

        threading.Thread(
            target=run_validation,
            args=(seq, out_dir),
            name=f"webui-out-validate-{draft_id}",
            daemon=True,
        ).start()

        with out_val_lock:
            return out_val_state[draft_id]["result"]

    @app.get("/api/drafts/{draft_id}/validate-out/status")
    def api_validate_out_status(draft_id: int) -> Dict[str, Any]:
        with out_val_lock:
            st = out_val_state.get(draft_id)
            if st and st.get("result") is not None:
                return st["result"]
        data = _draft_get(draft_id)
        if data.get("out_result"):
            return data["out_result"]
        return {
            "out_dir": data.get("out_dir", ""),
            "ok": False,
            "running": False,
            "exists": False,
            "will_create": False,
            "empty": None,
            "entries_count": 0,
            "audio_decoded": {
                "exists": False,
                "raw_csv": False,
                "filtered_csv": False,
            },
            "can_reuse_audio_decoded": False,
            "split_decoded_count": 0,
            "serial_chunks_count": 0,
            "audio_metadata": {
                "exists": False,
                "abs_json": False,
            },
            "synced_segments_count": 0,
            "synced_segments_preview": [],
            "total_segments": None,
            "checks": [
                {"name": n, "status": "pending", "message": ""} for n in out_check_names
            ],
            "error": None,
        }

    @app.get("/wizard/{draft_id}/audio", response_class=HTMLResponse)
    def wizard_audio(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        return templates.TemplateResponse(
            "wizard_audio.html",
            {
                "request": request,
                "draft_id": draft_id,
                "audio_dir": data.get("audio_dir", ""),
                "audio_ok": bool(data.get("audio_ok", False)),
                "audio_result": data.get("audio_result"),
                "error": data.get("audio_error"),
                **_wizard_flow_context(data, step=1),
            },
        )

    @app.post("/wizard/{draft_id}/audio")
    def wizard_audio_post(
        draft_id: int, audio_dir: str = Form(...)
    ) -> RedirectResponse:
        audio_dir = audio_dir.strip()
        data = _draft_get(draft_id)
        if data.get("audio_dir") != audio_dir:
            # Changing audio dir invalidates downstream choices.
            data.pop("video_dir", None)
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("target_pairs", None)
            data.pop("manual_target_pairs", None)
            data.pop("range_config", None)
            data.pop("out_dir", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)

        data["audio_dir"] = audio_dir
        # Validate on POST as a no-JS fallback (JS uses /api/drafts/<id>/validate-audio).
        result = validate_audio_dir(
            audio_dir, logger=logging.getLogger("webui.validate.audio")
        )
        data["audio_ok"] = bool(result.get("ok"))
        data["audio_result"] = result
        data["audio_error"] = result.get("error")
        _draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)

    @app.get("/wizard/{draft_id}/video", response_class=HTMLResponse)
    def wizard_video(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            # If a user jumps directly here, we still allow them to set video_dir and validate.
            pass
        return templates.TemplateResponse(
            "wizard_video.html",
            {
                "request": request,
                "draft_id": draft_id,
                "video_dir": data.get("video_dir", ""),
                "video_ok": bool(data.get("video_ok", False)),
                "video_result": data.get("video_result"),
                "error": data.get("video_error"),
                **_wizard_flow_context(data, step=2),
            },
        )

    @app.post("/wizard/{draft_id}/video")
    def wizard_video_post(
        draft_id: int, video_dir: str = Form(...)
    ) -> RedirectResponse:
        video_dir = video_dir.strip()
        data = _draft_get(draft_id)
        if data.get("video_dir") != video_dir:
            data.pop("segments_all", None)
            data.pop("cameras_all", None)
            data.pop("segments", None)
            data.pop("cameras", None)
            data.pop("all_segments", None)
            data.pop("all_cameras", None)
            data.pop("target_pairs", None)
            data.pop("manual_target_pairs", None)
            data.pop("range_config", None)
            data.pop("out_ok", None)
            data.pop("out_result", None)
            data.pop("out_error", None)
            data.pop("timestamp_output_path", None)
            data.pop("timestamp_run_id", None)

        data["video_dir"] = video_dir
        result = validate_video_dir(video_dir)
        data["video_ok"] = bool(result.get("ok"))
        data["video_result"] = result
        data["video_error"] = result.get("error")
        if data["video_ok"]:
            try:
                idx = discover_from_video_dir(Path(video_dir).expanduser())
                data["segments_all"] = idx["segments"]
                data["cameras_all"] = idx["cameras"]
            except Exception as e:
                data["video_ok"] = False
                data["video_error"] = str(e)
                data["video_result"] = {**result, "ok": False, "error": str(e)}

        _draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)

    @app.get("/wizard/{draft_id}/output", response_class=HTMLResponse)
    def wizard_output(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        mode = str(data.get("mode") or "sync")
        template_name = (
            "wizard_output_timestamp.html"
            if mode == "audio_timestamp"
            else "wizard_output_sync.html"
        )
        continue_url = (
            f"/wizard/{draft_id}/select"
            if mode == "audio_timestamp"
            else f"/wizard/{draft_id}/select"
        )
        return templates.TemplateResponse(
            template_name,
            {
                "request": request,
                "draft_id": draft_id,
                "out_dir": data.get("out_dir", ""),
                "out_ok": bool(data.get("out_ok", False)),
                "out_result": data.get("out_result"),
                "error": data.get("out_error"),
                "continue_url": continue_url,
                **_wizard_flow_context(data, step=3),
            },
        )

    @app.post("/wizard/{draft_id}/output")
    def wizard_output_post(draft_id: int, out_dir: str = Form(...)) -> RedirectResponse:
        out_dir = out_dir.strip()
        data = _draft_get(draft_id)
        if data.get("out_dir") != out_dir:
            data.pop("target_pairs", None)
            data.pop("manual_target_pairs", None)
            data.pop("range_config", None)
            data.pop("timestamp_output_path", None)
            data.pop("timestamp_run_id", None)
        data["out_dir"] = out_dir
        segments = data.get("segments_all")
        if not isinstance(segments, list):
            segments = None
        result = validate_out_dir(out_dir, segments=segments)
        data["out_ok"] = bool(result.get("ok"))
        data["out_result"] = result
        data["out_error"] = result.get("error")
        _draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

    @app.get("/wizard/{draft_id}/select", response_class=HTMLResponse)
    def wizard_select(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        if str(data.get("mode") or "sync") == "audio_timestamp":
            if not bool(data.get("audio_ok", False)):
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/audio", status_code=303
                )
            if not bool(data.get("video_ok", False)):
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/video", status_code=303
                )
            if not bool(data.get("out_ok", False)):
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/output", status_code=303
                )
            output_json = _resolve_timestamp_output_path(data)
            return templates.TemplateResponse(
                "wizard_timestamp_select.html",
                {
                    "request": request,
                    "draft_id": draft_id,
                    "site": data.get("site", "nbu_lounge"),
                    "output_json": output_json or "",
                    "error": data.get("select_error"),
                    **_wizard_flow_context(data, step=4),
                },
            )
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)
        video_result = data.get("video_result") or {}
        out_result = data.get("out_result") or {}
        available_pair_map = _extract_available_pair_map(video_result)
        missing_json_map = _extract_missing_json_map(video_result)
        synced_pair_map = _extract_synced_pair_map(out_result)

        cameras = [str(c) for c in (data.get("cameras_all") or []) if str(c).strip()]
        if not cameras:
            cameras = sorted(
                {k.split("::", 1)[1] for k in available_pair_map.keys() if "::" in k}
            )
        range_config = _coerce_range_config(data.get("range_config"), cameras)
        selection_mode = str(data.get("selection_mode") or "segments").strip().lower()
        if selection_mode not in {"segments", "time", "sample"}:
            selection_mode = "segments"
        video_dir = Path(str(data.get("video_dir", "")).strip()).expanduser()
        out_dir = Path(str(data.get("out_dir", "")).strip()).expanduser()
        try:
            range_catalog = _build_range_catalog(
                video_dir=video_dir,
                out_dir=out_dir,
                available_pair_map=available_pair_map,
            )
        except Exception:
            range_catalog = {
                "pairs": {},
                "pairs_by_camera": {},
                "camera_time_bounds": {},
                "camera_sample_bounds": {},
                "global_time_bounds": None,
                "global_sample_bounds": None,
                "sample_ready": False,
            }
        return templates.TemplateResponse(
            "wizard_select.html",
            {
                "request": request,
                "draft_id": draft_id,
                "site": data.get("site", "nbu_lounge"),
                "segments": data.get("segments_all", []),
                "cameras": cameras,
                "selected_pairs": data.get(
                    "manual_target_pairs", data.get("target_pairs", [])
                ),
                "skip_decode": bool(data.get("skip_decode", False)),
                "run_error": data.get("run_error"),
                "reuse_audio_available": bool(
                    out_result.get("audio_decoded", {}).get("filtered_csv")
                ),
                "available_pair_map": available_pair_map,
                "missing_json_map": missing_json_map,
                "synced_pair_map": synced_pair_map,
                "range_config": range_config,
                "range_catalog": range_catalog,
                "selection_mode": selection_mode,
                "error": data.get("select_error"),
                **_wizard_flow_context(data, step=4),
            },
        )

    @app.post("/wizard/{draft_id}/select")
    async def wizard_select_post(draft_id: int, request: Request) -> RedirectResponse:
        form = await request.form()
        site = str(form.get("site") or "nbu_lounge").strip() or "nbu_lounge"
        log_level = str(form.get("log_level") or "INFO").strip() or "INFO"
        reuse_audio = form.get("reuse_audio")
        output_json = str(form.get("output_json") or "").strip() or None
        selection_mode = str(form.get("selection_mode") or "segments").strip().lower()
        if selection_mode not in {"segments", "time", "sample"}:
            selection_mode = "segments"
        picked_pairs = _normalize_target_pairs(
            [str(v) for v in form.getlist("target_pairs")]
        )

        data = _draft_get(draft_id)
        mode = str(data.get("mode") or "sync")
        data["site"] = site
        data["all_segments"] = False
        data["all_cameras"] = False
        data["segments"] = []
        data["cameras"] = []
        data["selection_mode"] = selection_mode

        data["select_error"] = None
        data["run_error"] = None
        if mode == "audio_timestamp":
            out_path = _resolve_timestamp_output_path(data, override=output_json)
            if not out_path:
                data["select_error"] = "Output JSON path is required."
                _draft_set(draft_id, data)
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/select", status_code=303
                )
            data["timestamp_output_path"] = out_path
            data["target_pairs"] = []
            data["log_level"] = log_level
            data["skip_decode"] = False
            data["overwrite_clips"] = False
            data["split"] = False
            data["split_overwrite"] = False
            data["split_clean"] = False
            data["split_chunk_seconds"] = 3600
            data["schedule_at"] = ""
            data["timestamp_run_id"] = None
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/summary", status_code=303)

        video_result = data.get("video_result") or {}
        available_pair_map = _extract_available_pair_map(video_result)
        cameras = [str(c) for c in (data.get("cameras_all") or []) if str(c).strip()]
        if not cameras:
            cameras = sorted(
                {k.split("::", 1)[1] for k in available_pair_map.keys() if "::" in k}
            )
        prev_cfg = _coerce_range_config(data.get("range_config"), cameras)
        range_config = _default_range_config(cameras)
        for cam in cameras:
            rule = range_config["cameras"][cam]
            prev_rule = (
                (prev_cfg.get("cameras") or {}).get(cam)
                if isinstance(prev_cfg.get("cameras"), dict)
                else {}
            ) or {}
            if selection_mode == "time":
                enabled_raw = (
                    str(form.get(f"time_camera_enabled_{cam}") or "1").strip().lower()
                )
                rule["enabled"] = enabled_raw not in {"0", "false", "off", "no"}
            elif selection_mode == "sample":
                enabled_raw = (
                    str(form.get(f"sample_camera_enabled_{cam}") or "1").strip().lower()
                )
                rule["enabled"] = enabled_raw not in {"0", "false", "off", "no"}
            else:
                rule["enabled"] = True
            if selection_mode == "segments":
                rule["mode"] = "manual"
                rule["time_start"] = _normalize_time_text(prev_rule.get("time_start"))
                rule["time_end"] = _normalize_time_text(prev_rule.get("time_end"))
                rule["time_zone"] = (
                    str(prev_rule.get("time_zone") or WEBUI_DEFAULT_TIME_ZONE).strip()
                    or WEBUI_DEFAULT_TIME_ZONE
                )
                ss0 = _parse_int_or_none(prev_rule.get("sample_start"))
                ss1 = _parse_int_or_none(prev_rule.get("sample_end"))
                rule["sample_start"] = "" if ss0 is None else str(ss0)
                rule["sample_end"] = "" if ss1 is None else str(ss1)
            elif selection_mode == "time":
                rule["mode"] = "time"
                ts_key = f"time_start_{cam}"
                te_key = f"time_end_{cam}"
                tz_key = f"time_zone_{cam}"
                ts_raw = (
                    form.get(ts_key) if ts_key in form else prev_rule.get("time_start")
                )
                te_raw = (
                    form.get(te_key) if te_key in form else prev_rule.get("time_end")
                )
                tz_raw = (
                    form.get(tz_key) if tz_key in form else prev_rule.get("time_zone")
                )
                rule["time_start"] = _normalize_time_text(ts_raw)
                rule["time_end"] = _normalize_time_text(te_raw)
                rule["time_zone"] = (
                    str(tz_raw or WEBUI_DEFAULT_TIME_ZONE).strip()
                    or WEBUI_DEFAULT_TIME_ZONE
                )
            else:
                rule["mode"] = "sample"
                ss_key = f"sample_start_{cam}"
                se_key = f"sample_end_{cam}"
                ss_raw = (
                    form.get(ss_key)
                    if ss_key in form
                    else prev_rule.get("sample_start")
                )
                se_raw = (
                    form.get(se_key) if se_key in form else prev_rule.get("sample_end")
                )
                s0 = _parse_int_or_none(ss_raw)
                s1 = _parse_int_or_none(se_raw)
                rule["sample_start"] = "" if s0 is None else str(s0)
                rule["sample_end"] = "" if s1 is None else str(s1)
        data["range_config"] = range_config
        data["manual_target_pairs"] = picked_pairs

        video_dir = Path(str(data.get("video_dir", "")).strip()).expanduser()
        out_dir_path = Path(str(data.get("out_dir", "")).strip()).expanduser()
        range_catalog = _build_range_catalog(
            video_dir=video_dir,
            out_dir=out_dir_path,
            available_pair_map=available_pair_map,
        )
        if selection_mode == "segments" and not picked_pairs:
            data["select_error"] = "Select at least one segment/camera pair."
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

        effective_pairs, range_error = _compute_effective_target_pairs(
            manual_pairs=picked_pairs,
            cameras=cameras,
            available_pair_map=available_pair_map,
            range_config=range_config,
            range_catalog=range_catalog,
        )
        if range_error:
            data["select_error"] = range_error
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)
        data["target_pairs"] = effective_pairs

        data["title"] = ""
        data["log_level"] = log_level
        can_reuse = bool(
            (data.get("out_result") or {}).get("audio_decoded", {}).get("filtered_csv")
        )
        reuse_enabled = reuse_audio is not None and can_reuse
        data["skip_decode"] = reuse_enabled
        overwrite_clips = False
        out_dir = str(data.get("out_dir", "")).strip()
        if out_dir and effective_pairs:
            segs = sorted({p.split("::", 1)[0] for p in effective_pairs if "::" in p})
            cams = sorted({p.split("::", 1)[1] for p in effective_pairs if "::" in p})
            synced_pairs = discover_synced_pairs(
                out_dir, segments=segs or None, cameras=cams or None
            ).get("synced_pairs", [])
            synced_set = {
                f"{p.get('segment','')}::{p.get('camera','')}"
                for p in synced_pairs
                if p.get("segment") and p.get("camera")
            }
            overwrite_clips = any(pair in synced_set for pair in effective_pairs)
        data["overwrite_clips"] = overwrite_clips
        data["split"] = not reuse_enabled
        data["split_overwrite"] = not reuse_enabled
        data["split_clean"] = False
        data["split_chunk_seconds"] = 3600
        data["schedule_at"] = ""

        _draft_set(draft_id, data)
        return RedirectResponse(url=f"/wizard/{draft_id}/summary", status_code=303)

    @app.get("/wizard/{draft_id}/run")
    def wizard_run(draft_id: int) -> RedirectResponse:
        return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

    @app.post("/wizard/{draft_id}/run")
    def wizard_run_post(draft_id: int) -> RedirectResponse:
        return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

    @app.get("/wizard/{draft_id}/summary", response_class=HTMLResponse)
    def wizard_draft_summary(request: Request, draft_id: int) -> HTMLResponse:
        data = _draft_get(draft_id)
        mode = str(data.get("mode") or "sync")
        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        if not str(data.get("out_dir", "")).strip():
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        args: Dict[str, Any] = {
            "audio_dir": data.get("audio_dir", ""),
            "video_dir": data.get("video_dir", ""),
            "out_dir": data.get("out_dir", ""),
            "site": data.get("site", "nbu_lounge"),
            "segments": data.get("segments", []),
            "cameras": data.get("cameras", []),
            "target_pairs": data.get("target_pairs", []),
            "log_level": data.get("log_level", "INFO"),
            "skip_decode": bool(data.get("skip_decode", False)),
            "overwrite_clips": bool(data.get("overwrite_clips", False)),
            "split": bool(data.get("split", False)),
            "split_overwrite": bool(data.get("split_overwrite", False)),
            "split_clean": bool(data.get("split_clean", False)),
            "split_chunk_seconds": int(data.get("split_chunk_seconds", 3600)),
            "selection_mode": str(data.get("selection_mode") or "segments"),
            "range_config": _coerce_range_config(
                data.get("range_config"),
                [str(c) for c in (data.get("cameras_all") or []) if str(c).strip()],
            ),
        }
        if mode == "audio_timestamp":
            output_path = _resolve_timestamp_output_path(data)
            return templates.TemplateResponse(
                "wizard_timestamp_summary.html",
                {
                    "request": request,
                    "draft_id": draft_id,
                    "args": args,
                    "output_path": output_path,
                    **_wizard_flow_context(data, step=5),
                    "back_url": f"/wizard/{draft_id}/select",
                },
            )
        run_groups = _build_sync_run_groups(data)
        return templates.TemplateResponse(
            "wizard_summary.html",
            {
                "request": request,
                "draft_id": draft_id,
                "args": args,
                "run_groups": run_groups,
                **_wizard_flow_context(data, step=5),
                "back_url": f"/wizard/{draft_id}/select",
            },
        )

    @app.post("/wizard/{draft_id}/start")
    def wizard_start(draft_id: int) -> RedirectResponse:
        data = _draft_get(draft_id)
        data["run_error"] = None
        mode = str(data.get("mode") or "sync")

        if not bool(data.get("audio_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/audio", status_code=303)
        if not bool(data.get("video_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/video", status_code=303)
        if not bool(data.get("out_ok", False)):
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)
        if not str(data.get("out_dir", "")).strip():
            data["run_error"] = "Output dir is required."
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/output", status_code=303)

        if mode == "audio_timestamp":
            existing_run_id = data.get("timestamp_run_id")
            if isinstance(existing_run_id, int):
                return RedirectResponse(
                    url=f"/audio-timestamp/runs/{existing_run_id}", status_code=303
                )
            output_path = _resolve_timestamp_output_path(data)
            args: Dict[str, Any] = {
                "audio_dir": data.get("audio_dir", ""),
                "video_dir": data.get("video_dir", ""),
                "out_dir": data.get("out_dir", ""),
                "site": data.get("site", "nbu_lounge"),
                "timezone": DEFAULT_TZ.key,
                "output_json": output_path,
            }
            run_id = _create_timestamp_run(args, output_path=output_path or None)
            data["timestamp_run_id"] = run_id
            _draft_set(draft_id, data)
            _start_audio_timestamp_job(draft_id, run_id=run_id)
            return RedirectResponse(
                url=f"/audio-timestamp/runs/{run_id}", status_code=303
            )

        base_args: Dict[str, Any] = {
            "audio_dir": data.get("audio_dir", ""),
            "video_dir": data.get("video_dir", ""),
            "out_dir": data.get("out_dir", ""),
            "site": data.get("site", "nbu_lounge"),
            "segments": data.get("segments", []),
            "cameras": data.get("cameras", []),
            "log_level": data.get("log_level", "INFO"),
            "skip_decode": bool(data.get("skip_decode", False)),
            "overwrite_clips": bool(data.get("overwrite_clips", False)),
            "split": bool(data.get("split", False)),
            "split_overwrite": bool(data.get("split_overwrite", False)),
            "split_clean": bool(data.get("split_clean", False)),
            "split_chunk_seconds": int(data.get("split_chunk_seconds", 3600)),
        }
        run_groups = _build_sync_run_groups(data)
        if not run_groups:
            data["run_error"] = "No valid target pairs to run."
            _draft_set(draft_id, data)
            return RedirectResponse(url=f"/wizard/{draft_id}/select", status_code=303)

        schedule_at = str(data.get("schedule_at", "") or "").strip()
        now = utc_now_iso()
        scheduled_iso: Optional[str] = None
        status = "queued"
        if schedule_at:
            try:
                dt = datetime.fromisoformat(schedule_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
                scheduled_iso = dt.astimezone(timezone.utc).isoformat(
                    timespec="seconds"
                )
                status = "scheduled"
            except Exception as e:
                data["run_error"] = f"Bad schedule_at: {e}"
                _draft_set(draft_id, data)
                return RedirectResponse(
                    url=f"/wizard/{draft_id}/select", status_code=303
                )

        logs_dir = Path(".webui") / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        conn = get_conn()
        created_run_ids: List[int] = []
        with tx(conn):
            total_groups = len(run_groups)
            for idx, group in enumerate(run_groups, start=1):
                args = dict(base_args)
                args["target_pairs"] = list(group.get("target_pairs") or [])
                if group.get("mode") == "time":
                    args["time_start"] = str(group.get("time_start") or "")
                    args["time_end"] = str(group.get("time_end") or "")
                    args["time_zone"] = str(
                        group.get("time_zone") or WEBUI_DEFAULT_TIME_ZONE
                    )
                elif group.get("mode") == "sample":
                    args["audio_sample_start"] = group.get("audio_sample_start")
                    args["audio_sample_end"] = group.get("audio_sample_end")

                cmd = build_cli_cmd(args)
                title = data.get("title") or "video-sync run"
                if total_groups > 1:
                    mode_label = str(group.get("mode") or "manual")
                    title = f"{title} ({mode_label} {idx}/{total_groups})"

                cur = conn.execute(
                    """
                    INSERT INTO runs(title, created_at, scheduled_at, status, cwd, cmd_json, args_json, log_path)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        title,
                        now,
                        scheduled_iso,
                        status,
                        str(Path.cwd()),
                        json.dumps(cmd),
                        json.dumps(args),
                        str(logs_dir / "pending.log"),
                    ),
                )
                run_id = int(cur.lastrowid)
                created_run_ids.append(run_id)
                log_path = logs_dir / f"run-{run_id}.log"
                conn.execute(
                    "UPDATE runs SET log_path=? WHERE id=?", (str(log_path), run_id)
                )

        _draft_delete(draft_id)
        if len(created_run_ids) == 1:
            return RedirectResponse(url=f"/runs/{created_run_ids[0]}", status_code=303)
        return RedirectResponse(url="/runs", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: int) -> HTMLResponse:
        conn = get_conn()
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            args = json.loads(row["args_json"])
        except Exception:
            args = {}
        return templates.TemplateResponse(
            "run_detail.html",
            {
                "request": request,
                "run": row,
                "args": args,
            },
        )

    @app.get("/audio-timestamp/runs/{run_id}", response_class=HTMLResponse)
    def audio_timestamp_run_detail(request: Request, run_id: int) -> HTMLResponse:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM timestamp_runs WHERE id=?", (run_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            args = json.loads(row["args_json"])
        except Exception:
            args = {}
        json_text: Optional[str] = None
        output_path = row["output_path"]
        if output_path:
            try:
                json_text = Path(output_path).read_text(encoding="utf-8")
            except Exception:
                json_text = None
        return templates.TemplateResponse(
            "audio_timestamp_run_detail.html",
            {
                "request": request,
                "run": row,
                "args": args,
                "json_text": json_text,
            },
        )

    @app.post("/audio-timestamp/runs/{run_id}/delete")
    def audio_timestamp_run_delete(run_id: int) -> RedirectResponse:
        conn = get_conn()
        row = conn.execute(
            "SELECT status FROM timestamp_runs WHERE id=?", (run_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        if str(row["status"]) == "running":
            raise HTTPException(status_code=400, detail="Cannot delete a running run")
        with tx(conn):
            conn.execute("DELETE FROM timestamp_runs WHERE id=?", (run_id,))
        return RedirectResponse(url="/audio-timestamp/runs", status_code=303)

    # (Summary is shown before starting; see /wizard/{draft_id}/summary.)

    @app.post("/runs/{run_id}/cancel")
    def run_cancel(run_id: int) -> RedirectResponse:
        ok = runner.request_cancel(run_id)
        if not ok:
            raise HTTPException(status_code=400, detail="Cannot cancel this run")
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.post("/runs/{run_id}/delete")
    def run_delete(run_id: int) -> RedirectResponse:
        """
        Remove a run from history (DB row) and delete its log file (best-effort).
        """
        conn = get_conn()
        row = conn.execute(
            "SELECT status, log_path FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        if str(row["status"]) == "running":
            raise HTTPException(status_code=400, detail="Cannot delete a running run")

        try:
            p = Path(row["log_path"])
            if p.exists():
                p.unlink()
        except Exception:
            pass

        with tx(conn):
            conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return RedirectResponse(url="/runs", status_code=303)

    @app.get("/runs/{run_id}/logs")
    def run_logs(run_id: int) -> StreamingResponse:
        conn = get_conn()
        row = conn.execute("SELECT log_path FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        path = Path(row["log_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Log not found")

        def _iter() -> Any:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 64)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(_iter(), media_type="text/plain")

    @app.get("/runs/{run_id}/logs/stream")
    async def run_logs_stream(run_id: int) -> StreamingResponse:
        conn = get_conn()
        row = conn.execute("SELECT log_path FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        path = Path(row["log_path"])
        return StreamingResponse(
            tail_log_sse(path),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: int) -> Dict[str, Any]:
        conn = get_conn()
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        d = dict(row)
        # don't return huge blobs
        return d

    @app.get("/api/audio-timestamp/runs/{run_id}")
    def api_audio_timestamp_run(run_id: int) -> Dict[str, Any]:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM timestamp_runs WHERE id=?", (run_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "records_count": row["records_count"],
            "error": row["error"],
        }

    return app


app = create_app()
