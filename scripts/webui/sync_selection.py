from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from scripts.parsers.jsonfileparser import JsonParser


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


def _parse_time_text_in_utc(raw: object, source_tz: str = "UTC") -> Optional[datetime]:
    """
    Parse wall-clock text and convert to naive UTC datetime for internal comparison.
    """
    dt = _parse_time_text(raw)
    if dt is None:
        return None
    tz_name = str(source_tz or "UTC").strip() or "UTC"
    if tz_name == "UTC":
        return dt
    try:
        tzinfo = ZoneInfo(tz_name)
    except Exception:
        return None
    return dt.replace(tzinfo=tzinfo).astimezone(timezone.utc).replace(tzinfo=None)


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
            jp = JsonParser(str(json_path))
            payload = jp.dic
        except Exception:
            continue

        real_times = payload.get("real_times") or []
        segment_time_start = _normalize_time_text(real_times[0]) if real_times else ""
        segment_time_end = _normalize_time_text(real_times[-1]) if real_times else ""

        serials_raw = payload.get("serials") or []
        for cam_serial_raw in serials_raw:
            cam = str(cam_serial_raw).strip()
            key = f"{seg_id}::{cam}"
            if key not in available_pair_map:
                continue

            pair_info: Dict[str, Any] = {
                "time_start": "" if sample_ready else segment_time_start,
                "time_end": "" if sample_ready else segment_time_end,
                "sample_start": None,
                "sample_end": None,
            }

            if sample_ready:
                sample_min: Optional[int] = None
                sample_max: Optional[int] = None
                matched_start_idx: Optional[int] = None
                matched_end_idx: Optional[int] = None
                try:
                    fixed_serials = jp.get_fixed_chunk_serial_list(cam_serial_raw)
                except Exception:
                    fixed_serials = []

                for frame_idx, serial_val_raw in enumerate(fixed_serials):
                    if frame_idx >= len(real_times):
                        continue
                    try:
                        serial_val = int(serial_val_raw)
                    except Exception:
                        continue
                    rng = serial_sample_map.get(serial_val)
                    if rng is None:
                        continue

                    s0, s1 = rng
                    sample_min = s0 if sample_min is None else min(sample_min, s0)
                    sample_max = s1 if sample_max is None else max(sample_max, s1)
                    if matched_start_idx is None:
                        matched_start_idx = frame_idx
                    matched_end_idx = frame_idx

                if sample_min is not None and sample_max is not None:
                    pair_info["sample_start"] = int(sample_min)
                    pair_info["sample_end"] = int(sample_max)
                if matched_start_idx is not None and matched_end_idx is not None:
                    pair_info["time_start"] = _normalize_time_text(
                        real_times[matched_start_idx]
                    )
                    pair_info["time_end"] = _normalize_time_text(
                        real_times[matched_end_idx]
                    )

            pairs[key] = pair_info
            pairs_by_camera.setdefault(cam, []).append(key)

            t0 = pair_info.get("time_start")
            t1 = pair_info.get("time_end")
            if isinstance(t0, str) and isinstance(t1, str) and t0 and t1:
                cur = camera_time_bounds.get(cam)
                if cur is None:
                    camera_time_bounds[cam] = {"start": t0, "end": t1}
                else:
                    if t0 < cur["start"]:
                        cur["start"] = t0
                    if t1 > cur["end"]:
                        cur["end"] = t1

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


def _build_range_config_from_form(
    *,
    selection_mode: str,
    cameras: List[str],
    form: Mapping[str, Any],
    prev_range_config: object,
) -> Dict[str, Any]:
    prev_cfg = _coerce_range_config(prev_range_config, cameras)
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
            ts_raw = form.get(ts_key) if ts_key in form else prev_rule.get("time_start")
            te_raw = form.get(te_key) if te_key in form else prev_rule.get("time_end")
            tz_raw = form.get(tz_key) if tz_key in form else prev_rule.get("time_zone")
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
                form.get(ss_key) if ss_key in form else prev_rule.get("sample_start")
            )
            se_raw = form.get(se_key) if se_key in form else prev_rule.get("sample_end")
            s0 = _parse_int_or_none(ss_raw)
            s1 = _parse_int_or_none(se_raw)
            rule["sample_start"] = "" if s0 is None else str(s0)
            rule["sample_end"] = "" if s1 is None else str(s1)

    return range_config


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
        rule_tz = str(rule.get("time_zone") or WEBUI_DEFAULT_TIME_ZONE).strip() or "UTC"
        t0 = _parse_time_text_in_utc(rule.get("time_start"), rule_tz)
        t1 = _parse_time_text_in_utc(rule.get("time_end"), rule_tz)
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
