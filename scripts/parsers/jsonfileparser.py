import json
from datetime import datetime
from scripts.fix.jsonserialfixer import JsonSerialFixer
from scripts.fix.frameidfixer import FrameIDFixer
import argparse
from pathlib import Path


class JsonParser:
    """
    Wrapper of video json file
    """

    def __init__(self, json_path) -> None:
        self.json_path = json_path
        with open(json_path, "r", encoding="utf-8") as f:
            self.dic = json.load(f)
        self.init_vars()

    def init_vars(self):
        self.num_cameras = self.get_num_cameras()
        self.length_of_recording = self.get_length_of_recording()
        self.timeOrigin = self.dic["real_times"][0] if self.dic["real_times"] else None
        self.duration_readable = (
            self.calculate_duration(self.dic["real_times"])
            if self.dic["real_times"]
            else None
        )

    def calculate_duration(self, real_times) -> str:
        start_time = datetime.strptime(real_times[0], "%Y-%m-%d %H:%M:%S.%f")
        end_time = datetime.strptime(real_times[-1], "%Y-%m-%d %H:%M:%S.%f")
        return str(end_time - start_time)

    def get_duration_readable(self):
        return self.duration_readable

    def get_time_origin(self):
        return self.timeOrigin

    def get_num_cameras(self):
        return len(self.dic["serials"])

    def get_length_of_recording(self):
        return len(self.dic["timestamps"])

    def get_camera_serials(self) -> list:
        return list(self.dic["serials"])

    def get_start_realtime(self) -> datetime:
        """Return the start real time (UTC)"""
        if not self.dic["real_times"]:
            return None
        return datetime.strptime(self.dic["real_times"][0], "%Y-%m-%d %H:%M:%S.%f")

    def get_end_realtime(self) -> datetime:
        """Return the end real time (UTC)."""
        if not self.dic["real_times"]:
            return None
        return datetime.strptime(self.dic["real_times"][-1], "%Y-%m-%d %H:%M:%S.%f")

    def get_chunk_serial_list(self, cam_serial):
        """Return the list of chunk serial"""
        assert (
            cam_serial in self.get_camera_serials()
        ), "Camera serial not found in JSON"
        cam_idx = self.get_camera_serials().index(cam_serial)
        res = []
        for chunk_serial in self.dic["chunk_serial_data"]:
            res.append(chunk_serial[cam_idx])
        return res

    def get_fixed_chunk_serial_list(self, cam_serial):
        """Return the fixed chunk serial"""
        serial = self.get_chunk_serial_list(cam_serial)
        fixer = JsonSerialFixer()
        fixed = fixer.fix(serial)
        return fixed

    def get_frame_ids_list(self, cam_serial):
        assert (
            cam_serial in self.get_camera_serials()
        ), "Camera serial not found in JSON"
        cam_idx = self.get_camera_serials().index(cam_serial)
        res = []
        for frame_ids in self.dic["frame_id"]:
            res.append(frame_ids[cam_idx])
        return res

    def get_fixed_frame_ids_list(self, cam_serial):
        frameids = self.get_frame_ids_list(cam_serial)
        gap_fixer = JsonSerialFixer()
        gap_fixed = gap_fixer.fix(frameids)
        wrap_fixer = FrameIDFixer()
        wrap_fixed = wrap_fixer.fix(gap_fixed)
        return wrap_fixed

    def get_fixed_reindexed_frame_ids_list(self, cam_serial):
        """Return fixed frame_ids reindexed to start at 0 (contiguous).

        Reindex rule: subtract the first fixed frame id from each element.
        If the list is empty, return an empty list.
        """
        fixed = self.get_fixed_frame_ids_list(cam_serial)
        if not fixed:
            return []
        base = int(fixed[0])
        return [int(fid) - base for fid in fixed]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="json-extract-cam",
        description="Extract per-camera lists (raw/fixed frame_ids & chunk_serials) from a recording JSON.",
    )
    p.add_argument("--json", required=True, help="Path to the segment JSON.")
    p.add_argument("--camera-serial", required=True, help="Camera serial to extract.")
    p.add_argument(
        "--out-dir", required=True, help="Directory to write the output JSON."
    )
    return p


def _derive_out_path(in_json: Path, cam_serial: str, out_dir: Path) -> Path:
    # same name as original but with suffix .<camera_serial>.json
    return out_dir / f"{in_json.stem}.{cam_serial}.json"


def _main(argv=None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    in_path = Path(args.json)
    out_dir = Path(args.out_dir)
    cam_serial = str(args.camera_serial)

    if not in_path.is_file():
        print(f"[ERROR] JSON not found: {in_path}")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _derive_out_path(in_path, cam_serial, out_dir)

    try:
        jp = JsonParser(str(in_path))
        # fetch lists
        frame_ids = jp.get_frame_ids_list(cam_serial)
        chunk_serials = jp.get_chunk_serial_list(cam_serial)
        fixed_frame_ids = jp.get_fixed_frame_ids_list(cam_serial)
        fixed_chunk_serials = jp.get_fixed_chunk_serial_list(cam_serial)
        fixed_frameids_reindexed = jp.get_fixed_reindexed_frame_ids_list(cam_serial)
    except AssertionError as e:
        print(f"[ERROR] {e}")
        return 2
    except Exception as e:
        print(f"[ERROR] Failed to parse/extract: {e}")
        return 2

    payload = {
        "camera_serial": cam_serial,
        "frame_ids": frame_ids,
        "chunk_serials": chunk_serials,
        "fixed_frame_ids": fixed_frame_ids,
        "fixed_chunk_serials": fixed_chunk_serials,
        "real_times": jp.dic.get("real_times", []),
        "fixed_frameids_reindexed": fixed_frameids_reindexed,
    }

    try:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Could not write output: {e}")
        return 2

    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
