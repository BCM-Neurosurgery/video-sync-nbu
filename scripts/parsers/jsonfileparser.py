import json
from datetime import datetime
from typing import List
from scripts.fix.jsonserialfixer import JsonSerialFixer
from scripts.fix.frameidfixer import FrameIDFixer


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
        fixer = FrameIDFixer()
        fixed = fixer.fix(frameids)
        return fixed
