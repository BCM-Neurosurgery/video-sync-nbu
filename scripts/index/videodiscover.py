from pathlib import Path
import logging
from dataclasses import replace
from typing import Dict, Optional, Tuple, List
from datetime import datetime


from scripts.index.filepatterns import FilePatterns
from scripts.index.common import (
    _DirMixin,
    _safe_glob,
    DEFAULT_TZ,
)
from scripts.parsers.videofileparser import VideoFileParser
from scripts.parsers.jsonfileparser import JsonParser
from scripts.fix.jsonserialfixer import JsonSerialFixer
from scripts.models import (
    Video,
    CamJson,
    Json,
    VideoGroup,
)


class VideoDiscoverer(_DirMixin):
    """Discovers JSON-defined segments and attaches MP4s by camera serial."""

    def __init__(self, video_dir: Path, *, log: logging.Logger):
        self.video_dir = video_dir
        self.log = log

    # ---- scanning helpers --------------------------------------------------

    def _index_jsons(self) -> Dict[str, Path]:
        self._ensure_exists(self.video_dir)
        json_by_seg: Dict[str, Path] = {}
        for jp in _safe_glob(self.video_dir, ("*.json",)):
            seg_id = FilePatterns.parse_json_filename(jp)
            if not seg_id:
                self.log.warning("Skipping JSON with unexpected name: %s", jp.name)
                continue
            if seg_id in json_by_seg:
                self.log.warning(
                    "Duplicate JSON for %s; keeping first: %s (ignoring %s)",
                    seg_id,
                    json_by_seg[seg_id].name,
                    jp.name,
                )
                continue
            json_by_seg[seg_id] = jp
        if not json_by_seg:
            self.log.warning(
                "No valid segment JSON files found under %s", self.video_dir
            )
        return json_by_seg

    def _index_mp4s(self) -> Dict[str, Dict[str, Path]]:
        vids_by_seg: Dict[str, Dict[str, Path]] = {}
        for vp in _safe_glob(self.video_dir, ("*.mp4",)):
            parsed = FilePatterns.parse_video_filename(vp)
            if not parsed:
                self.log.warning("Skipping MP4 with unexpected name: %s", vp.name)
                continue
            seg_id, cam_serial = parsed
            vids_by_seg.setdefault(seg_id, {})
            if cam_serial in vids_by_seg[seg_id]:
                self.log.warning(
                    "Duplicate MP4 for %s cam %s; keeping first: %s (ignoring %s)",
                    seg_id,
                    cam_serial,
                    vids_by_seg[seg_id][cam_serial].name,
                    vp.name,
                )
                continue
            vids_by_seg[seg_id][cam_serial] = vp
        return vids_by_seg

    # ---- metadata helpers --------------------------------------------------

    def _extract_video_meta(self, mp4_path: Path) -> tuple[float, str, float, int]:
        """Return (duration_sec, 'WxH', fps, frame_count). On failure, return zeros/empty."""
        try:
            vp = VideoFileParser(str(mp4_path))
            w, h = vp.resolution
            return vp.duration, f"{w}x{h}", vp.fps, vp.frame_count
        except Exception as e:
            self.log.warning(
                "ffprobe failed for %s: %s; leaving meta blank.", mp4_path.name, e
            )
            return 0.0, "", 0.0, 0

    def _build_videos_for_seg(
        self,
        seg_id: str,
        cams: Dict[str, Path],
        ts: Optional[datetime],
        cam_jsons: Optional[Dict[str, CamJson]],
    ) -> List[Video]:
        videos: List[Video] = []
        for cam_serial, mp4_path in sorted(cams.items(), key=lambda kv: kv[0]):
            dur, res, fps, frame_count = self._extract_video_meta(mp4_path)
            cam_key = str(cam_serial)
            companion = cam_jsons.get(cam_key) if cam_jsons else None
            start_rt: Optional[datetime] = None
            if companion and companion.start_realtime:
                start_rt = companion.start_realtime
            elif companion and companion.timestamp:
                start_rt = companion.timestamp
            elif ts:
                start_rt = ts
            else:
                start_rt = datetime.min.replace(tzinfo=DEFAULT_TZ)
            if companion is None:
                companion = CamJson(
                    cam_serial=cam_key,
                    timestamp=ts,
                    path=None,
                    start_realtime=start_rt,
                    raw_serials=None,
                    raw_frame_ids=None,
                    fixed_serials=None,
                    fixed_frame_ids=None,
                    fixed_reidx_frame_ids=None,
                )
            videos.append(
                Video(
                    path=mp4_path,
                    segment_id=seg_id,
                    cam_serial=cam_key,
                    timestamp=ts,
                    start_realtime=start_rt,
                    duration=dur,
                    resolution=res,
                    frame_rate=fps,
                    frame_count=frame_count,
                    companion_json=companion,
                )
            )
        return videos

    def _extract_cam_jsons(
        self, json_path: Path, ts: Optional[datetime]
    ) -> tuple[list[str], dict[str, CamJson]]:
        """
        Use JsonParser to populate per-camera CamJson objects.

        Returns: (cam_serials_as_strings_in_json, cam_jsons_map_by_serial)
        """
        cam_jsons: dict[str, CamJson] = {}
        try:
            jp = JsonParser(str(json_path))
            parser_serials = jp.get_camera_serials()  # e.g., [24253445, ...]
            cam_serials_all = [str(s) for s in parser_serials]

            for s in parser_serials:
                s_str = str(s)
                try:
                    start_realtime = jp.get_start_realtime()
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_start_realtime(): %s", json_path.name, e
                    )
                    start_realtime = None

                try:
                    raw_serials = jp.get_chunk_serial_list(s)
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_chunk_serial_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    raw_serials = None

                try:
                    raw_frame_ids = jp.get_frame_ids_list(s)
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_frame_ids_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    raw_frame_ids = None

                try:
                    fixed_serials = jp.get_fixed_chunk_serial_list(s)
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_fixed_chunk_serial_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    fixed_serials = None

                try:
                    fixed_frame_ids = jp.get_fixed_frame_ids_list(s)
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_fixed_frame_ids_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    fixed_frame_ids = None

                try:
                    fixed_reidx_frame_ids = (
                        [f - fixed_frame_ids[0] for f in fixed_frame_ids]
                        if fixed_frame_ids
                        else None
                    )
                except Exception as e:
                    self.log.warning(
                        "JSON %s: failed get_fixed_reindexed_frame_ids_list(%s): %s",
                        json_path.name,
                        s_str,
                        e,
                    )
                    fixed_reidx_frame_ids = None

                cam_jsons[s_str] = CamJson(
                    cam_serial=s_str,
                    timestamp=ts,
                    path=json_path,
                    start_realtime=start_realtime,
                    real_times=jp.dic.get("real_times"),
                    raw_serials=raw_serials,
                    raw_frame_ids=raw_frame_ids,
                    fixed_serials=fixed_serials,
                    fixed_frame_ids=fixed_frame_ids,
                    fixed_reidx_frame_ids=fixed_reidx_frame_ids,
                )

            cam_jsons = self._apply_raw_serial_fallbacks(
                cam_jsons,
                json_name=json_path.name,
            )

            return cam_serials_all, cam_jsons

        except Exception as e:
            self.log.warning("Failed to parse JSON %s: %s", json_path.name, e)
            return [], {}

    def _build_json_wrapper(
        self, json_path: Path, ts: Optional[datetime]
    ) -> Tuple[Json, List[str], Dict[str, CamJson]]:
        cam_serials_from_json, cam_jsons = self._extract_cam_jsons(json_path, ts)
        json_wrap = Json(
            cam_serials=cam_serials_from_json or None,
            timestamp=ts,
            path=json_path,
            cam_jsons=cam_jsons,
        )
        return json_wrap, cam_serials_from_json, cam_jsons

    # ---- data cleanup helpers ---------------------------------------------

    def _apply_raw_serial_fallbacks(
        self,
        cam_jsons: Dict[str, CamJson],
        *,
        json_name: str,
    ) -> Dict[str, CamJson]:
        """Replace all-(-1) raw serial lists with another camera's data."""
        if not cam_jsons:
            return cam_jsons

        exemplar_by_length: dict[int, tuple[list[int], Optional[list[int]]]] = {}
        for cam_json in cam_jsons.values():
            raw = cam_json.raw_serials
            if raw and not self._raw_serials_all_placeholder(raw):
                fixed_copy = (
                    list(cam_json.fixed_serials) if cam_json.fixed_serials else None
                )
                exemplar_by_length.setdefault(
                    len(raw),
                    (list(raw), fixed_copy or self._compute_fixed_serials(raw)),
                )

        if not exemplar_by_length:
            return cam_jsons

        updated = dict(cam_jsons)
        for cam_serial, cam_json in cam_jsons.items():
            raw = cam_json.raw_serials
            if not raw or not self._raw_serials_all_placeholder(raw):
                continue

            replacement = exemplar_by_length.get(len(raw))
            if not replacement:
                continue

            fallback_raw, fallback_fixed = replacement
            if fallback_fixed is None:
                fallback_fixed = self._compute_fixed_serials(fallback_raw)

            self.log.warning(
                "JSON %s camera %s raw_serials all -1; using fallback from another camera",
                json_name,
                cam_serial,
            )
            updated[cam_serial] = replace(
                cam_json,
                raw_serials=list(fallback_raw),
                fixed_serials=list(fallback_fixed) if fallback_fixed else None,
            )

        return updated

    def _compute_fixed_serials(self, raw: List[int]) -> Optional[list[int]]:
        """Run JsonSerialFixer on the provided raw serials."""
        if not raw:
            return None
        try:
            fixed = JsonSerialFixer().fix(raw)
        except Exception as exc:
            self.log.warning(
                "Failed to compute fixed serials for fallback raw_serials (len=%s): %s",
                len(raw),
                exc,
            )
            return None
        return list(fixed) if fixed else None

    @staticmethod
    def _raw_serials_all_placeholder(raw: List[int]) -> bool:
        """Return True if every entry is effectively missing (-1/None)."""
        if not raw:
            return False
        for value in raw:
            if value is None:
                continue
            try:
                if int(value) != -1:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    # ---- public: fast single-segment path (no cam filter) -----------------

    def discover_one(self, segment_id: str) -> Optional[VideoGroup]:
        """
        Build a single VideoGroup for a given segment_id, including all cameras.
        Avoids a full directory scan.
        """
        self._ensure_exists(self.video_dir)

        # JSON for this segment
        json_matches = list(self.video_dir.glob(f"{segment_id}.json"))
        if not json_matches:
            self.log.warning("No JSON found for segment %s", segment_id)
            return None
        json_path = json_matches[0]

        ts = FilePatterns.parse_tail_datetime(segment_id, DEFAULT_TZ)

        # All MP4s for this segment
        cams: Dict[str, Path] = {}
        for vp in self.video_dir.glob(f"{segment_id}.*.mp4"):
            parsed = FilePatterns.parse_video_filename(vp)
            if parsed:
                _, cam = parsed
                if cam not in cams:  # keep first if duplicates
                    cams[cam] = vp

        json_wrap, _, cam_jsons = self._build_json_wrapper(json_path, ts)

        videos = self._build_videos_for_seg(segment_id, cams, ts, cam_jsons)

        return VideoGroup(
            group_id=segment_id,
            timestamp=ts,
            json=json_wrap,
            videos=videos or None,
            cam_serials=(sorted({v.cam_serial for v in videos}) if videos else None),
        )

    def discover_video(self, segment_id: str, cam_serial: str) -> Optional[Video]:
        """
        Return a single :class:`Video` for the given segment and camera.

        Parameters
        ----------
        segment_id : str
            Segment ID like 'TRBD002_20250806_104707'.
        cam_serial : str
            Camera serial (e.g., '23512909').

        Returns
        -------
        Optional[Video]
            Populated Video with ffprobe metadata and, when available, the matching
            `companion_json` (CamJson) from the segment's JSON.
        """
        self._ensure_exists(self.video_dir)
        ts = FilePatterns.parse_tail_datetime(segment_id, DEFAULT_TZ)

        # Resolve MP4 for (segment_id, cam_serial)
        matches = sorted(self.video_dir.glob(f"{segment_id}.{cam_serial}.mp4"))
        if not matches:
            self.log.warning(
                "No MP4 found for segment %s cam %s", segment_id, cam_serial
            )
            return None
        mp4_path = matches[0]

        # Extract basic video meta
        dur, res, fps, frame_count = self._extract_video_meta(mp4_path)

        # Attach companion CamJson if present
        companion: Optional[CamJson] = None
        json_matches = list(self.video_dir.glob(f"{segment_id}.json"))
        if json_matches:
            json_path = json_matches[0]
            _, _, cam_jsons = self._build_json_wrapper(json_path, ts)
            companion = cam_jsons.get(str(cam_serial))
            if companion is None:
                self.log.warning(
                    "Camera %s not present in JSON %s; CamJson will be None.",
                    cam_serial,
                    json_path.name,
                )
        else:
            self.log.info(
                "No JSON found for %s; companion_json will be None.", segment_id
            )

        return Video(
            path=mp4_path,
            segment_id=segment_id,
            cam_serial=str(cam_serial),
            timestamp=ts,
            start_realtime=companion.start_realtime if companion else None,
            duration=dur,
            resolution=res,
            frame_rate=fps,
            frame_count=frame_count,
            companion_json=companion,
        )

    # ---- public: full directory path --------------------------------------

    def discover(self) -> List[VideoGroup]:
        json_by_seg = self._index_jsons()
        vids_by_seg = self._index_mp4s()

        videogroups: List[VideoGroup] = []
        for seg_id, json_path in json_by_seg.items():
            ts = FilePatterns.parse_tail_datetime(seg_id, DEFAULT_TZ)
            cams = vids_by_seg.get(seg_id, {})

            json_wrap, _, cam_jsons = self._build_json_wrapper(json_path, ts)

            videos = self._build_videos_for_seg(seg_id, cams, ts, cam_jsons)
            if not videos:
                self.log.warning(
                    "No MP4s found for segment %s (JSON: %s)", seg_id, json_path.name
                )

            videogroups.append(
                VideoGroup(
                    group_id=seg_id,
                    timestamp=ts,
                    json=json_wrap,
                    videos=videos or None,
                    cam_serials=(
                        sorted({v.cam_serial for v in videos}) if videos else None
                    ),
                )
            )

        # Orphan MP4s (no JSON)
        orphans = sorted(set(vids_by_seg.keys()) - set(json_by_seg.keys()))
        for seg_id in orphans:
            self.log.warning(
                "Found MP4(s) for %s but no matching JSON; they will be ignored.",
                seg_id,
            )

        # Sort chronologically by tail
        videogroups.sort(key=lambda s: FilePatterns.videogroup_sort_key(s.group_id))
        self.log.info("Discovered %d segment(s).", len(videogroups))
        return videogroups


def build_video_obj(
    video_dir: Path,
    segment_id: str,
    cam_serial: str,
    log: Optional[logging.Logger] = None,
) -> Optional[Video]:
    """
    Convenience wrapper around VideoDiscoverer.discover_video.

    Parameters
    ----------
    video_dir : Path
        Directory containing the segment JSON/MP4 files.
    segment_id : str
        Segment ID like 'TRBD002_20250806_104707'.
    cam_serial : str
        Camera serial (e.g., '23512909').
    log : Optional[logging.Logger]
        Logger to use. If None, uses module logger.

    Returns
    -------
    Optional[Video]
        A single Video instance (or None if not found).
    """
    logger = log or logging.getLogger(__name__)
    return VideoDiscoverer(video_dir, log=logger).discover_video(segment_id, cam_serial)
