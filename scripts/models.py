"""
Define abstract dataclasses

For each recording session, we expect to have 1 group of audios and multiple groups of videos.

- Audio
    each audio group should come with 3 audio files in the format of
        - <PREFIX>-<INDEX>.<EXT>
        e.g. Test-01.wav
    where <INDEX> should be 01, 02, or 03.
    audio that ends with 03 is the one that has serial encodings

- Video
    comes in k-min groups. Each group has several mp4 videos in the format of
        - <PREFIX>_<TIMESTAMP>.<CAM_SERIAL>.mp4
        e.g. Test_20250101_120000.23512099.mp4
    each group also comes with 1 json in the format of
        - <PREFIX>_<TIMESTAMP>.json
"""

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import pandas as pd


@dataclass(frozen=True)
class CamJson:
    """Represents an abstract JSON object for a camera.

    Attributes:
    ----------
    cam_serial: The serial number of this camera e.g. 23512099
    timestamp: The timestamp extracted from the JSON file
    path: The file path to this camera's JSON file.
    start_realtime: The start real time (UTC) from json.
    real_times: List of all real times (UTC) from json
    raw_serials: List of original chunk serial data
    raw_frame_ids: List of original frame ids
    fixed_serials: List of fixed chunk serial data
    fixed_frame_ids: List of wrapped frame ids
    fixed_reidx_frame_ids: List of reindexed, fixed frame ids
    """

    cam_serial: Optional[str]
    timestamp: Optional[datetime]
    path: Optional[Path]
    start_realtime: Optional[datetime]
    real_times: Optional[List[datetime]]
    raw_serials: Optional[List[int]]
    raw_frame_ids: Optional[List[int]]
    fixed_serials: Optional[List[int]]
    fixed_frame_ids: Optional[List[int]]
    fixed_reidx_frame_ids: Optional[List[int]]


@dataclass(frozen=True)
class Json:
    """An abstract Json object.

    cam_serials: list of camera serial numbers
    timestamp: the timestamp extracted from file name
    path: path to the json file
    cam_jsons: mapping camera serial -> CamJson
    """

    cam_serials: Optional[List[str]]
    timestamp: Optional[datetime]
    path: Path
    cam_jsons: Dict[str, CamJson]


@dataclass(frozen=True)
class Audio:
    """An abstract generic Audio object for a single audio file.

    Attributes
    ----------
    path: Path to this audio file
    duration: length in seconds
    file_size: file size in MB
    sample_rate: the sample rate of the audio file in Hz
    extension: the file extension (e.g. "wav" or "mp3")
    channel: which audio channel this file belongs to
    """

    path: Optional[Path]
    duration: float
    file_size: float
    sample_rate: int
    extension: str
    channel: int


@dataclass(frozen=True)
class SerialAudio(Audio):
    """A SerialAudio object for audio files with serial encodings."""

    pass


@dataclass(frozen=True)
class AudioGroup:
    """Audio files discovered under the audio directory.

    Attributes
    ----------
    audios: mapping channel -> Audio. e.g. 1 -> Audio(path=..., duration=..., ...)
    serial_audio: Optional[SerialAudio]
    shared_extension: Optional[str] whether the group is all wav or mp3
    """

    audios: Dict[int, Audio]
    serial_audio: Optional[SerialAudio]
    shared_extension: Optional[str]


@dataclass(frozen=True)
class Video:
    """An abstract representation of a mp4 video.

    Attributes
    ----------
    path: absolute path to this mp4 video file.
    segment_id: derived from the file name, e.g. "TRBD001_20250101_120000"
    cam_serial: derived from the file name
    timestamp: timestamp extracted from file name
    start_realtime: the time of the first frame
    duration: length in seconds
    resolution: video resolution (e.g. "1920x1080")
    frame_rate: frames per second (e.g. 30.0)
    frame_count: total number of frames
    companion_json: the CamJson associated with this video, could be missing
    """

    path: Path
    segment_id: str
    cam_serial: str
    timestamp: datetime
    start_realtime: datetime
    duration: float
    resolution: str
    frame_rate: float
    frame_count: int
    companion_json: CamJson


@dataclass(frozen=True)
class VideoGroup:
    """A chunked n-minute recording segment defined by its JSON.

    Attributes
    ----------
    group_id: The shared BASE identifier, e.g. "Test_20250101_120000".
    timestamp: timezone-aware datetime parsed from group_id
    json: a Json object
    videos: list of Video objects in the group
    cam_serials: list of cam serials in this group
    """

    group_id: str
    timestamp: Optional[datetime]
    json: Json
    videos: Optional[List[Video]]
    cam_serials: Optional[List[str]]


@dataclass(frozen=True)
class AudioVideoSession:
    """A session contains one audio group and multiple video groups.

    Attributes
    ----------
    audiogroup: The audio group associated with this session.
    videogroups: The video groups associated with this session.
    shared_cam_serials: The camera serials shared across video groups.

    """

    audiogroup: AudioGroup
    videogroups: List[VideoGroup]
    shared_cam_serials: Optional[List[str]]


## =================================================== ##
## ===== The following applies to EMU recordings ===== ##
## =================================================== ##


@dataclass(frozen=True)
class DIGIEVTS:
    """A digital events.

    Attributes
    ----------
    raw: List[Dict[str, Any]], the raw digital events data from Nev. It has 3 keys:
        - TimeStamps: List[int], the timestamps in ticks starting from timeOrigin
        - InsertionReason: List[int], the insertion reasons (1 or 129)
        - UnparsedData: List[int], the unparsed data
    raw_df: pd.DataFrame, the raw digital events data in a pandas DataFrame
    chunk_serial_df: Optional[pd.DataFrame], the chunk serial DataFrame
    start_serial: Optional[int], the starting chunk serial number
    end_serial: Optional[int], the ending chunk serial number
    start_timestamp: Optional[int], the starting timestamp in ticks
    end_timestamp: Optional[int], the ending timestamp in ticks
    """

    raw: Optional[List[Dict[str, Any]]]
    raw_df: Optional[pd.DataFrame]
    chunk_serial_df: Optional[pd.DataFrame]
    start_serial: Optional[int]
    end_serial: Optional[int]
    start_timestamp: Optional[int]
    end_timestamp: Optional[int]


@dataclass(frozen=True)
class NEV:
    """A NEV file.

    Attributes
    ----------
    path: Path to this NEV file
    start_utc_time: the start UTC timestamp of the recording, same as timeOrigin in the NEV file header
    sample_resolution: the time resolution of the NEV file in microseconds, same as SampleTimeResolution
    duration: the duration of the recording in seconds
    digital_events: DIGIEVTS
    """

    path: Path
    start_utc_time: datetime
    sample_resolution: float
    duration: float
    digital_events: DIGIEVTS


@dataclass(frozen=True)
class RoomAudio:
    """An abstract representation of room audio.

    Attributes
    ----------
    raw_array: Optional[float]
    start_timestamp: Optional[int], the start timestamp in ticks
    end_timestamp: Optional[int], the end timestamp in ticks
    duration: Optional[float], the duration in seconds
    """

    raw_array: Optional[float]
    start_timestamp: Optional[int]
    end_timestamp: Optional[int]
    duration: Optional[float]


@dataclass(frozen=True)
class StitchedNS5:
    """An Stitched NS5 file.

    Attributes
    ----------
    path: Path to this NS5 file
    start_utc_time: the start UTC timestamp of the recording, same as timeOrigin in the NS5 file header
    sample_resolution: the time resolution of the NS5 file in microseconds, same as SampleTimeResolution
    duration: the duration of the recording in seconds
    room_mic1: audio recorded in RoomMic1
    room_mic2: audio recorded in RoomMic2
    """

    path: Path
    start_utc_time: datetime
    sample_resolution: float
    duration: float
    room_mic1: RoomAudio
    room_mic2: RoomAudio
