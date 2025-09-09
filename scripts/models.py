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
from typing import List, Dict, Optional


@dataclass(frozen=True)
class CamJson:
    """Represents an abstract JSON object for a camera.

    Attributes:
    ----------
    cam_serial: The serial number of this camera e.g. 23512099
    timestamp: The timestamp extracted from the JSON file
    path: The file path to this camera's JSON file.
    raw_serials: List of original chunk serial data
    raw_frame_ids: List of original frame ids
    fixed_serials: List of fixed chunk serial data
    fixed_frame_ids: List of wrapped frame ids
    fixed_reidx_frame_ids: List of reindexed, fixed frame ids
    """

    cam_serial: Optional[str]
    timestamp: Optional[datetime]
    path: Optional[Path]
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
    path: Path to this mp4 video file.
    cam_serial: derived from the file name
    timestamp: timestamp extracted from file name
    duration: length in seconds
    resolution: video resolution (e.g. "1920x1080")
    frame_rate: frames per second (e.g. 30.0)
    frame_count: total number of frames
    companion_json: the CamJson associated with this video, if any
    """

    path: Path
    cam_serial: Optional[str]
    timestamp: Optional[datetime]
    duration: float
    resolution: str
    frame_rate: float
    frame_count: int
    companion_json: Optional[CamJson]


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
