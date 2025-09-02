import re
from typing import Optional, Tuple
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime
from scripts.index.common import DEFAULT_TZ


class FilePatterns:
    """
    Centralized filename parsing utilities for the A/V sync workflow.

    Supported Basename Patterns
    ---------------------------
    Video MP4
        ``<BASE>.<CAM>.mp4``, where ``<BASE> = "<prefix>_YYYYMMDD_HHMMSS"`` and
        ``<CAM>`` is an alphanumeric camera identifier (often a serial).
        Example: ``TRBD002_20250806_104707.23512909.mp4``

    Segment JSON
        ``<BASE>.json``, sharing the same ``<BASE>`` as the matching videos.
        Example: ``TRBD002_20250806_104707.json``

    Audio (program/serial)
        ``<prefix>-<chan>.<ext>``, where ``<chan>`` is two digits and ``<ext>`` is
        ``wav`` or ``mp3``.
        Example: ``TRBD002_08062025-01.mp3``

    Notes
    -----
    - All patterns are matched against the basename (i.e., ``Path.name``).
    - Extensions are matched case-insensitively.
    """

    RE_TAIL = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})$")
    RE_VIDEO = re.compile(
        r"^(?P<base>.+?_\d{8}_\d{6})\.(?P<cam>[0-9A-Za-z]+)\.mp4$", re.IGNORECASE
    )
    RE_JSON = re.compile(r"^(?P<base>.+?_\d{8}_\d{6})\.json$", re.IGNORECASE)
    RE_AUDIO = re.compile(
        r"^(?P<prefix>.+)-(?P<chan>\d{2})\.(?P<ext>wav|mp3)$", re.IGNORECASE
    )

    @classmethod
    def parse_video_filename(cls, p: Path) -> Optional[Tuple[str, str]]:
        """
        Parse a video basename of the form ``<BASE>.<CAM>.mp4``.

        Parameters
        ----------
        p : pathlib.Path
            Path to the MP4 file (only the basename is inspected).

        Returns
        -------
        Optional[Tuple[str, str]]
            ``(base, cam)`` where ``base`` is ``"<prefix>_YYYYMMDD_HHMMSS"`` and
            ``cam`` is the camera identifier. Returns ``None`` if not matched.

        Examples
        --------
        >>> FilePatterns.parse_video_filename(Path("TRBD_20250101_010203.23512909.mp4"))
        ('TRBD_20250101_010203', '23512909')
        """
        m = cls.RE_VIDEO.match(p.name)
        return (m.group("base"), m.group("cam")) if m else None

    @classmethod
    def parse_json_filename(cls, p: Path) -> Optional[str]:
        """
        Parse a segment JSON basename of the form ``<BASE>.json``.

        Parameters
        ----------
        p : pathlib.Path
            Path to the JSON file (only the basename is inspected).

        Returns
        -------
        Optional[str]
            The ``base`` string ``"<prefix>_YYYYMMDD_HHMMSS"`` if matched;
            otherwise ``None``.

        Examples
        --------
        >>> FilePatterns.parse_json_filename(Path("TRBD_20250101_010203.json"))
        'TRBD_20250101_010203'
        """
        m = cls.RE_JSON.match(p.name)
        return m.group("base") if m else None

    @classmethod
    def parse_audio_filename(cls, p: Path) -> Optional[Tuple[int, str]]:
        """
        Parse an audio basename of the form ``<prefix>-<chan>.<ext>``.

        Parameters
        ----------
        p : pathlib.Path
            Path to an audio file whose channel is encoded as two digits.

        Returns
        -------
        Optional[Tuple[int, str]]
            ``(chan, ext)`` where ``chan`` is the integer channel number and
            ``ext`` is the lowercase extension (``"wav"`` or ``"mp3"``).
            Returns ``None`` if not matched or if channel is not an integer.

        Examples
        --------
        >>> FilePatterns.parse_audio_filename(Path("TRBD002_08062025-03.mp3"))
        (3, 'mp3')
        """
        m = cls.RE_AUDIO.match(p.name)
        if not m:
            return None
        try:
            ch = int(m.group("chan"))
        except ValueError:
            return None
        return ch, m.group("ext").lower()

    @classmethod
    def videogroup_sort_key(cls, seg_id: str) -> Tuple[int, int, str]:
        """
        Build a sort key for segment identifiers that end with ``YYYYMMDD_HHMMSS``.

        Parameters
        ----------
        seg_id : str
            Segment identifier, typically ``"<prefix>_YYYYMMDD_HHMMSS"`` or a
            string containing that pattern as its tail.

        Returns
        -------
        tuple of (int, int, str)
            ``(YYYYMMDD, HHMMSS, seg_id)`` if the tail pattern is found; otherwise
            a large sentinel date/time tuple ``(10**12, 10**8, seg_id)`` so that
            unmatched IDs sort to the end while remaining stable.

        Examples
        --------
        >>> FilePatterns.videogroup_sort_key("TRBD_20250101_010203")
        (20250101, 10203, 'TRBD_20250101_010203')
        """
        m = cls.RE_TAIL.search(seg_id)
        if m:
            return int(m.group("date")), int(m.group("time")), seg_id
        return (10**12, 10**8, seg_id)

    @classmethod
    def parse_tail_datetime(
        cls, seg_id: str, tz: ZoneInfo = DEFAULT_TZ
    ) -> Optional[datetime]:
        """
        Extract a timezone-aware ``datetime`` from a segment ID tail.

        Parameters
        ----------
        seg_id : str
            String whose tail may contain ``YYYYMMDD_HHMMSS``.
        tz : zoneinfo.ZoneInfo, default=DEFAULT_TZ
            Timezone to attach to the parsed naive datetime.

        Returns
        -------
        Optional[datetime]
            A timezone-aware datetime if the tail matches; otherwise ``None``.

        Notes
        -----
        The function does not validate the semantic correctness of the date/time
        (e.g., February 30); it relies on ``datetime.strptime`` to raise if needed.

        Examples
        --------
        >>> FilePatterns.parse_tail_datetime("TRBD_20250101_010203")
        datetime.datetime(2025, 1, 1, 1, 2, 3, tzinfo=ZoneInfo(key='America/Chicago'))
        """
        m = cls.RE_TAIL.search(seg_id)
        if not m:
            return None
        dt = datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=tz)
