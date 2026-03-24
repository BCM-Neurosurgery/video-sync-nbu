"""Format synced video output filenames from user-defined templates.

Placeholders
-------------
{segment_id}  — full segment ID, e.g. ``TRBD001_20250603_133409``
{patient}     — patient/prefix portion, e.g. ``TRBD001``
{cam_serial}  — camera serial number, e.g. ``23512909``
{datetime}    — synced clip start time as ``YYYYMMDD_HHMMSS``
{date}        — date portion only, ``YYYYMMDD``
{time}        — time portion only, ``HHMMSS``
"""

from __future__ import annotations

from datetime import datetime

DEFAULT_TEMPLATE = "{segment_id}.serial{cam_serial}_synced"


def format_synced_tag(
    template: str,
    segment_id: str,
    cam_serial: str,
    synced_start: datetime | None = None,
) -> str:
    """Build a filename stem by expanding *template* with sync metadata.

    Parameters
    ----------
    template:
        Format string with placeholder names (see module docstring).
    segment_id:
        Full segment identifier.
    cam_serial:
        Camera serial number string.
    synced_start:
        Wall-clock time of the synced clip's first frame, derived from
        ``video.start_realtime`` (JSON metadata).  When *None*, date/time
        placeholders resolve to ``00000000`` / ``000000``.

    Returns
    -------
    str
        Expanded filename stem (no extension).
    """
    if synced_start is not None:
        synced_date = synced_start.strftime("%Y%m%d")
        synced_time = synced_start.strftime("%H%M%S")
    else:
        synced_date = "00000000"
        synced_time = "000000"

    patient = segment_id.rsplit("_", 2)[0] if segment_id.count("_") >= 2 else segment_id

    return template.format(
        segment_id=segment_id,
        patient=patient,
        cam_serial=cam_serial,
        datetime=f"{synced_date}_{synced_time}",
        date=synced_date,
        time=synced_time,
    )
