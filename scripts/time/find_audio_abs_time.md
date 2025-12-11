# find_audio_abs_time output schema

When you run `python -m scripts.time.find_audio_abs_time` the tool writes a JSON file (default: `<out>/audio_metadata/audio_abs_start.json`). Each object looks like this:

- `audio_path`: absolute path to the audio channel file (MP3/WAV) described.
- `channel`: numeric audio channel identifier (01/02/03) inferred from the filename.
- `sample_rate`: sampling rate (Hz) reported by `AudioDiscoverer` for this channel.
- `start_time_utc`: absolute start timestamp of the audio file in UTC, derived from the earliest anchor’s realtime minus the anchor’s audio-sample offset.
- `start_time_chicago`: same timestamp converted to America/Chicago (`DEFAULT_TZ`).
- `reference_audio_sample`: audio sample index of the earliest anchor that linked the serial track to video.
- `reference_sample_rate`: sample rate (Hz) of the serial channel used to convert samples to seconds.
- `anchor_segment`: segment ID (e.g. `TRBD002_20250806_104707`) containing the reference anchor.
- `anchor_cam_serial`: camera serial whose JSON supplied realtime metadata for that anchor.
- `anchor_frame_id`: raw frame ID (from the camera JSON) tied to the anchor.
- `anchor_frame_id_reidx`: reindexed frame ID if the JSON companion supplied `fixed_reidx_frame_ids` (otherwise null).
- `anchor_frame_time_utc`: UTC timestamp of the anchor’s video frame, taken from the camera JSON `real_times` array (or approximated via `start_realtime` + frame index if necessary).
- `anchor_json_path`: path to the camera JSON file that provided the companion metadata.

These keys correspond to the `AudioStartRecord` dataclass in `scripts/time/find_audio_abs_time.py` and allow downstream tools (e.g., `cli_nbu`) to reuse the computed basetimes without re-running anchor generation.
