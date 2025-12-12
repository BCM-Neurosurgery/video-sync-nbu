# Absolute audio timestamps

Quick way to learn the real-world start time of each audio channel so you can align with notes, physiology, or other systems without re-running full sync.

## What it does

- Uses the serial track (`-03.wav`) and its earliest anchor to link an audio sample to a camera frame.
- Reads that frame’s realtime from the camera JSON, subtracts the sample offset, and computes the audio start time.
- Reuses existing pipeline artifacts when present (decoded/gapfilled serial CSVs, anchors); otherwise generates them on the fly.
- Applies the same start time to all channels in the visit and reports both UTC and America/Chicago.

## How to run

```bash
python -m scripts.time.find_audio_abs_time \
  --audio-dir /path/to/audio \
  --video-dir /path/to/video \
  --out-dir   /path/to/out \
  --site jamail  # or nbu_lounge / nbu_sleep
```

Example run on Elias

```bash
python -m scripts.time.find_audio_abs_time \
  --audio-dir /mnt/datalake/synced_videos/TRBD-53761/TRBD001/TRBD001_04162025/lounge/audio \
  --video-dir /mnt/datalake/data/TRBD-53761/TRBD001/NBU/2025-04-16/video/lounge \
  --out-dir /mnt/datalake/synced_videos/TRBD-53761/TRBD001/TRBD001_04162025/lounge/out \
  --site nbu_lounge
```

Writes `audio_metadata/audio_abs_start.json` under `--out-dir` and also prints the records.

### Example output (trimmed)

```json
[
  {
    "audio_path": "/.../TRBD001_04-16-2025-01.mp3",
    "channel": 1,
    "sample_rate": 44100,
    "start_time_utc": "2025-04-16T14:52:37.228474+00:00",
    "start_time_chicago": "2025-04-16T09:52:37.228474-05:00",
    "reference_audio_sample": 817,
    "reference_sample_rate": 44100,
    "anchor_segment": "TRBD001_20250416_094618",
    "anchor_cam_serial": "18486634",
    "anchor_frame_id": 11059,
    "anchor_frame_id_reidx": 11058,
    "anchor_frame_time_utc": "2025-04-16T14:52:37.247000+00:00",
    "anchor_json_path": "/.../TRBD001_20250416_094618/18486634/work/gapfilled-filtered-anchors.json"
  }
]
```

## Output schema (key meanings)

Each JSON object describes one audio channel and how its start time was derived:

- `audio_path`: Absolute path to the channel file (`-01/-02/-03` WAV/MP3) this record refers to.
- `channel`: Channel number inferred from the filename; all channels share the same start time.
- `sample_rate`: Sampling rate (Hz) reported for that channel.
- `start_time_utc`: Clock time when the audio file begins, in UTC, computed from the anchor’s realtime minus the serial sample offset.
- `start_time_chicago`: Same as above converted to America/Chicago for quick human reading.
- `reference_audio_sample`: The serial sample index at which we found the first anchor tying audio to video; used to back-compute the start time.
- `reference_sample_rate`: Sample rate of the serial track used to convert samples to seconds.
- `anchor_segment`: Segment ID (e.g., `TRBD001_20250416_094618`) that contained the anchor.
- `anchor_cam_serial`: Camera serial whose JSON companion supplied the realtime for the anchor frame.
- `anchor_frame_id`: Original frame ID from the camera JSON for that anchor.
- `anchor_frame_id_reidx`: Reindexed frame ID if the JSON included `fixed_reidx_frame_ids` (otherwise null).
- `anchor_frame_time_utc`: The realtime for that frame pulled from the camera JSON (`real_times`) or approximated from `start_realtime` + frame index.
- `anchor_json_path`: Path to the anchor file that linked audio and video (lives under `<out>/<segment>/<cam>/work/`).
