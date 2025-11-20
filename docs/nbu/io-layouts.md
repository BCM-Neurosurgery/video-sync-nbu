# Input & Output Layouts (NBU)

What the NBU A/V sync CLI expects on input and where it writes results.

## Inputs

- **Audio dir**: `audio/`
    - Naming: `<PREFIX>-<CHAN>.<EXT>` where `<CHAN>` is `01/02/03` and `<EXT>` is `wav` or `mp3`.
    - In NBU, `-03` carries the serial encodings. Example: `TRBD002_08062025-03.wav`.
- **Video dir**: `video/`
    - Naming: `<BASE>.<CAM>.mp4` where `<BASE>` is `<PREFIX>_YYYYMMDD_HHMMSS` and `<CAM>` is the camera serial.
    - Companion JSON per segment: `<BASE>.json` (same `<BASE>` as the MP4s).
    - Example: `TRBD002_20250806_104707.24253448.mp4` + `TRBD002_20250806_104707.json`.

Minimal tree:

```
input_root/
├── audio/
│   ├── TRBD002_08062025-01.wav
│   ├── TRBD002_08062025-02.wav
│   └── TRBD002_08062025-03.wav
└── video/
    ├── TRBD002_20250806_104707.24253448.mp4
    ├── TRBD002_20250806_104707.json
    └── ...
```

## Outputs

All outputs land under your `--out-dir`:

- `audio_decoded/` — merged/raw CSVs (`raw`, `raw-gapfilled`, `raw-gapfilled-filtered`).
- `serial_audio_splitted/`, `split_decoded/` — only if `--split` is used.
- Per segment (`<out>/<segment_id>/`) and per camera (`<segment>/<camera>/`):
    - `synced_audio/` — final aligned WAVs (A1/A2).
    - `synced_video/` — final synced MP4s; look for `..._synced.mp4`.
    - `audio_clipped/`, `audio_padded/` — intermediate audio.
    - `work/` — intermediate JSON/CSV (anchors, clip plans, analysis).
    - `sync.log` — per-camera log; root `sync-run.log` sits at `--out-dir`.

Typical footprint:

```
out/
├── sync-run.log                  # root run log
├── audio_decoded/
├── serial_audio_splitted/        # if split enabled
├── split_decoded/                # if split enabled
└── <SEGMENT_ID>/                 # e.g., TRBD002_20250806_104707
    └── <CAM_SERIAL>/             # e.g., 24253448
        ├── synced_video/         # final synced MP4s
        ├── synced_audio/         # final aligned WAVs
        ├── audio_clipped/        # intermediates
        ├── audio_padded/         # intermediates
        ├── work/                 # plans/anchors/analysis
        └── sync.log              # per-camera log
```

## Tips

- Keep filenames untouched; discovery relies on naming patterns.
- Ensure FFmpeg is on `PATH`; many steps depend on it.
- Use `--overwrite-…` flags intentionally to avoid stale artifacts.
