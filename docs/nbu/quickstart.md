# Quick Start (Elias example)

Run the A/V sync pipeline end-to-end using the example dataset on Elias. Copy
the example folder to your own workspace before running so the shared data
stays pristine.

## What you need

- Access to Elias and the example path: `/mnt/datalake/synced_videos/Example_data/`.
- Project installed and Conda env active (see [Installation](installation.md)).
- FFmpeg on `PATH`.

## Example input layout

```
/mnt/datalake/synced_videos/Example_data/
├── audio/
│   ├── AA004_09042025-01_metadata.json
│   ├── AA004_09042025-01.wav
│   ├── AA004_09042025-02.wav
│   └── AA004_09042025-03.wav
├── video/
│   ├── AA004_20250904_112309.{24253445,24253458,24253463}.mp4
│   ├── AA004_20250904_112309.json
│   ├── AA004_20250904_113323.{24253445,24253458,24253463}.mp4
│   ├── AA004_20250904_113323.json
│   ├── AA004_20250904_114338.{24253445,24253458,24253463}.mp4
│   └── AA004_20250904_114338.json
└── out/
```

## Run all cameras and segments

```bash
cd /mnt/datalake/synced_videos/Example_data/
conda activate video-sync

python -m scripts.cli.cli_nbu \
  --audio-dir /mnt/datalake/synced_videos/Example_data/audio \
  --video-dir /mnt/datalake/synced_videos/Example_data/video \
  --out-dir   /mnt/datalake/synced_videos/Example_data/out \
  --site jamail \
  --split \
  --split-overwrite \
  --log-level INFO \
  --overwrite-clips
```

## Run specific segments/cameras

```bash
python -m scripts.cli.cli_nbu \
  --audio-dir /mnt/datalake/synced_videos/Example_data/audio \
  --video-dir /mnt/datalake/synced_videos/Example_data/video \
  --out-dir   /mnt/datalake/synced_videos/Example_data/out \
  --site jamail \
  --camera 24253458 \
  --segment AA004_20250904_112309 \
  --segment AA004_20250904_113323 \
  --split \
  --split-overwrite \
  --log-level INFO \
  --overwrite-clips
```

## Skip audio decode (reuse prior outputs)

```bash
python -m scripts.cli.cli_nbu \
  --audio-dir /mnt/datalake/synced_videos/Example_data/audio \
  --video-dir /mnt/datalake/synced_videos/Example_data/video \
  --out-dir   /mnt/datalake/synced_videos/Example_data/out \
  --site jamail \
  --log-level INFO \
  --overwrite-clips \
  --skip-decode
```

`run.example.sh` in the same folder contains these commands for copy/paste. Outputs
land under `out/` with per-segment/camera subfolders and logs. Adjust `--site`
if you target a different camera layout.
