# video-sync-nbu

Audio/video synchronization toolkit for NBU datasets, with both a CLI pipeline and a FastAPI Web UI.

## Features

- End-to-end NBU sync pipeline (`python -m scripts.cli.cli_nbu`)
- Three sync selection modes:
  - segment/camera pairs
  - time range (`--time-start`, `--time-end`, `--time-zone`)
  - audio sample range (`--audio-sample-start`, `--audio-sample-end`)
- Web UI wizard for validation, decode, selection, run scheduling, and log viewing
- Shared decoded-audio artifacts plus per-run outputs under `out/runs/runNNNN`

## Requirements

- Python 3.10+ (3.12 recommended)
- `ffmpeg` on `PATH`
- OS support for your chosen Python deps in `requirements.txt`

## Installation

```bash
git clone https://github.com/BCM-Neurosurgery/video-sync-nbu.git
cd video-sync-nbu

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

ffmpeg -version
```

You can also use Conda; see `docs/nbu/installation.md`.

## CLI Quick Start

Run all discovered segments/cameras:

```bash
python -m scripts.cli.cli_nbu \
  --audio-dir /path/to/audio \
  --video-dir /path/to/video \
  --out-dir /path/to/out \
  --site jamail \
  --split \
  --split-overwrite \
  --overwrite-clips
```

Run explicit segment/camera targets:

```bash
python -m scripts.cli.cli_nbu \
  --audio-dir /path/to/audio \
  --video-dir /path/to/video \
  --out-dir /path/to/out \
  --site nbu_lounge \
  --target SEGMENT_A::23512909 \
  --target SEGMENT_B::24253455
```

Run with time range:

```bash
python -m scripts.cli.cli_nbu \
  --audio-dir /path/to/audio \
  --video-dir /path/to/video \
  --out-dir /path/to/out \
  --site jamail \
  --time-start "2025-12-08 12:09:02" \
  --time-end "2025-12-08 12:15:06" \
  --time-zone "America/Chicago"
```

Run with audio sample range:

```bash
python -m scripts.cli.cli_nbu \
  --audio-dir /path/to/audio \
  --video-dir /path/to/video \
  --out-dir /path/to/out \
  --site jamail \
  --audio-sample-start 120000 \
  --audio-sample-end 360000
```

Notes:

- `--site` choices: `jamail`, `nbu_lounge`, `nbu_sleep`
- Choose one range mode only: time range or audio sample range
- Use `--skip-decode` only when decoded artifacts already exist under `<out>/audio_decoded`

## Web UI

Start the app:

```bash
make webui
```

Then open `http://127.0.0.1:8000`.

Optional env vars:

- `VSYNC_WEBUI_HOST` (default `127.0.0.1`)
- `VSYNC_WEBUI_PORT` (default `8000`)
- `VSYNC_WEBUI_MAX_PARALLEL` (default `1`) for queued sync runs
- `VSYNC_WEBUI_PYTHON` to choose the Python executable for launched CLI runs

## Output Layout

At a high level:

- Shared artifacts in `<out>/audio_decoded`, `<out>/serial_audio_splitted`, `<out>/split_decoded`
- Per-run results in `<out>/runs/runNNNN/...`

See `docs/nbu/io-layouts.md` for the full layout and examples by mode.

## Development

Run Web UI:

```bash
make webui
```

Format templates:

```bash
make format-html
```

Format/check Web UI assets (requires Node.js + `npx`):

```bash
make format-webui-assets
make check-webui-assets
```

Preview docs:

```bash
mkdocs serve -a 127.0.0.1:8000
```
