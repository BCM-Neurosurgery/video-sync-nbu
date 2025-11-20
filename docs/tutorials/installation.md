# Installation

Set up the video-sync toolkit locally so you can run the CLI pipelines and view
the docs.

## Prerequisites

- Miniconda/Anaconda installed (or Mamba).
- Python 3.10+ target inside Conda.
- FFmpeg available on your `PATH` (used for clipping/muxing).
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - Windows: `choco install ffmpeg` (or download from ffmpeg.org)
- Git (to clone) and enough disk space for audio/video outputs.

## Install with Conda

```bash
# 1) Grab the code
git clone https://github.com/BCM-Neurosurgery/video-sync-nbu.git
cd video-sync-nbu

# 2) Create and activate a Conda env
conda create -n video-sync python=3.12 -y   # adjust version if needed
conda activate video-sync

# 3) Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) Confirm ffmpeg is reachable
ffmpeg -version
```

## Verify the CLI works

Run the two main entry points to ensure imports resolve:

```bash
python -m scripts.cli.cli_nbu --help
python -m scripts.cli.cli_emu_time --help
```

If you use Prefect for EMU sync, also check `prefect version` and see the
[Prefect UI guide](../prefect/ui.md) for UI setup.

## (Optional) Preview this documentation site

```bash
conda activate video-sync  # if not already active
mkdocs serve -a 127.0.0.1:8000   # or bash bash_scripts/start_doc.sh
```

Open http://127.0.0.1:8000 to browse the docs locally.
