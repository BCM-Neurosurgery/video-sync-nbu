# Which CLI should I use?

This repo ships two A/V sync CLIs. Pick the path that matches your data and
workflow.

## NBU A/V Sync (`python -m scripts.cli.cli_nbu`)

- Use for Jamail/NBU lounge/sleep recordings with serial-bearing audio
  (`-03.wav`) and camera MP4s plus JSON companions.
- Inputs: `audio/` (01/02/03 wav), `video/` with `SEGMENT_TIMESTAMP.CAMSERIAL.mp4`
  and `SEGMENT_TIMESTAMP.json`.
- Typical flags: `--site`, `--camera`, `--segment`, `--split`, `--skip-decode`,
  `--overwrite-*`.
- Start here:
  - Quick start with Elias sample: [Quick Start](../nbu/quickstart.md)
  - Input/output shapes: [Input & Output Layouts](../nbu/io-layouts.md)

## EMU A/V Sync (`python -m scripts.cli.cli_emu_time`)

- Use for EMU tasks: stitched NS5 audio + first NEV, camera MP4s with JSON
  companions; often no chunk serials.
- Inputs (per task): `patient_dir` (NS5/NEV), `video_dir` (MP4+JSON), `out_dir`.
- Run headless via CLI or through Prefect UI for queued runs.
- Start here:
  - Prefect UI workflow: [Run via UI](../emu/ui.md)
    - EMU notes/examples: see the EMU Time Sync section in the nav.

## Shared tips

- Activate your Conda env before running (`conda activate video-sync`).
- Ensure FFmpeg is on `PATH` (`ffmpeg -version`).
- Keep filenames intact; discovery relies on the naming patterns above.
