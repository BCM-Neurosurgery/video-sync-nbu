# NBU CLI Recipes

Copy/paste-ready commands for common `cli_nbu` runs. Replace paths/IDs with your own.

## Run everything (all segments/cameras)

```bash
python -m scripts.cli.cli_nbu \
  --audio-dir /path/to/audio \
  --video-dir /path/to/video \
  --out-dir   /path/to/out \
  --site jamail \
  --split \
  --split-overwrite \
  --log-level INFO \
  --overwrite-clips
```

## Run specific segments and cameras

```bash
python -m scripts.cli.cli_nbu \
  --audio-dir /path/to/audio \
  --video-dir /path/to/video \
  --out-dir   /path/to/out \
  --site jamail \
  --camera 24253458 \
  --segment TRBD002_20250806_104707 \
  --segment TRBD002_20250806_105724 \
  --split \
  --split-overwrite \
  --log-level INFO \
  --overwrite-clips
```

## Reuse decoded audio (skip decode)

```bash
python -m scripts.cli.cli_nbu \
  --audio-dir /path/to/audio \
  --video-dir /path/to/video \
  --out-dir   /path/to/out \
  --site jamail \
  --skip-decode \
  --log-level INFO \
  --overwrite-clips
```

## Tips

- Keep filenames intact (`<PREFIX>-03.wav`, `<BASE>.<CAM>.mp4`, `<BASE>.json`).
- Use `--overwrite-…` flags when rerunning to avoid stale intermediates.
- Check logs: `sync-run.log` at `--out-dir`, and per-camera `sync.log` under each segment.
