# Run EMU Time Sync via Prefect UI

Launch the Prefect 3 UI locally and submit the EMU time-sync flow without using
the CLI directly.

## Prerequisites

- Project installed in an active Conda env (`pip install -r requirements.txt`).
- Prefect 3.4.23 installed (`prefect version`).
- Deployment file available: `prefect/time_sync_deployment.yaml`.
- Local ports free: API/UI on 4200, worker uses your shell.

## 1) Start the Prefect server + UI

```bash
prefect server start
```

Visit http://127.0.0.1:4200 to confirm the UI is up.

## 2) Point the CLI to the local API

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
```

## 3) Create a work pool for local runs

```bash
prefect work-pool create default-agent-pool --type process
```

## 4) Register the deployment from YAML

```bash
prefect deploy --prefect-file prefect/time_sync_deployment.yaml --name emu-local
```

## 5) Start a worker

```bash
prefect worker start --pool default-agent-pool
```

## 6) Trigger runs from the UI

- In the UI: Deployments → `time-sync-batch` → `emu-local` → **Run**.
- Provide parameters as JSON. You can queue multiple runs as a mapping so each
  run waits for the previous to finish:

```json
{
  "runs": {
    "YFO": {
      "patient_dir": "/mnt/stitched/EMU-18112/YFO/",
      "video_dir": "/mnt/datalake/data/emu/YFODatafile/VIDEO/",
      "out_dir": "/home/auto/CODE/utils/video-sync-nbu/data/emu_serial_example_1/out",
      "cam_serial": "18486638",
      "keywords": ["0062"],
      "room_mic": "roommic1",
      "log_level": "DEBUG",
      "overwrite": true
    }
  }
}
```

Add more entries to `runs` for chained jobs. For a single run you can also pass a
list instead of a mapping.

## 7) Stop services

- Ctrl+C the worker terminal.
- Ctrl+C the server/UI terminal.

## Troubleshooting

- No runs picked up: ensure the worker is bound to `default-agent-pool` and the
  deployment exists.
- API errors on deploy: confirm `PREFECT_API_URL` is set and the server is up.
- Import errors at runtime: verify your Conda env can import the project modules.
