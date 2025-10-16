# Running EMU Time Sync via Prefect UI (v3)

This guide walks you through launching the Prefect 3 server locally and running the EMU time-sync flow (cli_emu_time) entirely from the Prefect UI.

Tested with Prefect 3.4.23.

---

## Prerequisites

- Python/Conda environment with this project installed (editable install ok).
- A deployment spec: `prefect/time_sync_deployment.yaml` (provided in this repo).
- Network access to local Prefect server: UI will run at http://127.0.0.1:4200.

---

## 1) Start the Prefect server and UI

- Open a terminal and run:

```bash
prefect server start
```

- Visit the UI at: http://127.0.0.1:4200

What this does: launches an API and UI locally so you can register deployments and trigger runs.

---

## 2) Point your CLI to the local API

- In a new terminal:

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
```

What this does: tells the Prefect CLI where to register deployments and submit runs.

Tip: add this export to your shell profile to persist across terminals.

---

## 3) Create a work pool for local execution

Create a process-based pool that a local worker will poll:

```bash
prefect work-pool create default-agent-pool --type process
```

What this does: defines a queue (pool) so your worker knows where to pick jobs from.

---

## 4) Register the deployment from YAML

Use the deployment file in this repo:

```bash
prefect deploy --prefect-file prefect/time_sync_deployment.yaml --name emu-local
```

What this does: registers a deployment named `emu-local` under the `time-sync-batch` flow (per the YAML), so it appears in the UI.

---

## 5) Start a worker

In a new terminal, start a worker bound to the pool:

```bash
prefect worker start --pool default-agent-pool
```

What this does: the worker polls the pool for new runs and executes them locally.

---

## 6) Trigger chained runs from the UI

In the UI:

- Navigate to Deployments → `time-sync-batch` → `emu-local` → **Run**.
- Provide parameters mirroring your CLI usage. You can queue multiple jobs in order by placing multiple entries inside the `runs` list **or** by supplying a mapping where each key is a patient label—the flow executes them sequentially, so each job begins only after the previous one finishes successfully.

Example using a mapping so Prefect labels each task by patient (the second run waits for the first to finish):

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
    },
    "YFK": {
      "patient_dir": "/home/auto/CODE/utils/video-sync-nbu/data/emu_serial_example_2/YFK",
      "video_dir": "/mnt/datalake/data/emu/YFKDatafile/VIDEO/",
      "out_dir": "/home/auto/CODE/utils/video-sync-nbu/data/emu_serial_example_2/out",
      "cam_serial": "23512014",
      "keywords": ["0017"],
      "room_mic": "roommic1",
      "log_level": "DEBUG",
      "overwrite": true
    }
  }
}
```

When a mapping is used, the key (e.g. `"YFO"`) becomes the task name inside Prefect. You can also add an explicit `"name"` field to any run if you prefer to stay with a list structure. All runs use `cli_emu_time`; include any additional CLI flags by adding their snake_case versions to each run dictionary (e.g. `"extra_option": "value"`).

What this does: submits a parametrized batch run to the worker. You’ll see logs both in the worker terminal and in the Prefect UI for each run.

---

## 7) Stop services when finished

- Ctrl+C in the worker terminal to stop the worker.
- Ctrl+C in the server terminal to stop the Prefect server/UI.

---

## Troubleshooting

- No runs picked up: Ensure the worker is connected to the correct pool and the deployment exists in the UI.
- API errors on deploy: Verify `PREFECT_API_URL` points to `http://127.0.0.1:4200/api` and the server is running.
- Import errors at runtime: Verify your Python env can import the flow module defined in `time_sync_deployment.yaml`.
- Permission/path issues: Confirm `patient_dir`, `video_dir`, and `out_dir` exist and are readable/writable for your user.
- `ValueError` about missing fields: ensure your JSON uses the exact structure above with a top-level `runs` list or mapping and each run specifying `patient_dir`, `video_dir`, and `out_dir`.
