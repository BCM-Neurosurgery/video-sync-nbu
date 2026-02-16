from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts.webui.routers.runs import create_runs_router
from scripts.webui.routers.validation_api import create_validation_api_router
from scripts.webui.routers.misc import create_misc_router
from scripts.webui.routers.wizard import create_wizard_router
from scripts.webui.decode_workflow import rebuild_decode_artifacts_for_draft
from scripts.webui.services import draft_service, timestamp_service
from scripts.webui.services.validation_service import ValidationService
from scripts.webui.sync_selection import _segment_range_title
from scripts.webui.runner import Runner, RunnerConfig

ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
try:
    templates.env.globals["static_v"] = int(
        (ROOT / "static" / "app.css").stat().st_mtime
    )
except Exception:
    templates.env.globals["static_v"] = 1


def create_app() -> FastAPI:
    app = FastAPI(title="video-sync-nbu Web UI")
    app.mount(
        "/static",
        StaticFiles(directory=str(ROOT / "static")),
        name="static",
    )

    max_parallel = int(os.environ.get("VSYNC_WEBUI_MAX_PARALLEL", "1"))
    runner = Runner(cfg=RunnerConfig(max_parallel=max_parallel))
    runner.start()

    validation_service = ValidationService(
        draft_get=draft_service.get_draft,
        draft_set=draft_service.set_draft,
    )

    app.include_router(create_misc_router(templates=templates))

    app.include_router(
        create_runs_router(
            templates=templates,
            draft_create=draft_service.create_draft,
            segment_range_title=_segment_range_title,
            runner=runner,
        )
    )
    app.include_router(
        create_validation_api_router(validation_service=validation_service)
    )

    app.include_router(
        create_wizard_router(
            templates=templates,
            validation_service=validation_service,
            draft_get=draft_service.get_draft,
            draft_set=draft_service.set_draft,
            draft_delete=draft_service.delete_draft,
            rebuild_decode_artifacts_for_draft=rebuild_decode_artifacts_for_draft,
            resolve_timestamp_output_path=timestamp_service.resolve_timestamp_output_path,
            create_timestamp_run=timestamp_service.create_timestamp_run,
            start_audio_timestamp_job=timestamp_service.start_audio_timestamp_job,
        )
    )

    return app


app = create_app()
