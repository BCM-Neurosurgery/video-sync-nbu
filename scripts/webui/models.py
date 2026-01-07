from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    # Historical name: we now stamp runs in local time for user-facing clarity.
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Run:
    id: int
    title: str
    created_at: str
    scheduled_at: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    status: str
    exit_code: Optional[int]
    pid: Optional[int]
    cancel_requested: int
    cwd: str
    cmd_json: str
    args_json: str
    log_path: str
    error: Optional[str]

    @property
    def cmd(self) -> List[str]:
        return json.loads(self.cmd_json)

    @property
    def args(self) -> Dict[str, Any]:
        return json.loads(self.args_json)

    @property
    def log_file(self) -> Path:
        return Path(self.log_path)
