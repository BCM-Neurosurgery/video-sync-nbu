from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import HTTPException

from scripts.webui.models import utc_now_iso
from scripts.webui.repositories import draft_repo


def create_draft(*, mode: str = "sync") -> int:
    now = utc_now_iso()
    data = {"mode": mode}
    if mode == "audio_timestamp":
        data["site"] = "nbu_lounge"
    return draft_repo.create_draft(
        created_at=now,
        updated_at=now,
        data_json=json.dumps(data),
    )


def get_draft(draft_id: int) -> Dict[str, Any]:
    data_json = draft_repo.get_draft_data_json(draft_id)
    if data_json is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    try:
        return json.loads(data_json)
    except Exception:
        return {}


def set_draft(draft_id: int, data: Dict[str, Any]) -> None:
    draft_repo.update_draft_data_json(
        draft_id=draft_id,
        updated_at=utc_now_iso(),
        data_json=json.dumps(data),
    )


def delete_draft(draft_id: int) -> None:
    draft_repo.delete_draft(draft_id)
