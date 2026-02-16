from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from scripts.webui.services.validation_service import ValidationService


def create_validation_api_router(*, validation_service: ValidationService) -> APIRouter:
    router = APIRouter()

    @router.get("/api/drafts/{draft_id}/validate-audio")
    def api_validate_audio(draft_id: int, audio_dir: str) -> Dict[str, Any]:
        return validation_service.validate_audio(
            draft_id,
            audio_dir,
            include_timestamp_state=True,
        )

    @router.get("/api/drafts/{draft_id}/validate-audio/start")
    def api_validate_audio_start(draft_id: int, audio_dir: str) -> Dict[str, Any]:
        return validation_service.start_audio_validation(
            draft_id,
            audio_dir,
            include_timestamp_state=True,
        )

    @router.get("/api/drafts/{draft_id}/validate-audio/status")
    def api_validate_audio_status(draft_id: int) -> Dict[str, Any]:
        return validation_service.audio_validation_status(draft_id)

    @router.get("/api/drafts/{draft_id}/validate-video")
    def api_validate_video(draft_id: int, video_dir: str) -> Dict[str, Any]:
        return validation_service.validate_video(draft_id, video_dir)

    @router.get("/api/drafts/{draft_id}/validate-video/start")
    def api_validate_video_start(draft_id: int, video_dir: str) -> Dict[str, Any]:
        return validation_service.start_video_validation(draft_id, video_dir)

    @router.get("/api/drafts/{draft_id}/validate-video/status")
    def api_validate_video_status(draft_id: int) -> Dict[str, Any]:
        return validation_service.video_validation_status(draft_id)

    @router.get("/api/drafts/{draft_id}/validate-out")
    def api_validate_out(draft_id: int, out_dir: str) -> Dict[str, Any]:
        return validation_service.validate_out(draft_id, out_dir)

    @router.get("/api/drafts/{draft_id}/validate-out/start")
    def api_validate_out_start(draft_id: int, out_dir: str) -> Dict[str, Any]:
        return validation_service.start_out_validation(draft_id, out_dir)

    @router.get("/api/drafts/{draft_id}/validate-out/status")
    def api_validate_out_status(draft_id: int) -> Dict[str, Any]:
        return validation_service.out_validation_status(draft_id)

    return router
