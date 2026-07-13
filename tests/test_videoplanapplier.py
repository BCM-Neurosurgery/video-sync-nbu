from __future__ import annotations

import io
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from scripts.models import CamJson, Video
from scripts.pad.videoplanapplier import (
    apply_video_padding_plan,
    reconcile_video_companion_frame_count,
)


class _NonClosingBytesIO(io.BytesIO):
    def close(self) -> None:
        pass


class _FakeProcess:
    def __init__(self, *, stdout=None, stdin=None):
        self.stdout = stdout
        self.stdin = stdin
        self.stderr = io.BytesIO()
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


class _ParsedVideo:
    frame_count = 4


def _video_with_companion(
    *,
    frame_count: int,
    fixed_frame_ids: list[int],
    fixed_reidx_frame_ids: list[int],
) -> Video:
    length = len(fixed_frame_ids)
    companion = CamJson(
        cam_serial="18486634",
        timestamp=datetime(2025, 6, 11, 13, 7, 56),
        path=Path("TRBD001_20250611_080756.json"),
        start_realtime=datetime(2025, 6, 11, 13, 7, 56),
        real_times=[f"t{i}" for i in range(length)],
        raw_serials=[9141 + i for i in range(length)],
        raw_frame_ids=list(fixed_frame_ids),
        fixed_serials=[9141 + i for i in range(length)],
        fixed_frame_ids=list(fixed_frame_ids),
        fixed_reidx_frame_ids=list(fixed_reidx_frame_ids),
    )
    return Video(
        path=Path("TRBD001_20250611_080756.18486634.mp4"),
        segment_id="TRBD001_20250611_080756",
        cam_serial="18486634",
        timestamp=datetime(2025, 6, 11, 13, 7, 56),
        start_realtime=datetime(2025, 6, 11, 13, 7, 56),
        duration=1.0,
        resolution="2x2",
        frame_rate=30.0,
        frame_count=frame_count,
        companion_json=companion,
    )


def _write_one_insertion_plan(path: Path, video: Video) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "segment_id": video.segment_id,
                "cam_serial": video.cam_serial,
                "source_video": str(video.path),
                "source_fps": 30.0,
                "target_fps": 30.0,
                "policy": "dup-prev",
                "total_insertions": 1,
                "operations": [
                    {
                        "after_index": 1,
                        "insert": 1,
                        "frame_id_before": 10,
                        "frame_id_after": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class VideoPlanApplierMetadataTests(unittest.TestCase):
    def test_historical_first_segment_mismatch_fails_without_explicit_repair(
        self,
    ) -> None:
        video = _video_with_companion(
            frame_count=3,
            fixed_frame_ids=[10, 11, 12, 13],
            fixed_reidx_frame_ids=[0, 1, 2, 3],
        )

        with self.assertRaisesRegex(
            RuntimeError, "explicit repair option|CamJson array lengths"
        ):
            reconcile_video_companion_frame_count(video)

    def test_explicit_historical_first_segment_repair_drops_row_one(self) -> None:
        video = _video_with_companion(
            frame_count=3,
            fixed_frame_ids=[10, 11, 12, 13],
            fixed_reidx_frame_ids=[0, 1, 2, 3],
        )

        reconciled = reconcile_video_companion_frame_count(
            video,
            repair_historical_first_segment_mismatch=True,
        )

        self.assertIsNot(reconciled, video)
        self.assertEqual(reconciled.frame_count, 3)
        self.assertEqual(
            reconciled.companion_json.fixed_frame_ids,
            [10, 12, 13],
        )
        self.assertEqual(
            reconciled.companion_json.fixed_reidx_frame_ids,
            [0, 2, 3],
        )
        self.assertEqual(reconciled.companion_json.fixed_serials, [9141, 9143, 9144])
        self.assertEqual(reconciled.companion_json.real_times, ["t0", "t2", "t3"])

    def test_matching_lengths_are_left_unchanged(self) -> None:
        video = _video_with_companion(
            frame_count=3,
            fixed_frame_ids=[10, 11, 12],
            fixed_reidx_frame_ids=[0, 1, 2],
        )

        reconciled = reconcile_video_companion_frame_count(video)

        self.assertIs(reconciled, video)

    def test_optional_arrays_already_matching_video_are_preserved(self) -> None:
        video = _video_with_companion(
            frame_count=3,
            fixed_frame_ids=[10, 11, 12, 13],
            fixed_reidx_frame_ids=[0, 1, 2, 3],
        )
        video = replace(
            video,
            companion_json=replace(video.companion_json, real_times=["t0", "t2", "t3"]),
        )

        reconciled = reconcile_video_companion_frame_count(
            video,
            repair_historical_first_segment_mismatch=True,
        )

        self.assertEqual(reconciled.companion_json.fixed_frame_ids, [10, 12, 13])
        self.assertEqual(reconciled.companion_json.real_times, ["t0", "t2", "t3"])

    def test_optional_arrays_with_unexpected_lengths_fail(self) -> None:
        video = _video_with_companion(
            frame_count=3,
            fixed_frame_ids=[10, 11, 12, 13],
            fixed_reidx_frame_ids=[0, 1, 2, 3],
        )
        video = replace(
            video,
            companion_json=replace(
                video.companion_json, real_times=["only-one", "two"]
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "Optional CamJson array length"):
            reconcile_video_companion_frame_count(
                video,
                repair_historical_first_segment_mismatch=True,
            )

    def test_non_historical_mismatch_fails(self) -> None:
        video = _video_with_companion(
            frame_count=3,
            fixed_frame_ids=[10, 11, 13, 14],
            fixed_reidx_frame_ids=[0, 1, 3, 4],
        )

        with self.assertRaisesRegex(RuntimeError, "CamJson array lengths"):
            reconcile_video_companion_frame_count(
                video,
                repair_historical_first_segment_mismatch=True,
            )

    def test_apply_video_padding_plan_requires_explicit_historical_repair(
        self,
    ) -> None:
        video = _video_with_companion(
            frame_count=3,
            fixed_frame_ids=[10, 11, 12, 13],
            fixed_reidx_frame_ids=[0, 1, 2, 3],
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_json = root / "plan.json"
            _write_one_insertion_plan(plan_json, video)

            with patch("scripts.pad.videoplanapplier._require_ffmpeg"):
                with self.assertRaisesRegex(RuntimeError, "explicit repair option"):
                    apply_video_padding_plan(plan_json, video, root / "out")

    def test_apply_video_padding_plan_with_explicit_repair_updates_metadata(
        self,
    ) -> None:
        video = _video_with_companion(
            frame_count=3,
            fixed_frame_ids=[10, 11, 12, 13],
            fixed_reidx_frame_ids=[0, 1, 2, 3],
        )
        frame_size = 6
        decoded_frames = b"a" * frame_size + b"b" * frame_size + b"c" * frame_size
        encoded_stdin = _NonClosingBytesIO()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_json = root / "plan.json"
            _write_one_insertion_plan(plan_json, video)

            with (
                patch("scripts.pad.videoplanapplier._require_ffmpeg"),
                patch(
                    "scripts.pad.videoplanapplier._spawn_decoder",
                    return_value=_FakeProcess(stdout=io.BytesIO(decoded_frames)),
                ),
                patch(
                    "scripts.pad.videoplanapplier._spawn_encoder",
                    return_value=_FakeProcess(stdin=encoded_stdin),
                ),
                patch(
                    "scripts.pad.videoplanapplier.VideoFileParser",
                    return_value=_ParsedVideo(),
                ),
            ):
                out_path, padded_video = apply_video_padding_plan(
                    plan_json,
                    video,
                    root / "out",
                    repair_historical_first_segment_mismatch=True,
                )

        self.assertEqual(out_path.name, video.path.name)
        self.assertEqual(padded_video.frame_count, 4)
        self.assertEqual(padded_video.companion_json.fixed_frame_ids, [10, 11, 12, 13])
        self.assertEqual(
            padded_video.companion_json.fixed_reidx_frame_ids, [0, 1, 2, 3]
        )
        self.assertEqual(
            padded_video.companion_json.fixed_serials, [9141, 0, 9143, 9144]
        )
        self.assertEqual(len(encoded_stdin.getvalue()), 4 * frame_size)


if __name__ == "__main__":
    unittest.main()
