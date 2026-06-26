from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate import synced_video_qc as qc


class SyncedVideoQcTests(unittest.TestCase):
    def test_parse_sync_log_captures_window_and_non_forward_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "sync.log"
            log_path.write_text(
                "\n".join(
                    [
                        "INFO start",
                        (
                            "WARNING [SEG/CAM] Non-forward serial pair at rows "
                            "14069->14070 (delta=-52). Treating as no-gap."
                        ),
                        (
                            "INFO Matched window: frames [1..17999] (n=17999), "
                            "samples [220500..27509428) (618.797s), CFR=29.087104 fps"
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            result = qc.parse_sync_log(log_path)

        self.assertIsNotNone(result.window)
        assert result.window is not None
        self.assertEqual(result.window.frame0, 1)
        self.assertEqual(result.window.frame1, 17999)
        self.assertEqual(result.window.expected_frames, 17999)
        self.assertEqual(result.window.sample0, 220500)
        self.assertEqual(result.window.sample1, 27509428)
        self.assertEqual(len(result.non_forward_warnings), 1)

    def test_discover_camera_dirs_supports_direct_and_runs_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            direct = root / "SEG_A" / "111"
            direct.mkdir(parents=True)
            (direct / "sync.log").write_text("", encoding="utf-8")
            nested = root / "runs" / "run0001" / "SEG_B" / "222"
            nested.mkdir(parents=True)
            (nested / "sync.log").write_text("", encoding="utf-8")

            result = qc.discover_camera_dirs(root)

        self.assertEqual({path.name for path in result}, {"111", "222"})

    def test_anchor_residual_summary_uses_sync_window_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            anchors_path = Path(tmp_dir) / "anchors.json"
            anchors_path.write_text(
                json.dumps(
                    [
                        {"frame_index": 10, "audio_sample": 1000},
                        {"frame_index": 11, "audio_sample": 2000},
                        {"frame_index": 12, "audio_sample": 4500},
                    ]
                ),
                encoding="utf-8",
            )
            window = qc.SyncWindow(
                frame0=10,
                frame1=12,
                expected_frames=3,
                sample0=1000,
                sample1=3000,
                fps=1.0,
            )

            result = qc.summarize_anchor_residuals(
                anchors_path, window, sample_rate=1000
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.anchor_count, 3)
        self.assertAlmostEqual(result.max_abs_ms, 1500.0)

    def test_qc_camera_dir_fails_on_midclip_warning_and_media_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            camera_dir = Path(tmp_dir) / "SEG" / "23512909"
            synced_dir = camera_dir / "synced_video"
            synced_dir.mkdir(parents=True)
            (synced_dir / "SEG.serial23512909_synced.mp4").write_text(
                "", encoding="utf-8"
            )
            (camera_dir / "sync.log").write_text(
                "\n".join(
                    [
                        (
                            "WARNING [SEG/23512909] Non-forward serial pair at rows "
                            "14069->14070 (delta=-52). Treating as no-gap."
                        ),
                        (
                            "INFO Matched window: frames [1..17999] (n=17999), "
                            "samples [220500..27509428) (618.797s), CFR=29.087104 fps"
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            media = qc.MediaInfo(
                path=str(synced_dir / "SEG.serial23512909_synced.mp4"),
                video_duration_sec=618.645781,
                frame_count=17995,
                avg_frame_rate="7122421/244860",
                r_frame_rate="1979/50",
                audio_durations_sec=[618.796009],
                audio_sample_rates=[44100],
            )

            with patch.object(qc, "probe_media", return_value=media):
                result = qc.qc_camera_dir(camera_dir)

        self.assertEqual(result["status"], qc.STATUS_FAIL)
        messages = [issue["message"] for issue in result["issues"]]
        self.assertTrue(any("non-forward" in message for message in messages))
        self.assertTrue(any("frame count" in message for message in messages))
        self.assertTrue(any("audio duration" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
