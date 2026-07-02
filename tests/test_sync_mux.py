from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.align import sync


class SyncMuxTests(unittest.TestCase):
    def test_probe_duration_prefers_stream_duration(self) -> None:
        payload = {
            "streams": [{"duration": "12.345"}],
            "format": {"duration": "99.0"},
        }

        with patch.object(sync.shutil, "which", return_value="/usr/bin/ffprobe"):
            with patch.object(
                sync.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(payload),
                    stderr="",
                ),
            ):
                result = sync._probe_duration(Path("video.mp4"), "v:0")

        self.assertEqual(result, 12.345)

    def test_mux_pads_audio_to_video_duration_without_shortest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            video = root / "clip.mp4"
            audio_1 = root / "a1.wav"
            audio_2 = root / "a2.wav"
            out = root / "synced" / "out.mp4"

            with patch.object(sync.shutil, "which", return_value="/usr/bin/ffmpeg"):
                with patch.object(sync, "_probe_duration", return_value=616.069345):
                    with patch.object(
                        sync.subprocess,
                        "run",
                        return_value=SimpleNamespace(returncode=0),
                    ) as run_mock:
                        result = sync.mux_video_audio(
                            video,
                            audio_1,
                            audio_2,
                            fps=None,
                            out_path=out,
                        )

        self.assertEqual(result, out)
        cmd = run_mock.call_args.args[0]
        self.assertNotIn("-shortest", cmd)
        self.assertIn("-filter_complex", cmd)
        filter_arg = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("atrim=duration=616.069345000", filter_arg)
        self.assertIn("apad=whole_dur=616.069345000", filter_arg)
        self.assertIn("[1:a:0]", filter_arg)
        self.assertIn("[2:a:0]", filter_arg)
        self.assertEqual(cmd[cmd.index("-map") + 1], "0:v:0")
        self.assertIn("[a1]", cmd)
        self.assertIn("[a2]", cmd)
        self.assertIn("-c:v", cmd)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")


if __name__ == "__main__":
    unittest.main()
