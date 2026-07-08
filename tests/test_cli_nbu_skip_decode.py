from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.cli import cli_nbu


class CliNbuSkipDecodeTests(unittest.TestCase):
    def test_missing_skip_decode_artifacts_lists_required_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = cli_nbu._missing_skip_decode_artifacts(Path(tmp_dir))

        self.assertEqual(
            [path.name for path in missing],
            ["raw.csv", "raw-gapfilled-filtered.csv"],
        )
        self.assertTrue(all(path.parent.name == "audio_decoded" for path in missing))

    def test_skip_decode_preflight_runs_before_audio_input_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_dir = root / "audio"
            video_dir = root / "video"
            out_dir = root / "out"
            audio_dir.mkdir()
            video_dir.mkdir()

            with patch.object(cli_nbu, "_prepare_audio_input_dir") as prepare_mock:
                rc = cli_nbu.run_pipeline(
                    audio_dir=audio_dir,
                    video_dir=video_dir,
                    out_dir=out_dir,
                    site="nbu_sleep",
                    segments=None,
                    cameras=None,
                    target_pairs=None,
                    skip_decode=True,
                )

        self.assertEqual(rc, 4)
        prepare_mock.assert_not_called()

    def test_skip_decode_preflight_passes_when_required_csvs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            decoded_dir = root / "audio_decoded"
            decoded_dir.mkdir()
            (decoded_dir / "raw.csv").write_text("sample,value\n", encoding="utf-8")
            (decoded_dir / "raw-gapfilled-filtered.csv").write_text(
                "sample,value\n",
                encoding="utf-8",
            )

            self.assertTrue(cli_nbu._preflight_skip_decode_artifacts(root))


if __name__ == "__main__":
    unittest.main()
