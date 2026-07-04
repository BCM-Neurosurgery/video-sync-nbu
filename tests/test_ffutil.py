from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.utility import ffutil


class FfutilTests(unittest.TestCase):
    def test_nvenc_uses_calibrated_cq_for_libx264_crf_input(self) -> None:
        with patch.object(ffutil, "detect_h264_encoder", return_value="h264_nvenc"):
            args = ffutil.h264_encode_args(crf=18)

        self.assertIn("-cq", args)
        self.assertEqual(args[args.index("-cq") + 1], "23")
        self.assertNotIn("-crf", args)

    def test_nvenc_cq_mapping_is_clamped_to_ffmpeg_range(self) -> None:
        self.assertEqual(ffutil._nvenc_cq_from_crf(-20), 0)
        self.assertEqual(ffutil._nvenc_cq_from_crf(18), 23)
        self.assertEqual(ffutil._nvenc_cq_from_crf(60), 51)

    def test_libx264_keeps_requested_crf(self) -> None:
        with patch.object(ffutil, "detect_h264_encoder", return_value="libx264"):
            args = ffutil.h264_encode_args(crf=18, preset="fast")

        self.assertIn("-crf", args)
        self.assertEqual(args[args.index("-crf") + 1], "18")
        self.assertEqual(args[args.index("-preset") + 1], "fast")
        self.assertNotIn("-cq", args)


if __name__ == "__main__":
    unittest.main()
