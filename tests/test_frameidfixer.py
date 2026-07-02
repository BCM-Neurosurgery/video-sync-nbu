from __future__ import annotations

import unittest

from scripts.fix.frameidfixer import FrameIDFixer


class FrameIDFixerTests(unittest.TestCase):
    def test_small_local_decrease_is_not_treated_as_rollover(self) -> None:
        values = [24083, 24091, 24090, 24092]

        result = FrameIDFixer().fix(values)

        self.assertEqual(result, values)

    def test_large_decrease_is_treated_as_rollover(self) -> None:
        result = FrameIDFixer().fix([65533, 65534, 2, 3])

        self.assertEqual(result, [65533, 65534, 65537, 65538])


if __name__ == "__main__":
    unittest.main()
