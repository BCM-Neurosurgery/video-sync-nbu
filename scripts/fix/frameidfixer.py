from scripts.fix.serialfixer import SerialFixer
from typing import List
import numpy as np


class FrameIDFixer(SerialFixer):
    WRAP_VALUE = 65535
    WRAP_THRESHOLD = WRAP_VALUE // 2

    def fix(self, series: List[int]) -> List[int]:
        """
        Unwrap frame_id-style 16-bit counters so they continue increasing
        after 65535 instead of rolling over.

        Logic:
        - Treat only large decreases as true rollovers.
        - Small decreases can happen from local out-of-order frame-id glitches
          and must not shift the rest of the recording by a full counter cycle.
        - Add 65535 * counter to each element.
        """
        if not series:
            return []

        s = np.asarray(series, dtype=np.int64)
        counters = np.zeros(len(s), dtype=np.int64)

        counter = 0
        for i in range(1, len(s)):
            drop = s[i - 1] - s[i]
            if drop > self.WRAP_THRESHOLD:
                counter += 1
            counters[i] = counter

        fixed = s + self.WRAP_VALUE * counters
        return fixed.tolist()
