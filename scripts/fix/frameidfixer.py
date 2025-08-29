from scripts.fix.serialfixer import SerialFixer
from typing import List
import numpy as np


class FrameIDFixer(SerialFixer):

    def fix(self, series: List[int]) -> List[int]:
        """
        Unwrap frame_id-style 16-bit counters so they continue increasing
        after 65535 instead of rolling over.

        Logic:
        - Assume the only decreases are true rollovers.
        - Start counter at 0; whenever a drop is observed, counter += 1.
        - Add 65535 * counter to each element.
        """
        if not series:
            return []

        s = np.asarray(series, dtype=np.int64)
        counters = np.zeros(len(s), dtype=np.int64)

        counter = 0
        for i in range(1, len(s)):
            if s[i - 1] > s[i]:
                counter += 1
            counters[i] = counter

        fixed = s + 65535 * counters
        return fixed.tolist()
