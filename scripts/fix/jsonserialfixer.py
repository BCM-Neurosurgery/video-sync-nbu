from scripts.fix.serialfixer import SerialFixer
from typing import List


class JsonSerialFixer(SerialFixer):
    """Camera JSON strategy: apply gap fixes in this order: [2, 130]."""

    def fix(self, series: List[int]) -> List[int]:
        s = list(series)
        for gap in (2, 130):
            s = self.fix_midpoints_gap(s, gap)
        return s
