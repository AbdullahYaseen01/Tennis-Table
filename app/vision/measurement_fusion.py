from __future__ import annotations

from collections import deque

import numpy as np

from app.config import TEMPORAL_FUSION_WINDOW

class MeasurementFusion:
    

    def __init__(self, window: int = TEMPORAL_FUSION_WINDOW) -> None:
        self._window = window
        self._major_px: deque[float] = deque(maxlen=window)
        self._minor_px: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._major_px.clear()
        self._minor_px.clear()

    @staticmethod
    def _trimmed_median(values: deque[float], trim: float = 0.15) -> float:
        arr = np.array(values, dtype=np.float64)
        if len(arr) < 3:
            return float(np.median(arr))
        lo = int(len(arr) * trim)
        hi = max(lo + 1, len(arr) - lo)
        trimmed = np.sort(arr)[lo:hi]
        return float(np.median(trimmed))

    def apply_px(self, major_px: float, minor_px: float) -> tuple[float, float]:
        self._major_px.append(major_px)
        self._minor_px.append(minor_px)
        if len(self._major_px) < 2:
            return major_px, minor_px
        return self._trimmed_median(self._major_px), self._trimmed_median(self._minor_px)

    @property
    def sample_count(self) -> int:
        return len(self._minor_px)
