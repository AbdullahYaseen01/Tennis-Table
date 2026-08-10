from __future__ import annotations

from collections import deque

import numpy as np

from app.config import BASELINE_FRAME_COUNT, BASELINE_OUTLIER_SIGMA, COMPRESSION_MEDIAN_WINDOW, LOAD_AXIS
from app.core.models import BallMeasurement, CompressionSample

class CompressionTracker:
    

    def __init__(self, load_axis: str = LOAD_AXIS) -> None:
        self.load_axis = load_axis
        self.baseline_major_mm: float | None = None
        self.baseline_minor_mm: float | None = None
        self.baseline_diameter_mm: float | None = None
        self._baseline_buffer: list[tuple[float, float]] = []
        self._baseline_px_buffer: list[tuple[float, float]] = []
        self._filter_window = COMPRESSION_MEDIAN_WINDOW
        self._compression_history: deque[float] = deque(maxlen=self._filter_window)
        self._bulge_history: deque[float] = deque(maxlen=self._filter_window)
        self._timeseries: list[CompressionSample] = []
        self._max_compression_pct = 0.0

    def reset(self) -> None:
        self.baseline_major_mm = None
        self.baseline_minor_mm = None
        self.baseline_diameter_mm = None
        self._baseline_buffer.clear()
        self._baseline_px_buffer.clear()
        self._compression_history.clear()
        self._bulge_history.clear()
        self._timeseries.clear()
        self._max_compression_pct = 0.0

    def add_baseline_sample(self, measurement: BallMeasurement) -> bool:
        
        self._baseline_buffer.append((measurement.major_mm, measurement.minor_mm))
        self._baseline_px_buffer.append((measurement.major_px, measurement.minor_px))
        if len(self._baseline_buffer) < BASELINE_FRAME_COUNT:
            return False
        self._lock_baseline_from_buffer()
        return True

    def _lock_baseline_from_buffer(self) -> None:
        majors = np.array([m for m, _ in self._baseline_buffer], dtype=np.float64)
        minors = np.array([n for _, n in self._baseline_buffer], dtype=np.float64)

        def _robust_median(values: np.ndarray) -> float:
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med))) + 1e-6
            inliers = values[np.abs(values - med) <= BASELINE_OUTLIER_SIGMA * 1.4826 * mad]
            return float(np.median(inliers)) if len(inliers) > 0 else med

        self.baseline_major_mm = _robust_median(majors)
        self.baseline_minor_mm = _robust_median(minors)
        self.baseline_diameter_mm = (
            self.baseline_minor_mm if self.load_axis == "minor" else self.baseline_major_mm
        )

    def relock_baseline_mm_from_px(self, calibrator) -> None:
        
        if not self._baseline_px_buffer:
            return

        majors_px = np.array([p[0] for p in self._baseline_px_buffer], dtype=np.float64)
        minors_px = np.array([p[1] for p in self._baseline_px_buffer], dtype=np.float64)

        def _robust_median(values: np.ndarray) -> float:
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med))) + 1e-6
            inliers = values[np.abs(values - med) <= BASELINE_OUTLIER_SIGMA * 1.4826 * mad]
            return float(np.median(inliers)) if len(inliers) > 0 else med

        major_px = _robust_median(majors_px)
        minor_px = _robust_median(minors_px)
        self.baseline_major_mm = calibrator.px_to_mm(major_px)
        self.baseline_minor_mm = calibrator.px_to_mm(minor_px)
        self.baseline_diameter_mm = (
            self.baseline_minor_mm if self.load_axis == "minor" else self.baseline_major_mm
        )

    @property
    def baseline_median_minor_px(self) -> float | None:
        if not self._baseline_px_buffer:
            return None
        minors = np.array([p[1] for p in self._baseline_px_buffer], dtype=np.float64)
        return float(np.median(minors))

    def _load_perp_axes(self, measurement: BallMeasurement) -> tuple[float, float, float]:
        if self.load_axis == "minor":
            load_mm = measurement.minor_mm
            bulge_mm = measurement.major_mm
        else:
            load_mm = measurement.major_mm
            bulge_mm = measurement.minor_mm
        baseline_bulge = (
            self.baseline_major_mm if self.load_axis == "minor" else self.baseline_minor_mm
        )
        return load_mm, bulge_mm, baseline_bulge or bulge_mm

    def update(
        self, measurement: BallMeasurement, phase: str = ""
    ) -> CompressionSample | None:
        if self.baseline_diameter_mm is None or self.baseline_diameter_mm <= 0:
            return None

        load_mm, bulge_mm, baseline_bulge = self._load_perp_axes(measurement)
        raw_compression = (
            (self.baseline_diameter_mm - load_mm) / self.baseline_diameter_mm * 100.0
        )
        self._compression_history.append(raw_compression)
        self._bulge_history.append(bulge_mm - baseline_bulge)

        compression_pct = float(np.median(self._compression_history))
        bulge_delta = float(np.median(self._bulge_history))
        self._max_compression_pct = max(self._max_compression_pct, compression_pct)

        sample = CompressionSample(
            timestamp=measurement.timestamp,
            compression_pct=compression_pct,
            bulge_mm=bulge_delta,
            diameter_mm=load_mm,
            phase=phase,
        )
        self._timeseries.append(sample)
        return sample

    @property
    def max_compression_pct(self) -> float:
        return self._max_compression_pct

    @property
    def timeseries(self) -> list[CompressionSample]:
        return list(self._timeseries)

    @property
    def baseline_mm(self) -> float | None:
        return self.baseline_diameter_mm

    def live_compression_pct(self) -> float:
        if not self._compression_history:
            return 0.0
        return float(np.median(self._compression_history))
