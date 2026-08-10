from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from app.config import (
    AUTO_SCALE_ON_BASELINE,
    COMPRESSION_THRESHOLD_PCT,
    MIN_COMPRESSION_PEAK_PCT,
    REFERENCE_DIAMETERS_MM,
    RELEASE_THRESHOLD_PCT,
)
from app.core.models import (
    BallMeasurement,
    CompressionSample,
    RecoveryResult,
    SurfaceAnalysisResult,
    TestRunSummary,
    TestState,
)
from app.vision.calibration import Calibrator
from app.vision.compression import CompressionTracker
from app.vision.detect import BallDetector
from app.vision.recovery import RecoveryAnalyzer
from app.vision.surface import SurfaceAnalyzer

logger = logging.getLogger(__name__)

class TestPipeline:
    

    def __init__(self, calibrator: Calibrator) -> None:
        self.calibrator = calibrator
        self.detector = BallDetector(calibrator)
        self.compression = CompressionTracker()
        self.recovery = RecoveryAnalyzer()
        self.surface = SurfaceAnalyzer(calibrator)

        self.state = TestState.IDLE
        self.ball_type = "tennis"
        self.ball_id = "ball-001"
        self._last_measurement: BallMeasurement | None = None
        self._last_compression: CompressionSample | None = None
        self._recovery_result: RecoveryResult | None = None
        self._surface_result: SurfaceAnalysisResult | None = None
        self._peak_compression = 0.0
        self._was_compressing = False
        self._release_detected = False
        self._test_start_time: float | None = None
        self._callbacks: dict[str, list[Callable]] = {
            "state_changed": [],
            "measurement": [],
            "compression": [],
            "recovery": [],
            "surface": [],
            "scale_updated": [],
        }

    def on(self, event: str, callback: Callable) -> None:
        self._callbacks.setdefault(event, []).append(callback)

    def _emit(self, event: str, *args: object) -> None:
        for cb in self._callbacks.get(event, []):
            cb(*args)

    def set_ball_type(self, ball_type: str) -> None:
        self.ball_type = ball_type
        self.detector.set_ball_type(ball_type)
        self.surface.set_ball_type(ball_type)

    def set_ball_id(self, ball_id: str) -> None:
        self.ball_id = ball_id

    def reset_test(self) -> None:
        self.state = TestState.IDLE
        self.compression.reset()
        self.recovery.reset()
        self._last_measurement = None
        self._last_compression = None
        self._recovery_result = None
        self._surface_result = None
        self._peak_compression = 0.0
        self._was_compressing = False
        self._release_detected = False
        self._test_start_time = None
        self.detector.reset_roi()
        self._emit("state_changed", self.state)

    def start_baseline_capture(self) -> None:
        self.compression.reset()
        self.detector.reset_roi()
        self.state = TestState.BASELINE
        self._emit("state_changed", self.state)

    def start_test(self) -> None:
        if self.compression.baseline_mm is None:
            self.start_baseline_capture()
            return
        self.state = TestState.COMPRESSING
        self._peak_compression = 0.0
        self._was_compressing = False
        self._release_detected = False
        self.recovery.reset()
        self.recovery.set_baseline(self.compression.baseline_mm)
        self._test_start_time = None
        self._emit("state_changed", self.state)

    def start_surface_scan(self) -> None:
        self.state = TestState.SURFACE_SCAN
        self.surface.reset_rotate_capture()
        self._emit("state_changed", self.state)

    def capture_surface_rotation(self, frame: np.ndarray) -> int:
        if self._last_measurement is None:
            return 0
        count = self.surface.add_rotate_capture(frame, self._last_measurement)
        if count >= 8:
            self._surface_result = self.surface.analyze_rotate_capture()
            if self._surface_result:
                self._emit("surface", self._surface_result)
        return count

    def analyze_surface_single_view(self, frame: np.ndarray) -> SurfaceAnalysisResult | None:
        if self._last_measurement is None:
            return None
        self._surface_result = self.surface.analyze_single_view(frame, self._last_measurement)
        self._emit("surface", self._surface_result)
        return self._surface_result

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> tuple[BallMeasurement | None, CompressionSample | None]:
        if not self.calibrator.is_ready():
            return None, None

        measurement = self.detector.detect(frame, timestamp)
        self._last_measurement = measurement
        if measurement is None:
            return None, None

        self._emit("measurement", measurement)
        compression_sample = None

        if self.state == TestState.BASELINE:
            if self.compression.add_baseline_sample(measurement):
                if AUTO_SCALE_ON_BASELINE:
                    self._auto_calibrate_scale_from_baseline()
                self.recovery.set_baseline(self.compression.baseline_mm or 0)
                self.state = TestState.IDLE
                self._emit("state_changed", self.state)
                logger.info("Baseline locked: %.3f mm", self.compression.baseline_mm)

        elif self.state in (TestState.COMPRESSING, TestState.RELEASED, TestState.RECOVERING):
            compression_sample = self.compression.update(measurement, phase=self.state.value)
            self._last_compression = compression_sample
            self._emit("compression", compression_sample)

            if compression_sample:
                pct = compression_sample.compression_pct
                self._peak_compression = max(self._peak_compression, pct)

                if self.state == TestState.COMPRESSING:
                    if pct > COMPRESSION_THRESHOLD_PCT:
                        self._was_compressing = True
                    if self._was_compressing and self._peak_compression >= MIN_COMPRESSION_PEAK_PCT:
                        if pct < self._peak_compression - RELEASE_THRESHOLD_PCT:
                            self._on_release(timestamp)

                elif self.state == TestState.RECOVERING:
                    self.recovery.add_sample(timestamp, compression_sample.diameter_mm)
                    if self._should_finish_recovery(compression_sample, timestamp):
                        self._finish_recovery()

        elif self.state == TestState.SURFACE_SCAN:
            pass  

        return measurement, compression_sample

    def _auto_calibrate_scale_from_baseline(self) -> None:
        
        minor_px = self.compression.baseline_median_minor_px
        if minor_px is None or minor_px <= 0:
            return
        known_mm = REFERENCE_DIAMETERS_MM.get(self.ball_type, 67.0)
        if self.calibrator.calibrate_scale_from_ball_diameter(minor_px, known_mm):
            self.compression.relock_baseline_mm_from_px(self.calibrator)
            measured = self.compression.baseline_diameter_mm or known_mm
            val = self.calibrator.validate_with_measurement(measured, known_mm)
            self._emit("scale_updated", val.error_pct)
            logger.info(
                "Auto-scale from baseline: %.3f px/mm (ref %.1f mm, err %.2f%%)",
                self.calibrator.pixels_per_mm,
                known_mm,
                val.error_pct,
            )

    def _on_release(self, timestamp: float) -> None:
        if self._release_detected:
            return
        self._release_detected = True
        self.state = TestState.RELEASED
        self._emit("state_changed", self.state)
        self.recovery.start_recovery(timestamp)
        self.state = TestState.RECOVERING
        self._emit("state_changed", self.state)
        logger.info("Release detected at t=%.3f", timestamp)

    def _should_finish_recovery(
        self, sample: CompressionSample, timestamp: float
    ) -> bool:
        if self.compression.baseline_mm is None:
            return False
        recovered = abs(sample.diameter_mm - self.compression.baseline_mm) / self.compression.baseline_mm * 100
        if recovered < 0.5 and self.recovery.point_count > 20:
            return True
        if self.recovery.point_count > 180:
            return True
        return False

    def _finish_recovery(self) -> None:
        self._recovery_result = self.recovery.analyze()
        self.state = TestState.DONE
        self._emit("state_changed", self.state)
        if self._recovery_result:
            self._emit("recovery", self._recovery_result)
            logger.info(
                "Recovery: tau=%.3fs t95=%.3fs residual=%.2f%%",
                self._recovery_result.tau_s,
                self._recovery_result.t95_s,
                self._recovery_result.residual_pct,
            )

    def force_finish_recovery(self) -> RecoveryResult | None:
        self._recovery_result = self.recovery.analyze()
        self.state = TestState.DONE
        self._emit("state_changed", self.state)
        if self._recovery_result:
            self._emit("recovery", self._recovery_result)
        return self._recovery_result

    def build_run_summary(self, notes: str = "") -> TestRunSummary | None:
        if self.compression.baseline_mm is None:
            return None
        recovery = self._recovery_result
        return TestRunSummary(
            ball_type=self.ball_type,
            ball_id=self.ball_id,
            baseline_mm=self.compression.baseline_mm,
            max_compression_pct=self.compression.max_compression_pct,
            recovery_tau_s=recovery.tau_s if recovery else None,
            recovery_t95_s=recovery.t95_s if recovery else None,
            residual_pct=recovery.residual_pct if recovery else None,
            accuracy_error_pct=self.calibrator.get_accuracy_error_pct(),
            notes=notes,
            zone_scores=self._surface_result.zones if self._surface_result else [],
            timeseries=self.compression.timeseries,
            recovery=recovery,
        )

    @property
    def last_measurement(self) -> BallMeasurement | None:
        return self._last_measurement

    @property
    def recovery_result(self) -> RecoveryResult | None:
        return self._recovery_result

    @property
    def surface_result(self) -> SurfaceAnalysisResult | None:
        return self._surface_result
