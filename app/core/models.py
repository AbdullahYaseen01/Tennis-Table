from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

class TestState(str, Enum):
    IDLE = "IDLE"
    BASELINE = "BASELINE"
    COMPRESSING = "COMPRESSING"
    RELEASED = "RELEASED"
    RECOVERING = "RECOVERING"
    DONE = "DONE"
    SURFACE_SCAN = "SURFACE_SCAN"

@dataclass
class BallMeasurement:
    timestamp: float
    center_px: tuple[float, float]
    major_mm: float
    minor_mm: float
    angle: float
    eccentricity: float
    refined_edge_points: np.ndarray
    detection_confidence: float
    major_px: float = 0.0
    minor_px: float = 0.0
    contour: np.ndarray | None = None

@dataclass
class CompressionSample:
    timestamp: float
    compression_pct: float
    bulge_mm: float
    diameter_mm: float
    phase: str = ""

@dataclass
class RecoveryResult:
    tau_s: float
    t95_s: float
    residual_pct: float
    d_final_mm: float
    fit_confidence: float
    fitted_curve: tuple[np.ndarray, np.ndarray] | None = None
    raw_points: list[tuple[float, float]] = field(default_factory=list)

@dataclass
class ZoneResult:
    zone_index: int
    score: float
    flagged: bool
    fuzz_score: float = 0.0
    shape_score: float = 0.0
    texture_score: float = 0.0

@dataclass
class SurfaceAnalysisResult:
    zones: list[ZoneResult]
    mode: str  
    crack_detected: bool = False

@dataclass
class ValidationResult:
    measured_mm: float
    known_mm: float
    error_mm: float
    error_pct: float

@dataclass
class TestRunSummary:
    ball_type: str
    ball_id: str
    baseline_mm: float
    max_compression_pct: float
    recovery_tau_s: float | None
    recovery_t95_s: float | None
    residual_pct: float | None
    accuracy_error_pct: float
    notes: str = ""
    zone_scores: list[ZoneResult] = field(default_factory=list)
    timeseries: list[CompressionSample] = field(default_factory=list)
    recovery: RecoveryResult | None = None

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "ball_type": self.ball_type,
            "ball_id": self.ball_id,
            "baseline_mm": self.baseline_mm,
            "max_compression_pct": self.max_compression_pct,
            "recovery_tau_s": self.recovery_tau_s,
            "recovery_t95_s": self.recovery_t95_s,
            "residual_pct": self.residual_pct,
            "accuracy_error_pct": self.accuracy_error_pct,
            "notes": self.notes,
        }
