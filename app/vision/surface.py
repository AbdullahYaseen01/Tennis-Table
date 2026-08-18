from __future__ import annotations

import logging
from enum import Enum

import cv2
import numpy as np

from app.config import (
    NUM_SURFACE_ZONES,
    SURFACE_ANNULUS_INNER_RATIO,
    SURFACE_ANNULUS_OUTER_RATIO,
    ZONE_ZSCORE_THRESHOLD,
)
from app.core.models import BallMeasurement, SurfaceAnalysisResult, ZoneResult
from app.vision.calibration import Calibrator

logger = logging.getLogger(__name__)

class SurfaceMode(str, Enum):
    SINGLE_VIEW = "single_view"
    ROTATE_CAPTURE = "rotate_capture"

class SurfaceAnalyzer:
    

    def __init__(self, calibrator: Calibrator) -> None:
        self.calibrator = calibrator
        self.ball_type = "pickleball"
        self._rotate_captures: list[tuple[np.ndarray, BallMeasurement]] = []
        self._rotate_index = 0

    def set_ball_type(self, ball_type: str) -> None:
        self.ball_type = ball_type

    def reset_rotate_capture(self) -> None:
        self._rotate_captures.clear()
        self._rotate_index = 0

    def add_rotate_capture(self, frame: np.ndarray, measurement: BallMeasurement) -> int:
        self._rotate_captures.append((frame.copy(), measurement))
        self._rotate_index = len(self._rotate_captures)
        return self._rotate_index

    @property
    def rotate_capture_count(self) -> int:
        return len(self._rotate_captures)

    def analyze_single_view(
        self, frame: np.ndarray, measurement: BallMeasurement
    ) -> SurfaceAnalysisResult:
        return self._analyze_frame(frame, measurement, mode=SurfaceMode.SINGLE_VIEW)

    def analyze_rotate_capture(self) -> SurfaceAnalysisResult | None:
        if len(self._rotate_captures) < NUM_SURFACE_ZONES:
            return None

        zone_metrics: list[dict[str, float]] = []
        for frame, meas in self._rotate_captures[:NUM_SURFACE_ZONES]:
            result = self._compute_zone_metrics(frame, meas, zone_index_only=0)
            zone_metrics.append(result)

        zones = self._score_zones_from_metrics(
            [{k: v for k, v in m.items() if k != "zone_index"} for m in zone_metrics]
        )
        crack = any(self._detect_cracks(f, m) for f, m in self._rotate_captures[:NUM_SURFACE_ZONES])
        return SurfaceAnalysisResult(zones=zones, mode=SurfaceMode.ROTATE_CAPTURE.value, crack_detected=crack)

    def _analyze_frame(
        self,
        frame: np.ndarray,
        measurement: BallMeasurement,
        mode: SurfaceMode,
    ) -> SurfaceAnalysisResult:
        all_metrics = self._compute_all_zone_metrics(frame, measurement)
        zones = self._score_zones_from_metrics(all_metrics)
        crack = self._detect_cracks(frame, measurement)
        return SurfaceAnalysisResult(
            zones=zones, mode=mode.value, crack_detected=crack
        )

    def _compute_all_zone_metrics(
        self, frame: np.ndarray, measurement: BallMeasurement
    ) -> list[dict[str, float]]:
        metrics = []
        for z in range(NUM_SURFACE_ZONES):
            m = self._compute_zone_metrics(frame, measurement, zone_index_only=z)
            metrics.append({k: v for k, v in m.items() if k != "zone_index"})
        return metrics

    def _compute_zone_metrics(
        self,
        frame: np.ndarray,
        measurement: BallMeasurement,
        zone_index_only: int | None = None,
    ) -> dict[str, float]:
        undist = self.calibrator.undistort(frame)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        cx, cy = measurement.center_px
        radius = measurement.major_px / 2

        sector_size = 360 / NUM_SURFACE_ZONES
        zones_to_process = (
            [zone_index_only] if zone_index_only is not None else range(NUM_SURFACE_ZONES)
        )

        if zone_index_only is not None:
            z = zone_index_only
            start_angle = z * sector_size - measurement.angle
            end_angle = start_angle + sector_size
            fuzz, shape, texture = self._sector_metrics(
                gray, cx, cy, radius, measurement, start_angle, end_angle
            )
            return {
                "zone_index": float(z),
                "fuzz": fuzz,
                "shape": shape,
                "texture": texture,
            }

        result: dict[str, float] = {}
        for z in zones_to_process:
            start_angle = z * sector_size - measurement.angle
            end_angle = start_angle + sector_size
            fuzz, shape, texture = self._sector_metrics(
                gray, cx, cy, radius, measurement, start_angle, end_angle
            )
            result = {"fuzz": fuzz, "shape": shape, "texture": texture}
        return result

    def _sector_metrics(
        self,
        gray: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        measurement: BallMeasurement,
        start_angle: float,
        end_angle: float,
    ) -> tuple[float, float, float]:
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (int(cx), int(cy)),
            (int(radius), int(radius)),
            measurement.angle,
            start_angle,
            end_angle,
            255,
            -1,
        )

        
        inner_r = radius * SURFACE_ANNULUS_INNER_RATIO
        outer_r = radius * SURFACE_ANNULUS_OUTER_RATIO
        annulus = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            annulus,
            (int(cx), int(cy)),
            (int(outer_r), int(outer_r)),
            measurement.angle,
            start_angle,
            end_angle,
            255,
            -1,
        )
        cv2.ellipse(
            annulus,
            (int(cx), int(cy)),
            (int(inner_r), int(inner_r)),
            measurement.angle,
            start_angle,
            end_angle,
            0,
            -1,
        )
        edge_mask = cv2.bitwise_and(annulus, mask)

        
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        edge_pixels = lap[edge_mask > 0]
        fuzz_score = float(np.var(edge_pixels)) if len(edge_pixels) > 0 else 0.0

        
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_pixels = grad_mag[edge_mask > 0]
        if len(grad_pixels) > 0:
            fuzz_score += float(np.mean(grad_pixels))

        
        shape_score = 0.0
        if measurement.refined_edge_points is not None and len(measurement.refined_edge_points) > 0:
            pts = measurement.refined_edge_points.reshape(-1, 2)
            residuals: list[float] = []
            for px, py in pts:
                angle_pt = np.degrees(np.arctan2(py - cy, px - cx)) - measurement.angle
                angle_pt = angle_pt % 360
                sa = start_angle % 360
                ea = end_angle % 360
                in_sector = (sa <= ea and sa <= angle_pt <= ea) or (
                    sa > ea and (angle_pt >= sa or angle_pt <= ea)
                )
                if in_sector:
                    dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                    residuals.append(abs(dist - radius))
            shape_score = float(np.std(residuals)) if len(residuals) > 2 else 0.0

        
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            inner_mask,
            (int(cx), int(cy)),
            (int(inner_r * 0.9), int(inner_r * 0.9)),
            measurement.angle,
            start_angle,
            end_angle,
            255,
            -1,
        )
        inner_pixels = gray[inner_mask > 0]
        texture_score = float(np.std(inner_pixels)) if len(inner_pixels) > 0 else 0.0

        return fuzz_score, shape_score, texture_score

    def _score_zones_from_metrics(
        self, metrics: list[dict[str, float]]
    ) -> list[ZoneResult]:
        fuzz_vals = np.array([m["fuzz"] for m in metrics])
        shape_vals = np.array([m["shape"] for m in metrics])
        texture_vals = np.array([m["texture"] for m in metrics])

        def normalize(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        def z_scores(arr: np.ndarray) -> np.ndarray:
            std = arr.std()
            if std < 1e-9:
                return np.zeros_like(arr)
            return (arr - arr.mean()) / std

        fuzz_n = normalize(fuzz_vals)
        shape_n = normalize(shape_vals)
        texture_n = normalize(texture_vals)

        combined = 0.4 * fuzz_n + 0.35 * shape_n + 0.25 * texture_n
        combined_z = z_scores(combined)
        fuzz_z = z_scores(fuzz_vals)
        shape_z = z_scores(shape_vals)
        texture_z = z_scores(texture_vals)

        zones: list[ZoneResult] = []
        for i in range(len(metrics)):
            
            flagged = (
                abs(combined_z[i]) > ZONE_ZSCORE_THRESHOLD
                or abs(fuzz_z[i]) > ZONE_ZSCORE_THRESHOLD
                or abs(shape_z[i]) > ZONE_ZSCORE_THRESHOLD
                or abs(texture_z[i]) > ZONE_ZSCORE_THRESHOLD
            )
            zones.append(
                ZoneResult(
                    zone_index=i,
                    score=float(combined[i]),
                    flagged=flagged,
                    fuzz_score=float(fuzz_vals[i]),
                    shape_score=float(shape_vals[i]),
                    texture_score=float(texture_vals[i]),
                )
            )
        return zones

    def _detect_cracks(self, frame: np.ndarray, measurement: BallMeasurement) -> bool:
        if self.ball_type != "pickleball":
            return False
        if measurement.contour is None or len(measurement.contour) < 20:
            return False

        contour = measurement.contour.reshape(-1, 2).astype(np.float32)
        
        breaks = 0
        for i in range(len(contour)):
            p0 = contour[(i - 1) % len(contour)]
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            v1 = p1 - p0
            v2 = p2 - p1
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_angle = np.dot(v1, v2) / (n1 * n2)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            if angle > 45:
                breaks += 1
        return breaks > 5

    def artificially_wear_zone(
        self, frame: np.ndarray, measurement: BallMeasurement, zone_index: int
    ) -> np.ndarray:
        
        out = frame.copy()
        cx, cy = float(measurement.center_px[0]), float(measurement.center_px[1])
        radius = measurement.major_px / 2
        sector_size = 360 / NUM_SURFACE_ZONES
        mid_angle = (zone_index + 0.5) * sector_size - measurement.angle
        half_width = sector_size * 0.35
        start, end = mid_angle - half_width, mid_angle + half_width

        
        pts = []
        for deg in np.linspace(start, end, 30):
            rad = np.deg2rad(deg)
            r = radius * 0.82  
            px = int(cx + r * np.cos(rad))
            py = int(cy + r * np.sin(rad))
            pts.append([px, py])
        if len(pts) >= 3:
            cv2.fillPoly(out, [np.array(pts, dtype=np.int32)], (18, 18, 18))

        
        inner = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            inner,
            (int(cx), int(cy)),
            (int(radius * 0.55), int(radius * 0.55)),
            measurement.angle,
            start,
            end,
            255,
            -1,
        )
        out[inner > 0] = (out[inner > 0].astype(np.float32) * 0.35).astype(np.uint8)

        
        band = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            band, (int(cx), int(cy)), (int(radius), int(radius)),
            measurement.angle, start, end, 255, 4,
        )
        blurred = cv2.GaussianBlur(out, (9, 9), 0)
        out[band > 0] = blurred[band > 0]
        return out
