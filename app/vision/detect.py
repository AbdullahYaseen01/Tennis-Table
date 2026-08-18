from __future__ import annotations

import logging

import cv2
import numpy as np

from app.config import (
    COLOR_PROFILES,
    EDGE_INLIER_THRESHOLD_PX,
    MIN_CONTOUR_AREA_PX,
    MORPH_KERNEL_SIZE,
    REST_ECCENTRICITY_MAX,
    ROI_PADDING_PX,
    YOLO_ENABLED,
)
from app.core.models import BallMeasurement
from app.vision.calibration import Calibrator
from app.vision.geometry import (
    eccentricity,
    equivalent_diameter_px,
    fit_ellipse,
    fit_ellipse_from_edges,
    prepare_gray,
)
from app.vision.measurement_fusion import MeasurementFusion
from app.vision.yolo_roi import YoloRoiSeeder

logger = logging.getLogger(__name__)

class BallDetector:
    

    def __init__(self, calibrator: Calibrator) -> None:
        self.calibrator = calibrator
        self.ball_type = "pickleball"
        self.hsv_lower = np.array(COLOR_PROFILES["pickleball"]["hsv_lower"], dtype=np.uint8)
        self.hsv_upper = np.array(COLOR_PROFILES["pickleball"]["hsv_upper"], dtype=np.uint8)
        self._roi: tuple[int, int, int, int] | None = None
        self._last_center: tuple[float, float] | None = None
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        self._diameter_history: list[float] = []
        self._last_undist: np.ndarray | None = None
        self._fusion = MeasurementFusion()
        self._yolo = YoloRoiSeeder() if YOLO_ENABLED else None
        self._use_yolo = bool(self._yolo and self._yolo.available)
        self._yolo_hint_cache = None
        self._yolo_cache_age = 0
        if YOLO_ENABLED and not self._use_yolo:
            logger.info("YOLO ROI disabled — using colour segmentation only")

    def set_ball_type(self, ball_type: str) -> None:
        self.ball_type = ball_type
        profile = COLOR_PROFILES.get(ball_type, COLOR_PROFILES["pickleball"])
        self.hsv_lower = np.array(profile["hsv_lower"], dtype=np.uint8)
        self.hsv_upper = np.array(profile["hsv_upper"], dtype=np.uint8)

    def set_hsv_bounds(
        self, lower: tuple[int, int, int], upper: tuple[int, int, int]
    ) -> None:
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)

    def sample_color_at_point(self, frame: np.ndarray, x: int, y: int, margin: int = 15) -> None:
        h, w = frame.shape[:2]
        x0, x1 = max(0, x - margin), min(w, x + margin)
        y0, y1 = max(0, y - margin), min(h, y + margin)
        patch = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_min, s_min, v_min = hsv.min(axis=(0, 1))
        h_max, s_max, v_max = hsv.max(axis=(0, 1))
        pad_h, pad_s, pad_v = 8, 35, 35
        self.hsv_lower = np.array(
            [max(0, h_min - pad_h), max(0, s_min - pad_s), max(0, v_min - pad_v)],
            dtype=np.uint8,
        )
        self.hsv_upper = np.array(
            [min(179, h_max + pad_h), min(255, s_max + pad_s), min(255, v_max + pad_v)],
            dtype=np.uint8,
        )

    def reset_roi(self) -> None:
        self._roi = None
        self._last_center = None
        self._yolo_hint_cache = None
        self._yolo_cache_age = 0
        self._fusion.reset()

    def _segment(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
        if roi is not None:
            x, y, w, h = roi
            sub = frame[y : y + h, x : x + w]
        else:
            sub = frame

        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        
        if self.ball_type == "tennis":
            lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)
            _, a_channel, _ = cv2.split(lab)
            mask_lab = cv2.inRange(a_channel, 118, 255)
            combined = cv2.bitwise_and(mask, mask_lab)
            if cv2.countNonZero(combined) > cv2.countNonZero(mask) * 0.25:
                mask = combined
        elif self.ball_type.startswith("pickleball"):
            lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)
            _, a_channel, b_channel = cv2.split(lab)
            mask_lab = cv2.bitwise_and(
                cv2.inRange(a_channel, 95, 255),
                cv2.inRange(b_channel, 120, 255),
            )
            combined = cv2.bitwise_and(mask, mask_lab)
            if cv2.countNonZero(combined) > cv2.countNonZero(mask) * 0.20:
                mask = combined

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        if roi is not None:
            full = np.zeros(frame.shape[:2], dtype=np.uint8)
            full[y : y + h, x : x + w] = mask
            return full
        return mask

    def _select_contour(
        self, contours: list, frame_area: int, hint_center: tuple[float, float] | None
    ) -> np.ndarray | None:
        if not contours:
            return None
        scored: list[tuple[float, np.ndarray]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA_PX:
                continue
            if area > frame_area * 0.6:
                continue
            score = area
            if hint_center is not None:
                m = cv2.moments(c)
                if m["m00"] > 0:
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]
                    dist = np.hypot(cx - hint_center[0], cy - hint_center[1])
                    score = area / (1.0 + dist * 0.05)
            scored.append((score, c))
        if not scored:
            return None
        return max(scored, key=lambda x: x[0])[1]

    def detect(self, frame: np.ndarray, timestamp: float) -> BallMeasurement | None:
        undist = self.calibrator.undistort(frame)
        self._last_undist = undist
        gray, grad_mag = prepare_gray(undist)
        frame_area = undist.shape[0] * undist.shape[1]

        yolo_hint = None
        yolo_roi = None
        h, w = undist.shape[:2]
        min_ball_px = min(h, w) * 0.08

        if self._use_yolo and self._yolo is not None:
            if self._yolo_cache_age <= 0 or self._yolo_hint_cache is None:
                self._yolo_hint_cache = self._yolo.detect_roi(undist)
                self._yolo_cache_age = 30
            self._yolo_cache_age -= 1
            yolo_hint = self._yolo_hint_cache
            if yolo_hint is not None and yolo_hint.radius >= min_ball_px:
                yolo_roi = self._yolo.roi_box(yolo_hint, undist.shape, padding=1.4)

        roi = self._roi if self._roi else yolo_roi
        mask = self._segment(undist, roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        hint = self._last_center
        if yolo_hint is not None:
            hint = yolo_hint.center

        contour = self._select_contour(list(contours), frame_area, hint)
        if contour is None:
            
            if roi is not None:
                mask = self._segment(undist, None)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour = self._select_contour(list(contours), frame_area, hint)
            if contour is None:
                self.reset_roi()
                return None

        ellipse = fit_ellipse(contour)
        if ellipse is None:
            return None

        center, axes, angle = ellipse
        cx, cy = float(center[0]), float(center[1])
        major_px, minor_px = float(max(axes)), float(min(axes))

        refined, edge_pts, weights = fit_ellipse_from_edges(
            gray, grad_mag, (cx, cy), (major_px, minor_px), float(angle), mask=mask, two_pass=True
        )
        if refined is not None:
            center, axes, angle = refined
            cx, cy = float(center[0]), float(center[1])
            major_px, minor_px = float(max(axes)), float(min(axes))
            contour = edge_pts
        else:
            edge_pts = np.array([], dtype=np.float32).reshape(-1, 1, 2)
            weights = np.array([])

        ecc = eccentricity(major_px, minor_px)

        if major_px < min_ball_px or minor_px < min_ball_px * 0.5:
            self.reset_roi()
            return None

        
        if ecc <= REST_ECCENTRICITY_MAX:
            major_px, minor_px = self._fusion.apply_px(major_px, minor_px)

        major_mm = self.calibrator.px_to_mm(major_px)
        minor_mm = self.calibrator.px_to_mm(minor_px)

        pad = ROI_PADDING_PX
        h, w = undist.shape[:2]
        x0 = max(0, int(cx - major_px / 2 - pad))
        y0 = max(0, int(cy - major_px / 2 - pad))
        x1 = min(w, int(cx + major_px / 2 + pad))
        y1 = min(h, int(cy + major_px / 2 + pad))
        self._roi = (x0, y0, x1 - x0, y1 - y0)
        self._last_center = (cx, cy)

        area = cv2.contourArea(contour) if contour is not None else np.pi * (major_px / 2) ** 2
        confidence = float(np.clip(area / (np.pi * (major_px / 2) ** 2 + 1e-6), 0.0, 1.0))
        if len(edge_pts) >= 8 and len(weights) > 0:
            edge_quality = min(1.0, float(np.mean(weights)) / 50.0)
            confidence = float(np.clip(0.5 * confidence + 0.5 * edge_quality, 0.0, 1.0))

        meas = BallMeasurement(
            timestamp=timestamp,
            center_px=(cx, cy),
            major_mm=major_mm,
            minor_mm=minor_mm,
            angle=float(angle),
            eccentricity=ecc,
            refined_edge_points=edge_pts if len(edge_pts) > 0 else contour,
            detection_confidence=confidence,
            major_px=major_px,
            minor_px=minor_px,
            contour=contour,
        )
        self._diameter_history.append(minor_mm)
        return meas

    def measure_median_diameter_px(
        self, frames: list[np.ndarray], axis: str = "minor"
    ) -> float | None:
        
        values: list[float] = []
        for frame in frames:
            undist = self.calibrator.undistort(frame)
            gray, grad_mag = prepare_gray(undist)
            mask = self._segment(undist, None)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = self._select_contour(list(contours), undist.shape[0] * undist.shape[1], None)
            if contour is None:
                continue
            ellipse = fit_ellipse(contour)
            if ellipse is None:
                continue
            _, axes, _ = ellipse
            major_px, minor_px = float(max(axes)), float(min(axes))
            cx, cy = float(ellipse[0][0]), float(ellipse[0][1])
            refined, _, _ = fit_ellipse_from_edges(
                gray, grad_mag, (cx, cy), (major_px, minor_px), float(ellipse[2]), mask=mask
            )
            if refined is not None:
                _, axes, _ = refined
                major_px, minor_px = float(max(axes)), float(min(axes))
            if axis == "minor":
                values.append(minor_px)
            elif axis == "equiv":
                values.append(equivalent_diameter_px(major_px, minor_px))
            else:
                values.append(major_px)
        if len(values) < 5:
            return None
        arr = np.array(values)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-6
        inliers = arr[np.abs(arr - med) <= 2.5 * 1.4826 * mad]
        return float(np.median(inliers)) if len(inliers) > 0 else med

    def resting_diameter_std(self, window: int = 30) -> float:
        recent = self._diameter_history[-window:]
        if len(recent) < 2:
            return 0.0
        return float(np.std(recent))

    def draw_overlay(
        self, frame: np.ndarray, measurement: BallMeasurement | None
    ) -> np.ndarray:
        out = self._last_undist.copy() if self._last_undist is not None else self.calibrator.undistort(frame)
        if measurement is None:
            return out

        cx, cy = measurement.center_px
        cv2.ellipse(
            out,
            (int(cx), int(cy)),
            (int(measurement.major_px / 2), int(measurement.minor_px / 2)),
            measurement.angle,
            0,
            360,
            (0, 255, 0),
            2,
        )
        if measurement.refined_edge_points is not None and len(measurement.refined_edge_points) > 0:
            for pt in measurement.refined_edge_points[::12]:
                cv2.circle(out, (int(pt[0][0]), int(pt[0][1])), 2, (0, 200, 255), -1)

        yolo_tag = "YOLO+CV" if self._use_yolo else "CV"
        label = (
            f"[{yolo_tag}] D: {measurement.minor_mm:.2f}/{measurement.major_mm:.2f} mm  "
            f"ecc: {measurement.eccentricity:.3f}  conf: {measurement.detection_confidence:.2f}"
        )
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return out
