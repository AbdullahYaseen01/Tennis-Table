"""Camera intrinsics, scale calibration, and accuracy validation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import (
    CALIB_DIR,
    CHECKERBOARD_COLS,
    CHECKERBOARD_ROWS,
    CHECKERBOARD_SQUARE_MM,
    INTRINSICS_PATH,
    SCALE_PATH,
    VALIDATION_PATH,
)
from app.core.models import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    pixels_per_mm: float
    map1: np.ndarray | None = None
    map2: np.ndarray | None = None
    image_size: tuple[int, int] | None = None


class Calibrator:
    """Lens intrinsics + real-world scale at the ball plane."""

    def __init__(self) -> None:
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.pixels_per_mm: float | None = None
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None
        self._image_size: tuple[int, int] | None = None
        self._object_points_template = self._build_object_points()
        self._intrinsic_frames: list[np.ndarray] = []
        self._last_validation: ValidationResult | None = None
        self.load()

    @staticmethod
    def _build_object_points() -> np.ndarray:
        objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
        objp *= CHECKERBOARD_SQUARE_MM
        return objp

    def load(self) -> bool:
        loaded = False
        if INTRINSICS_PATH.exists():
            data = np.load(INTRINSICS_PATH)
            self.camera_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]
            if "image_size" in data:
                self._image_size = tuple(data["image_size"].tolist())
            loaded = True
            logger.info("Loaded intrinsics from %s", INTRINSICS_PATH)
        if SCALE_PATH.exists():
            data = np.load(SCALE_PATH)
            self.pixels_per_mm = float(data["pixels_per_mm"])
            loaded = True
            logger.info("Loaded scale: %.3f px/mm", self.pixels_per_mm)
        if VALIDATION_PATH.exists():
            data = np.load(VALIDATION_PATH)
            self._last_validation = ValidationResult(
                measured_mm=float(data["measured_mm"]),
                known_mm=float(data["known_mm"]),
                error_mm=float(data["error_mm"]),
                error_pct=float(data["error_pct"]),
            )
        return loaded

    def is_ready(self) -> bool:
        return (
            self.camera_matrix is not None
            and self.dist_coeffs is not None
            and self.pixels_per_mm is not None
            and self.pixels_per_mm > 0
        )

    def get_validation(self) -> ValidationResult | None:
        return self._last_validation

    def get_accuracy_error_pct(self) -> float:
        if self._last_validation:
            return self._last_validation.error_pct
        return float("nan")

    def reset_intrinsic_capture(self) -> None:
        self._intrinsic_frames = []

    def add_intrinsic_frame(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), None
        )
        if not found:
            return False
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        self._intrinsic_frames.append(corners)
        return True

    def intrinsic_frame_count(self) -> int:
        return len(self._intrinsic_frames)

    def compute_intrinsics(self, image_size: tuple[int, int]) -> bool:
        if len(self._intrinsic_frames) < 10:
            logger.warning("Need at least 10 checkerboard frames, have %d", len(self._intrinsic_frames))
            return False

        objpoints = [self._object_points_template for _ in self._intrinsic_frames]
        imgpoints = self._intrinsic_frames
        h, w = image_size[1], image_size[0]
        ret, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), None, None
        )
        if not ret:
            return False

        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self._image_size = (w, h)
        self._build_remap((w, h))

        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            INTRINSICS_PATH,
            camera_matrix=mtx,
            dist_coeffs=dist,
            image_size=np.array([w, h]),
        )
        logger.info("Saved intrinsics to %s", INTRINSICS_PATH)
        return True

    def _build_remap(self, image_size: tuple[int, int]) -> None:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return
        w, h = image_size
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_mtx, (w, h), cv2.CV_16SC2
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame
        h, w = frame.shape[:2]
        if self._map1 is None or self._image_size != (w, h):
            self._build_remap((w, h))
        if self._map1 is not None and self._map2 is not None:
            return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def detect_checkerboard(self, frame: np.ndarray) -> tuple[bool, np.ndarray | None]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pattern_size = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not found:
            # OpenCV 4.x fast SB detector when available
            sb = getattr(cv2, "findChessboardCornersSB", None)
            if sb is not None:
                found, corners = sb(gray, pattern_size, flags)
        if not found or corners is None:
            return False, None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners

    def draw_checkerboard(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        out = frame.copy()
        cv2.drawChessboardCorners(
            out, (CHECKERBOARD_COLS, CHECKERBOARD_ROWS), corners, True
        )
        return out

    def compute_scale_from_checkerboard(self, frame: np.ndarray) -> float | None:
        """Compute pixels-per-mm from checkerboard at the ball plane."""
        found, corners = self.detect_checkerboard(frame)
        if not found or corners is None:
            return None

        # Average adjacent corner distances in pixels
        corners_2d = corners.reshape(-1, 2)
        dists: list[float] = []
        for row in range(CHECKERBOARD_ROWS):
            for col in range(CHECKERBOARD_COLS - 1):
                i = row * CHECKERBOARD_COLS + col
                j = i + 1
                dists.append(float(np.linalg.norm(corners_2d[j] - corners_2d[i])))
        for col in range(CHECKERBOARD_COLS):
            for row in range(CHECKERBOARD_ROWS - 1):
                i = row * CHECKERBOARD_COLS + col
                j = i + CHECKERBOARD_COLS
                dists.append(float(np.linalg.norm(corners_2d[j] - corners_2d[i])))

        if not dists:
            return None
        dists_arr = np.array(dists)
        med = np.median(dists_arr)
        mad = np.median(np.abs(dists_arr - med)) + 1e-6
        inliers = dists_arr[np.abs(dists_arr - med) <= 3.0 * 1.4826 * mad]
        avg_px = float(np.mean(inliers)) if len(inliers) > 0 else float(med)
        ppm = avg_px / CHECKERBOARD_SQUARE_MM
        return ppm

    def save_scale(self, pixels_per_mm: float) -> None:
        self.pixels_per_mm = pixels_per_mm
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(SCALE_PATH, pixels_per_mm=pixels_per_mm)
        logger.info("Saved scale: %.3f px/mm", pixels_per_mm)

    def calibrate_scale_from_frame(self, frame: np.ndarray) -> bool:
        undist = self.undistort(frame)
        ppm = self.compute_scale_from_checkerboard(undist)
        if ppm is None:
            return False
        self.save_scale(ppm)
        return True

    def calibrate_scale_from_ball_diameter(
        self,
        diameter_px: float,
        known_diameter_mm: float,
    ) -> bool:
        """Set px/mm from a sub-pixel ball measurement and known spec diameter."""
        if diameter_px <= 0 or known_diameter_mm <= 0:
            return False
        ppm = diameter_px / known_diameter_mm
        self.save_scale(ppm)
        logger.info(
            "Scale from ball: %.3f px/mm (%.2f px / %.2f mm)",
            ppm,
            diameter_px,
            known_diameter_mm,
        )
        return True

    def validate_with_measurement(
        self, measured_mm: float, known_diameter_mm: float
    ) -> ValidationResult:
        error_mm = measured_mm - known_diameter_mm
        error_pct = (error_mm / known_diameter_mm) * 100.0
        result = ValidationResult(
            measured_mm=measured_mm,
            known_mm=known_diameter_mm,
            error_mm=error_mm,
            error_pct=error_pct,
        )
        self._last_validation = result
        np.savez(
            VALIDATION_PATH,
            measured_mm=measured_mm,
            known_mm=known_diameter_mm,
            error_mm=error_mm,
            error_pct=error_pct,
        )
        return result

    def validate_known_diameter(
        self, frame: np.ndarray, known_diameter_mm: float, detector=None
    ) -> ValidationResult | None:
        """Validate using the same sub-pixel detector pipeline when available."""
        undist = self.undistort(frame)
        if detector is not None:
            import time

            m = detector.detect(undist, time.perf_counter())
            if m is not None and self.pixels_per_mm:
                measured_mm = (m.major_mm + m.minor_mm) / 2.0
                return self.validate_with_measurement(measured_mm, known_diameter_mm)

        diameter_px = self._measure_circle_diameter_px_legacy(undist)
        if diameter_px is None or self.pixels_per_mm is None:
            return None
        measured_mm = diameter_px / self.pixels_per_mm
        return self.validate_with_measurement(measured_mm, known_diameter_mm)

    def _measure_circle_diameter_px_legacy(
        self, frame: np.ndarray, center: tuple[float, float] | None = None
    ) -> float | None:
        """Measure diameter of a circular object using edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        if center is None:
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=50,
                param1=80,
                param2=40,
                minRadius=20,
                maxRadius=500,
            )
            if circles is None:
                return None
            c = circles[0][0]
            return float(c[2] * 2)
        # Radial edge sampling at given center
        cx, cy = center
        h, w = gray.shape
        radii: list[float] = []
        for angle in np.linspace(0, 2 * np.pi, 180, endpoint=False):
            dx, dy = np.cos(angle), np.sin(angle)
            prev_val = gray[int(cy), int(cx)]
            for r in range(10, min(w, h) // 2):
                x = int(cx + r * dx)
                y = int(cy + r * dy)
                if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
                    break
                val = float(gray[y, x])
                if abs(val - prev_val) > 20:
                    radii.append(float(r))
                    break
                prev_val = val
        if len(radii) < 30:
            return None
        return float(np.median(radii) * 2)

    def px_to_mm(self, px: float) -> float:
        if self.pixels_per_mm is None or self.pixels_per_mm <= 0:
            return px
        return px / self.pixels_per_mm

    def get_data(self) -> CalibrationData | None:
        if not self.is_ready():
            return None
        assert self.camera_matrix is not None
        assert self.dist_coeffs is not None
        assert self.pixels_per_mm is not None
        return CalibrationData(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            pixels_per_mm=self.pixels_per_mm,
            map1=self._map1,
            map2=self._map2,
            image_size=self._image_size,
        )
