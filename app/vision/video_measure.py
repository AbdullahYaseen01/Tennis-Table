"""Unified high-accuracy video measurement — colour detect + YOLO ROI + sub-pixel refine."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import COLOR_PROFILES, REFERENCE_DIAMETERS_MM
from app.vision.bounce_measure import analyze_ball_in_frame, compute_baseline_from_video
from app.vision.calibration import Calibrator
from app.vision.detect import BallDetector
from app.vision.geometry import fit_ellipse_from_edges, prepare_gray
from app.vision.yolo_roi import RoiHint, YoloRoiSeeder

logger = logging.getLogger(__name__)

FLOOR_Y_FRAC = 0.52
FLOOR_CONTACT_Y_FRAC = 0.73
BOUNCE_MIN_GAP_FRAMES = 45
REBOUND_DROP_PX = 28
CONTACT_PEAK_WINDOW = 18


class ContactPhaseTracker:
    """Detect real floor contact — bottom of frame and at descent peak, not in-air fall."""

    def __init__(self, frame_h: int) -> None:
        self.frame_h = frame_h
        self._cy_window: list[float] = []

    def in_contact(self, cy: float) -> bool:
        floor_y = self.frame_h * FLOOR_CONTACT_Y_FRAC
        if cy < floor_y:
            return False
        self._cy_window.append(cy)
        if len(self._cy_window) > CONTACT_PEAK_WINDOW:
            self._cy_window.pop(0)
        peak = max(self._cy_window)
        return cy >= peak - REBOUND_DROP_PX


@dataclass
class VideoFrameResult:
    detected: bool = False
    status: str = "searching"
    cx: float = 0.0
    cy: float = 0.0
    major_px: float = 0.0
    minor_px: float = 0.0
    angle: float = 0.0
    vertical_load_px: float = 0.0
    diameter_mm: float = 0.0
    compression_pct: float = 0.0
    raw_compression_pct: float = 0.0
    confidence: float = 0.0
    eccentricity: float = 0.0
    baseline_locked: bool = False
    baseline_mm: float | None = None
    edge_points: np.ndarray | None = None
    yolo_active: bool = False
    ellipse: dict | None = None


def vertical_load_px(major_px: float, minor_px: float, angle: float) -> float:
    a, b = major_px / 2.0, minor_px / 2.0
    theta = np.deg2rad(angle)
    return 2.0 * float(np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2))


def visible_height_px(edge_points: np.ndarray | None, major_px: float, minor_px: float, angle: float) -> float:
    vload = vertical_load_px(major_px, minor_px, angle)
    if edge_points is None or len(edge_points) < 8:
        return vload
    pts = edge_points.reshape(-1, 2)
    span = float(pts[:, 1].max() - pts[:, 1].min())
    if span < 20:
        return vload
    return min(vload, span)


class BounceCounter:
    """Count one bounce per floor contact using deepest Y then rebound."""

    def __init__(self, frame_h: int, fps: float) -> None:
        self.frame_h = frame_h
        self.min_gap = max(BOUNCE_MIN_GAP_FRAMES, int(fps * 1.4))
        self.bounces = 0
        self._last_bounce = -9999
        self._in_floor = False
        self._peak_cy = 0.0
        self._peak_frame = -1
        self._armed = False

    def update(self, frame_idx: int, cy: float, *, reliable: bool) -> int:
        floor_band = self.frame_h * FLOOR_CONTACT_Y_FRAC
        if not reliable or cy < floor_band:
            if self._in_floor and self._armed and self._peak_cy - cy > 18:
                if self._peak_frame - self._last_bounce >= self.min_gap:
                    self.bounces += 1
                    self._last_bounce = self._peak_frame
            self._in_floor = False
            self._armed = False
            return self.bounces

        if not self._in_floor:
            self._in_floor = True
            self._peak_cy = cy
            self._peak_frame = frame_idx
            self._armed = True
            return self.bounces

        if cy >= self._peak_cy - 2:
            self._peak_cy = max(self._peak_cy, cy)
            self._peak_frame = frame_idx
        elif self._armed and self._peak_cy - cy > 22:
            if self._peak_frame - self._last_bounce >= self.min_gap:
                self.bounces += 1
                self._last_bounce = self._peak_frame
            self._armed = False
        return self.bounces


class VideoBallAnalyzer:
    def __init__(
        self,
        ball_type: str = "tennis",
        *,
        use_yolo: bool = True,
        fast_mode: bool = False,
    ) -> None:
        from app.config import IS_VERCEL

        self.ball_type = ball_type
        self.known_mm = REFERENCE_DIAMETERS_MM.get(ball_type, 67.0)
        self._fast = fast_mode or IS_VERCEL
        self.calibrator = Calibrator()
        self.detector = BallDetector(self.calibrator)
        self.detector.set_ball_type(ball_type)
        use_yolo = use_yolo and not self._fast
        self._yolo = YoloRoiSeeder() if use_yolo else None
        self._use_yolo = bool(self._yolo and self._yolo.available)
        self.baseline_locked = False
        self.baseline_vertical_px: float | None = None
        self._frame_h = 0
        self._frame_w = 0
        self._fps = 30.0
        self._frame_idx = 0
        self._px_per_mm: float | None = None
        self._track_hint: tuple[float, float, float] | None = None
        self._yolo_cache: RoiHint | None = None
        self._yolo_age = 0
        self.bounce_counter: BounceCounter | None = None
        self._contact: ContactPhaseTracker | None = None
        self._last_cy: float | None = None
        self._last_cx: float | None = None

    @property
    def yolo_active(self) -> bool:
        return self._use_yolo

    def _mask_visible_height(self, frame: np.ndarray, cx: float, cy: float, radius: float) -> float | None:
        profile = COLOR_PROFILES.get(self.ball_type, COLOR_PROFILES["tennis"])
        h, w = frame.shape[:2]
        pad = int(max(radius * 1.15, 90))
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = min(w, int(cx + pad)), min(h, int(cy + pad))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(profile["hsv_lower"], dtype=np.uint8),
            np.array(profile["hsv_upper"], dtype=np.uint8),
        )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cx_l, cy_l = cx - x0, cy - y0
        best, best_d = None, 1e9
        for c in contours:
            if cv2.contourArea(c) < 180:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            d = np.hypot(m["m10"] / m["m00"] - cx_l, m["m01"] / m["m00"] - cy_l)
            if d < best_d:
                best_d, best = d, c
        if best is None:
            return None
        pts = best.reshape(-1, 2)
        cy_l = cy - y0
        upper = pts[pts[:, 1] <= cy_l + radius * 0.15]
        lower = pts[pts[:, 1] >= cy_l - radius * 0.05]
        if len(upper) < 4 or len(lower) < 4:
            top = float(pts[:, 1].min())
            floor = float(pts[:, 1].max())
        else:
            top = float(upper[:, 1].min())
            floor = float(lower[:, 1].max())
        height = floor - top
        return height if height >= 35 else None

    def _temporal_guard(self, cx: float, cy: float, hint: RoiHint | None) -> tuple[float, float]:
        if self._frame_h <= 0:
            return cx, cy
        if self._last_cy is not None and self._last_cy > self._frame_h * 0.58 and cy < self._frame_h * 0.25:
            if hint is not None:
                return hint.center[0], hint.center[1]
            return self._last_cx or cx, self._last_cy
        if (
            self._last_cx is not None
            and self._frame_w > 0
            and self._last_cy is not None
            and self._last_cy > self._frame_h * 0.58
            and abs(cx - self._last_cx) > self._frame_w * 0.18
        ):
            if hint is not None:
                return hint.center[0], hint.center[1]
            return self._last_cx, self._last_cy
        return cx, cy

    def _yolo_hint(self, frame: np.ndarray, *, force: bool = False) -> RoiHint | None:
        if not self._use_yolo or self._yolo is None:
            return None
        in_air = self._last_cy is None or self._last_cy < self._frame_h * 0.62
        refresh = force or in_air or self._yolo_age <= 0 or self._yolo_cache is None
        if refresh:
            self._yolo_cache = self._yolo.detect_roi(frame)
            self._yolo_age = 3 if in_air else 8
        self._yolo_age -= 1
        return self._yolo_cache

    def _extract_shape(self, bm: dict) -> tuple[float, float, float, float, float, dict | None] | None:
        ell = bm.get("ellipse")
        if not bm.get("detected") and not ell:
            return None
        cx = float(bm.get("cx") or (ell["cx"] if ell else 0))
        cy = float(bm.get("cy") or (ell["cy"] if ell else 0))
        major = float(bm.get("major_px") or (ell["a"] * 2 if ell else 0))
        minor = float(bm.get("minor_px") or (ell["b"] * 2 if ell else 0))
        angle = float(ell["angle"] if ell else 0)
        if major < 10 or minor < 10:
            return None
        return cx, cy, major, minor, angle, ell

    def _score_candidate(
        self,
        bm: dict,
        cx: float,
        cy: float,
        major: float,
        minor: float,
        hint: RoiHint | None,
    ) -> float:
        load = float(bm.get("load_axis_px") or vertical_load_px(major, minor, float(bm.get("ellipse", {}).get("angle", 0))))
        score = 0.0
        if self.baseline_vertical_px and self.baseline_vertical_px > 0:
            ratio = load / self.baseline_vertical_px
            if 0.78 <= ratio <= 1.10:
                score += 45.0
            elif 0.62 <= ratio <= 1.18:
                score += 22.0
            elif ratio < 0.45 or ratio > 1.35:
                score -= 40.0
        rnd = float(bm.get("roundness") or (minor / max(major, 1.0)))
        score += rnd * 35.0
        if bm.get("detected"):
            score += 12.0
        if hint is not None:
            dist = np.hypot(cx - hint.center[0], cy - hint.center[1])
            if dist <= hint.radius * 0.95:
                score += 28.0 * (1.0 - dist / max(hint.radius, 1.0))
            else:
                score -= 20.0
        if self._last_cx is not None and self._last_cy is not None:
            jump = np.hypot(cx - self._last_cx, cy - self._last_cy)
            if jump <= 90:
                score += 30.0 * (1.0 - jump / 90.0)
            elif jump <= 180:
                score += 8.0
            else:
                score -= 35.0
        if cy < 25 or cy > self._frame_h - 25:
            score -= 60.0
        if cx < 20 or cx > self._frame_w - 20:
            score -= 25.0
        return score

    def _pick_detection(self, frame: np.ndarray, hint: RoiHint | None) -> tuple[dict, float, float, float, float, float] | None:
        candidates: list[tuple[float, dict, float, float, float, float, float]] = []
        full = self._colour_at(frame, None)
        shape = self._extract_shape(full)
        if shape:
            cx, cy, major, minor, angle, _ = shape
            candidates.append((self._score_candidate(full, cx, cy, major, minor, hint), full, cx, cy, major, minor, angle))

        roi_sources: list[tuple[float, float, float] | None] = []
        if hint is not None:
            roi_sources.append((hint.center[0], hint.center[1], hint.radius * 1.05))
        if self._track_hint is not None:
            roi_sources.append(self._track_hint)
        seen = set()
        for roi in roi_sources:
            if roi is None or roi in seen:
                continue
            seen.add(roi)
            bm = self._colour_at(frame, roi)
            shape = self._extract_shape(bm)
            if shape:
                cx, cy, major, minor, angle, _ = shape
                candidates.append((self._score_candidate(bm, cx, cy, major, minor, hint), bm, cx, cy, major, minor, angle))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, bm, cx, cy, major, minor, angle = candidates[0]
        if best_score < 8.0 and len(candidates) > 1:
            _, bm, cx, cy, major, minor, angle = candidates[1]
        return bm, cx, cy, major, minor, angle

    def _colour_at(self, frame: np.ndarray, roi_hint: tuple[float, float, float] | None) -> dict:
        return analyze_ball_in_frame(
            frame,
            ball_type=self.ball_type,
            baseline_minor_px=self.baseline_vertical_px,
            px_per_mm=self._px_per_mm,
            roi_hint=roi_hint,
        )

    def _position_trusted(self, cx: float, cy: float, hint: RoiHint | None) -> bool:
        if self._frame_h <= 0:
            return True
        if hint is None:
            return 20 < cy < self._frame_h - 20
        return np.hypot(cx - hint.center[0], cy - hint.center[1]) <= hint.radius * 0.75

    def _snap_to_yolo(self, cx: float, cy: float, hint: RoiHint | None) -> tuple[float, float]:
        if hint is None or self._frame_h <= 0:
            return cx, cy
        yc = hint.center[1]
        dist = np.hypot(cx - hint.center[0], cy - hint.center[1])
        colour_on_floor = cy > self._frame_h * 0.55
        yolo_on_floor = yc > self._frame_h * 0.55
        if colour_on_floor != yolo_on_floor or dist > hint.radius * 0.55:
            return hint.center[0], hint.center[1]
        if dist > hint.radius * 0.25:
            return 0.4 * cx + 0.6 * hint.center[0], 0.4 * cy + 0.6 * hint.center[1]
        return cx, cy

    def _refine_shape(
        self, frame: np.ndarray, cx: float, cy: float, major: float, minor: float, ang: float,
    ) -> tuple[np.ndarray | None, float, float, float]:
        try:
            gray, grad = prepare_gray(frame)
            refined, pts, _ = fit_ellipse_from_edges(
                gray, grad, (cx, cy), (major, minor), ang, two_pass=True,
            )
            if refined is None or pts is None or len(pts) < 8:
                return None, major, minor, ang
            (rx, ry), (ea, eb), rang = refined
            if np.hypot(rx - cx, ry - cy) > max(major, minor) * 0.2:
                return pts, major, minor, ang
            return pts, float(max(ea, eb)), float(min(ea, eb)), float(rang)
        except Exception:
            return None, major, minor, ang

    def _from_colour(self, frame: np.ndarray) -> VideoFrameResult | None:
        hint = self._yolo_hint(frame)
        picked = self._pick_detection(frame, hint)
        if picked is None:
            return None
        bm, cx, cy, major, minor, angle = picked

        cx, cy = self._snap_to_yolo(cx, cy, hint)
        cx, cy = self._temporal_guard(cx, cy, hint)
        if hint is not None and not self._position_trusted(cx, cy, hint):
            cx, cy = hint.center

        radius = hint.radius if hint else max(major, minor) * 0.5
        if self._contact is None and self._frame_h > 0:
            self._contact = ContactPhaseTracker(self._frame_h)
        in_contact = self._contact.in_contact(cy) if self._contact else False

        edge_pts = None
        if not self._fast:
            edge_pts, major_r, minor_r, angle_r = self._refine_shape(frame, cx, cy, major, minor, angle)
            if in_contact and minor_r >= max(major, minor) * 0.35:
                major, minor, angle = major_r, minor_r, angle_r

        vload = float(bm.get("load_axis_px") or 0.0)
        if vload <= 0 or (self.baseline_vertical_px and vload >= self.baseline_vertical_px * 0.98):
            vload = visible_height_px(edge_pts, major, minor, angle)

        if in_contact:
            mask_h = self._mask_visible_height(frame, cx, cy, radius)
            if mask_h is not None:
                vload = min(vload, mask_h)
            if (
                not self._fast
                and self.baseline_vertical_px
                and vload >= self.baseline_vertical_px * 0.90
            ):
                ts = self._frame_idx / max(self._fps, 1.0)
                det = self.detector.detect(frame, ts)
                if det is not None:
                    dc = det.center_px
                    if np.hypot(dc[0] - cx, dc[1] - cy) <= radius * 0.9:
                        det_v = vertical_load_px(
                            det.major_px or det.major_mm / max(self._px_per_mm or 1, 1e-6),
                            det.minor_px or det.minor_mm / max(self._px_per_mm or 1, 1e-6),
                            det.angle,
                        )
                        if det_v > 20:
                            vload = min(vload, det_v)
                        if det.refined_edge_points is not None and len(det.refined_edge_points) >= 8:
                            edge_pts = det.refined_edge_points
            if bm.get("detected") and bm.get("minor_px"):
                colour_minor = float(bm["minor_px"])
                if colour_minor < vload:
                    vload = colour_minor
            if self.baseline_vertical_px:
                major = self.baseline_vertical_px
                minor = max(40.0, min(minor, vload))
                angle = 0.0
        elif self.baseline_vertical_px:
            major = minor = self.baseline_vertical_px
            angle = 0.0
            vload = self.baseline_vertical_px
            edge_pts = None

        rnd = minor / max(major, 1.0)
        ecc = float(np.sqrt(max(0.0, 1.0 - rnd * rnd)))

        trusted = self._position_trusted(cx, cy, hint)
        size_ok = (
            self.baseline_vertical_px is None
            or (0.55 * self.baseline_vertical_px <= vload <= 1.20 * self.baseline_vertical_px)
            or in_contact
        )
        trusted = trusted and size_ok

        raw_comp = 0.0
        if self.baseline_vertical_px and self.baseline_vertical_px > 0 and in_contact and trusted:
            raw_comp = max(0.0, (self.baseline_vertical_px - vload) / self.baseline_vertical_px * 100.0)
            if raw_comp > 42.0:
                raw_comp = 40.0
            if raw_comp > 5.0 and ecc < 0.45:
                raw_comp = 0.0
                minor = major = self.baseline_vertical_px
                vload = self.baseline_vertical_px

        dia_mm = vload / self._px_per_mm if self._px_per_mm else self.known_mm
        if trusted:
            self._track_hint = (cx, cy, max(major, minor) * 0.5)

        return VideoFrameResult(
            detected=bool(bm.get("detected")) and trusted,
            status="compressing" if raw_comp > 8 else ("tracking" if trusted else "partial"),
            cx=cx, cy=cy,
            major_px=major, minor_px=minor, angle=angle,
            vertical_load_px=vload,
            diameter_mm=dia_mm if trusted else self.known_mm,
            compression_pct=raw_comp,
            raw_compression_pct=raw_comp,
            confidence=0.88 if bm.get("detected") and trusted else (0.72 if trusted else 0.45),
            eccentricity=ecc,
            baseline_locked=self.baseline_locked,
            baseline_mm=self.known_mm if self.baseline_locked else None,
            edge_points=edge_pts,
            yolo_active=self._use_yolo,
            ellipse={"cx": cx, "cy": cy, "a": major / 2, "b": minor / 2, "angle": angle},
        )

    def lock_baseline_from_video(self, video_path: str) -> bool:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        bpx, ppm = compute_baseline_from_video(cap, ball_type=self.ball_type)
        cap.release()
        if bpx <= 0:
            return False
        self.baseline_vertical_px = bpx
        self._px_per_mm = ppm
        self.calibrator.pixels_per_mm = ppm
        self.baseline_locked = True
        self.bounce_counter = BounceCounter(self._frame_h, self._fps)
        self._contact = ContactPhaseTracker(self._frame_h)
        logger.info("Baseline locked: %.1f px (%.3f px/mm)", bpx, ppm)
        return True

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None) -> VideoFrameResult:
        if self._frame_h == 0:
            self._frame_h, self._frame_w = frame.shape[:2]
            if self.bounce_counter is None:
                self.bounce_counter = BounceCounter(self._frame_h, self._fps)
            if self._contact is None:
                self._contact = ContactPhaseTracker(self._frame_h)

        r = self._from_colour(frame)
        if r is None:
            self._frame_idx += 1
            return VideoFrameResult(
                detected=False,
                status="searching",
                baseline_locked=self.baseline_locked,
                baseline_mm=self.known_mm if self.baseline_locked else None,
                yolo_active=self._use_yolo,
            )

        hint = self._yolo_hint(frame)
        track_cy = r.cy
        if hint is not None and r.cy < self._frame_h * 0.45 and hint.center[1] > self._frame_h * 0.55:
            track_cy = hint.center[1]
        reliable = r.confidence >= 0.65 and track_cy > self._frame_h * FLOOR_CONTACT_Y_FRAC
        if self.bounce_counter and r.major_px > 10:
            self.bounce_counter.update(self._frame_idx, track_cy, reliable=reliable)

        if r.major_px > 10 and r.confidence >= 0.65 and r.diameter_mm > self.known_mm * 0.45:
            self._last_cy = r.cy
            self._last_cx = r.cx

        self._frame_idx += 1
        return r

    @property
    def bounce_count(self) -> int:
        return self.bounce_counter.bounces if self.bounce_counter else 0
