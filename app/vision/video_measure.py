from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.ball_profiles import floor_contact_y_frac, get_profile, reference_diameter_mm
from app.config import REFERENCE_DIAMETERS_MM
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

COMP_ASPECT_MIN = 1.16
COMP_ASPECT_STRONG = 1.28
COMP_ROUNDNESS_MAX = 0.87
COMP_REPORT_MIN_PCT = 8.0
IMPACT_STATUS_MIN_PCT = 12.0

class ContactPhaseTracker:
    

    def __init__(self, frame_h: int, *, floor_frac: float = FLOOR_CONTACT_Y_FRAC) -> None:
        self.frame_h = frame_h
        self.floor_frac = floor_frac
        self._cy_window: list[float] = []

    def on_floor(self, cy: float) -> bool:
        
        return cy >= self.frame_h * self.floor_frac

    def in_contact(self, cy: float) -> bool:
        
        if not self.on_floor(cy):
            return False
        
        if self._cy_window:
            ref = float(np.median(self._cy_window[-5:]))
            if abs(cy - ref) > 70:
                return True  
        self._cy_window.append(cy)
        if len(self._cy_window) > CONTACT_PEAK_WINDOW:
            self._cy_window.pop(0)
        
        arr = np.asarray(self._cy_window, dtype=np.float64)
        peak = float(np.percentile(arr, 90)) if len(arr) >= 4 else float(arr.max())
        if cy < peak - REBOUND_DROP_PX:
            
            return cy >= self.frame_h * (self.floor_frac + 0.02)
        if cy < self.frame_h * (self.floor_frac + 0.06) and len(self._cy_window) >= 3:
            recent_vel = self._cy_window[-1] - self._cy_window[-3]
            if recent_vel > 18:
                return False
        return True

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
    

    def __init__(self, frame_h: int, fps: float, *, floor_frac: float = FLOOR_CONTACT_Y_FRAC) -> None:
        self.frame_h = frame_h
        self.floor_frac = floor_frac
        self.min_gap = max(BOUNCE_MIN_GAP_FRAMES, int(fps * 1.4))
        self.bounces = 0
        self._last_bounce = -9999
        self._in_floor = False
        self._peak_cy = 0.0
        self._peak_frame = -1
        self._armed = False
        self._min_cy = 1e9

    def _real_drop(self) -> bool:
        
        return (self._peak_cy - self._min_cy) >= self.frame_h * 0.22

    def update(self, frame_idx: int, cy: float, *, reliable: bool) -> int:
        if reliable:
            self._min_cy = min(self._min_cy, cy)
        floor_band = self.frame_h * self.floor_frac
        if not reliable or cy < floor_band:
            if (
                self._in_floor
                and self._armed
                and self._real_drop()
                and self._peak_cy >= self.frame_h * 0.90
                and self._peak_cy - cy > 55
                and self._peak_frame - self._last_bounce >= self.min_gap
            ):
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
            if (
                self._real_drop()
                and self._peak_cy >= self.frame_h * 0.90
                and self._peak_cy - cy > 55
                and self._peak_frame - self._last_bounce >= self.min_gap
            ):
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
        self.known_mm = reference_diameter_mm(ball_type)
        self._floor_frac = floor_contact_y_frac(ball_type)
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
        self._last_floor_shape: tuple[float, float, float, int] | None = None  
        self._floor_peak_cy: float = 0.0

    @property
    def yolo_active(self) -> bool:
        return self._use_yolo

    def _clean_ball_mask(
        self, frame: np.ndarray, cx: float, cy: float, radius: float
    ) -> tuple[np.ndarray | None, int, int]:
        
        profile = get_profile(self.ball_type)
        h, w = frame.shape[:2]
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(radius)):
            return None, 0, 0
        pad = int(max(radius * 1.25, 90))
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = min(w, int(cx + pad)), min(h, int(cy + pad))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None, x0, y0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lo = profile.hsv_lower.copy()
        hi = profile.hsv_upper.copy()
        lo[1] = max(int(lo[1]), 100)
        lo[2] = max(int(lo[2]), 130)
        yellow = cv2.inRange(hsv, lo, hi)
        
        skin = cv2.inRange(hsv, (0, 25, 50), (25, 160, 255))
        skin |= cv2.inRange(hsv, (160, 25, 50), (180, 160, 255))
        yellow = cv2.bitwise_and(yellow, cv2.bitwise_not(skin))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k)
        
        k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k_fill, iterations=2)
        return yellow, x0, y0

    def _hand_dent_pct(self, frame: np.ndarray, cx: float, cy: float, radius: float) -> float:
        
        mask, x0, y0 = self._clean_ball_mask(frame, cx, cy, radius)
        if mask is None:
            return 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0
        cx_l, cy_l = float(cx - x0), float(cy - y0)
        best, best_score = None, -1e18
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 800:
                continue
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            dist = float(np.hypot(m["m10"] / m["m00"] - cx_l, m["m01"] / m["m00"] - cy_l))
            if dist > radius * 0.65:
                continue
            peri = cv2.arcLength(c, True) + 1e-6
            circ = 4.0 * np.pi * area / (peri * peri)
            score = area * circ - dist * 40.0
            if score > best_score:
                best_score, best = score, c
        if best is None:
            return 0.0

        (ex, ey), R = cv2.minEnclosingCircle(best)
        if R < 40:
            return 0.0
        pts = best.reshape(-1, 2).astype(np.float64)
        dist = np.sqrt((pts[:, 0] - ex) ** 2 + (pts[:, 1] - ey) ** 2)
        deficit = np.maximum(0.0, float(R) - dist)
        score = float(np.percentile(deficit, 60)) / float(R) * 100.0
        
        pad = int(max(radius * 1.2, 80))
        x0b, y0b = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1b, y1b = min(frame.shape[1], int(cx + pad)), min(frame.shape[0], int(cy + pad))
        crop = frame[y0b:y1b, x0b:x1b]
        skin_dent = 0.0
        if crop.size:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            skin = cv2.inRange(hsv, (0, 30, 50), (25, 170, 255))
            skin |= cv2.inRange(hsv, (160, 30, 50), (180, 170, 255))
            yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
            disk = (xx - (cx - x0b)) ** 2 + (yy - (cy - y0b)) ** 2 <= (radius * 0.92) ** 2
            tot = int(disk.sum())
            if tot > 200:
                frac = float(skin[disk].astype(bool).sum()) / float(tot)
                
                if frac >= 0.065:
                    skin_dent = min(32.0, 11.0 + (frac - 0.065) * 160.0)

        
        shape_dent = 0.0
        if score >= 9.0:
            shape_dent = 10.0 + (score - 9.0) * 3.5
        dent = max(shape_dent, skin_dent)
        if dent < 11.0:
            return 0.0
        return max(0.0, min(35.0, dent))

    def _mask_visible_height(self, frame: np.ndarray, cx: float, cy: float, radius: float) -> float | None:
        profile = get_profile(self.ball_type)
        h, w = frame.shape[:2]
        pad = int(max(radius * 1.15, 90))
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = min(w, int(cx + pad)), min(h, int(cy + pad))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, profile.hsv_lower, profile.hsv_upper)
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
        if get_profile(self.ball_type).use_convex_hull:
            best = cv2.convexHull(best)
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
        
        if (
            self._last_cy is not None
            and self._last_cy > self._frame_h * 0.52
            and cy < self._frame_h * 0.48
        ):
            if self._track_hint is not None:
                return self._track_hint[0], self._track_hint[1]
            return self._last_cx or cx, self._last_cy
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
            if self._last_cy > self._frame_h * 0.52 and cy < self._frame_h * 0.48:
                score -= 80.0
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

    def _yellow_ball_lock(
        self, frame: np.ndarray
    ) -> tuple[float, float, float, float] | None:
        
        profile = get_profile(self.ball_type)
        h, w = frame.shape[:2]
        
        pads: list[tuple[int, int, int, int]] = []
        if (
            self._last_cx is not None
            and self._last_cy is not None
            and np.isfinite(self._last_cx)
            and np.isfinite(self._last_cy)
        ):
            bpx = self.baseline_vertical_px
            if bpx is None or not np.isfinite(bpx) or bpx < 20:
                bpx = 220.0
            rad = int(bpx * 0.85)
            x0 = max(0, int(self._last_cx) - rad)
            y0 = max(0, int(self._last_cy) - rad)
            x1 = min(w, int(self._last_cx) + rad)
            y1 = min(h, int(self._last_cy) + rad)
            pads.append((x0, y0, x1, y1))
        pads.append((0, 0, w, h))  

        lo = profile.hsv_lower.copy()
        hi = profile.hsv_upper.copy()
        lo[1] = max(int(lo[1]), 110)
        lo[2] = max(int(lo[2]), 140)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        last = (self._last_cx, self._last_cy) if self._last_cx is not None else None
        area_base = self.baseline_vertical_px
        if area_base is None or not np.isfinite(area_base) or area_base < 20:
            area_base = 200.0
        min_area = area_base ** 2 * 0.15

        for x0, y0, x1, y1 in pads:
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lo, hi)
            skin = cv2.inRange(hsv, (0, 30, 40), (25, 170, 255))
            skin |= cv2.inRange(hsv, (160, 30, 40), (180, 170, 255))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best = None
            best_score = -1e18
            for c in contours:
                area = float(cv2.contourArea(c))
                if area < max(900.0, min_area * 0.35):
                    continue
                peri = cv2.arcLength(c, True) + 1e-6
                circ = 4.0 * np.pi * area / (peri * peri)
                if circ < 0.42:
                    continue
                (x, y), (bw, bh), _ang = cv2.minAreaRect(c)
                x += x0
                y += y0
                maj, mnr = max(bw, bh), min(bw, bh)
                if maj < 40 or mnr < 35:
                    continue
                if self.baseline_vertical_px:
                    if maj > self.baseline_vertical_px * 1.55 or maj < self.baseline_vertical_px * 0.50:
                        continue
                    if mnr < self.baseline_vertical_px * 0.42:
                        continue
                score = area * circ
                
                if self._frame_h > 0 and y > self._frame_h * 0.90:
                    score -= 8000.0
                if last is not None:
                    jump = float(np.hypot(x - last[0], y - last[1]))
                    if jump > 90:
                        score -= (jump - 90) * 120.0
                    else:
                        score += (90 - jump) * 4.0
                if score > best_score:
                    best_score = score
                    best = (float(x), float(y), float(maj), float(mnr))
            if best is not None:
                return best
        return None

    def _from_colour(self, frame: np.ndarray) -> VideoFrameResult | None:
        hand_press_mode = self.bounce_counter is None and self.ball_type.startswith("pickleball")
        hint = None if hand_press_mode else self._yolo_hint(frame)

        
        if hand_press_mode:
            ylock = self._yellow_ball_lock(frame)
            if ylock is not None:
                cx, cy, major, minor = ylock
                if self._last_cx is not None and self._last_cy is not None:
                    jump = float(np.hypot(cx - self._last_cx, cy - self._last_cy))
                    if jump > 120:
                        
                        cx, cy = self._last_cx, self._last_cy
                        near = self._colour_at(frame, (cx, cy, max(major, minor) * 0.55))
                        shape_n = self._extract_shape(near)
                        if shape_n:
                            cx, cy, major, minor, angle, _ = shape_n
                        else:
                            angle = 0.0
                    else:
                        angle = 0.0
                else:
                    angle = 0.0
                bm = self._colour_at(frame, (cx, cy, max(major, minor) * 0.55))
                
                if bm.get("detected"):
                    bcx = float(bm.get("cx") or cx)
                    bcy = float(bm.get("cy") or cy)
                    if np.hypot(bcx - cx, bcy - cy) < max(major, minor) * 0.35:
                        cx, cy = bcx, bcy
                        major = float(bm.get("major_px") or major)
                        minor = float(bm.get("minor_px") or minor)
                picked = (bm, cx, cy, major, minor, angle)
            else:
                picked = self._pick_detection(frame, None)
        else:
            picked = self._pick_detection(frame, hint)

        if picked is None:
            return None
        bm, cx, cy, major, minor, angle = picked

        if not hand_press_mode:
            cx, cy = self._snap_to_yolo(cx, cy, hint)
            prev_cx, prev_cy = cx, cy
            cx, cy = self._temporal_guard(cx, cy, hint)
            if (cx, cy) != (prev_cx, prev_cy) and self._track_hint is not None:
                bm2 = self._colour_at(frame, self._track_hint)
                shape2 = self._extract_shape(bm2)
                if shape2:
                    cx2, cy2, major2, minor2, angle2, _ = shape2
                    if cy2 > self._frame_h * 0.48 and 25 < cx2 < self._frame_w - 25:
                        bm, cx, cy, major, minor, angle = bm2, cx2, cy2, major2, minor2, angle2
                    elif self._last_cx is not None and self._last_cy is not None:
                        cx, cy = self._last_cx, self._last_cy
            if cx < 40 or cx > self._frame_w - 40:
                if self._track_hint is not None:
                    bm_fix = self._colour_at(frame, self._track_hint)
                    shape_fix = self._extract_shape(bm_fix)
                    if shape_fix:
                        cx_f, cy_f, major_f, minor_f, angle_f, _ = shape_fix
                        if 40 < cx_f < self._frame_w - 40:
                            bm, cx, cy, major, minor, angle = bm_fix, cx_f, cy_f, major_f, minor_f, angle_f
                        else:
                            cx, cy = self._track_hint[0], self._track_hint[1]
                    else:
                        cx, cy = self._track_hint[0], self._track_hint[1]
                elif self._last_cx is not None and self._last_cy is not None:
                    cx, cy = self._last_cx, self._last_cy
            if hint is not None and not self._position_trusted(cx, cy, hint):
                cx, cy = hint.center

        radius = (hint.radius if hint else max(major, minor) * 0.5)
        if self._contact is None and self._frame_h > 0:
            self._contact = ContactPhaseTracker(self._frame_h, floor_frac=self._floor_frac)
        on_floor = self._contact.on_floor(cy) if self._contact else False
        in_contact = self._contact.in_contact(cy) if self._contact else False
        if on_floor:
            self._floor_peak_cy = max(self._floor_peak_cy, cy)
        else:
            self._floor_peak_cy = 0.0
        
        rebounding = (
            on_floor
            and self._floor_peak_cy > self._frame_h * 0.72
            and (self._floor_peak_cy - cy) > 40
        )
        
        
        hand_press_mode = self.bounce_counter is None and self.ball_type.startswith("pickleball")
        hand_held = hand_press_mode or (
            self.ball_type.startswith("pickleball") and not on_floor
        )
        measure_comp = on_floor or in_contact or hand_held

        colour_major = float(bm.get("major_px") or major)
        colour_minor = float(bm.get("minor_px") or minor)
        colour_aspect = colour_major / max(colour_minor, 1.0)
        colour_roundness = float(bm.get("roundness") or (colour_minor / max(colour_major, 1.0)))
        live_colour_ok = (
            self.baseline_vertical_px is not None
            and colour_major >= self.baseline_vertical_px * 0.70
            and colour_minor >= self.baseline_vertical_px * 0.55
            and colour_aspect <= 2.20
        )

        
        if measure_comp and self.baseline_vertical_px and not live_colour_ok:
            recovered = False
            for roi in (self._track_hint, None):
                bm_fix = self._colour_at(frame, roi)
                shape_fix = self._extract_shape(bm_fix)
                if not shape_fix:
                    continue
                cx_f, cy_f, maj_f, min_f, ang_f, _ = shape_fix
                asp_f = maj_f / max(min_f, 1.0)
                if (
                    maj_f >= self.baseline_vertical_px * 0.70
                    and min_f >= self.baseline_vertical_px * 0.55
                    and asp_f <= 2.20
                    and 40 < cx_f < self._frame_w - 40
                ):
                    bm, cx, cy, major, minor, angle = bm_fix, cx_f, cy_f, maj_f, min_f, ang_f
                    colour_major, colour_minor = maj_f, min_f
                    colour_aspect = asp_f
                    live_colour_ok = True
                    on_floor = self._contact.on_floor(cy) if self._contact else on_floor
                    in_contact = self._contact.in_contact(cy) if self._contact else in_contact
                    measure_comp = on_floor or in_contact
                    recovered = True
                    break
            held_shape = False
            if (
                not recovered
                and self._last_floor_shape is not None
                and self._frame_idx - self._last_floor_shape[3] <= 2
            ):
                colour_major, colour_minor, angle, _ = self._last_floor_shape
                major, minor = colour_major, colour_minor
                colour_aspect = colour_major / max(colour_minor, 1.0)
                held_shape = True
                
        else:
            held_shape = False

        if held_shape:
            colour_load = vertical_load_px(colour_major, colour_minor, angle)
        else:
            colour_load = float(
                bm.get("load_axis_px") or vertical_load_px(colour_major, colour_minor, angle)
            )

        edge_pts = None
        if not self._fast:
            edge_pts, major_r, minor_r, angle_r = self._refine_shape(frame, cx, cy, major, minor, angle)
            refine_ok = (
                minor_r >= colour_minor * 0.78
                and major_r <= max(colour_major, major) * 1.35
                and (
                    not self.baseline_vertical_px
                    or minor_r >= self.baseline_vertical_px * 0.72
                )
            )
            if measure_comp and refine_ok:
                major, minor, angle = major_r, minor_r, angle_r

        vload = colour_load
        if vload <= 0 or (self.baseline_vertical_px and vload >= self.baseline_vertical_px * 0.98):
            vload = visible_height_px(edge_pts, major, minor, angle)

        aspect_gate = COMP_ASPECT_STRONG if rebounding else COMP_ASPECT_MIN
        roundness_max = COMP_ROUNDNESS_MAX
        report_min = COMP_REPORT_MIN_PCT
        
        display_major = colour_major if colour_major > 20 else (self.baseline_vertical_px or major)
        display_minor = colour_minor if colour_minor > 20 else display_major
        display_angle = angle

        size_ok_colour = (
            self.baseline_vertical_px is not None
            and colour_major >= self.baseline_vertical_px * 0.70
            and colour_minor >= self.baseline_vertical_px * 0.55
        )
        
        
        clearly_squashed = (
            (not hand_held)
            and measure_comp
            and size_ok_colour
            and colour_aspect >= aspect_gate
            and colour_roundness <= roundness_max
        )

        if clearly_squashed and self.baseline_vertical_px:
            aspect_vload = self.baseline_vertical_px * (colour_minor / colour_major)
            candidates = [vload]
            if aspect_vload < self.baseline_vertical_px:
                candidates.append(aspect_vload)
            if colour_minor < self.baseline_vertical_px * 0.95:
                candidates.append(colour_minor)
            vload = min(candidates) if colour_aspect < 1.45 else float(np.median(candidates))
            mask_h = self._mask_visible_height(frame, cx, cy, radius)
            if (
                mask_h is not None
                and self.baseline_vertical_px * 0.58 <= mask_h < self.baseline_vertical_px * 0.95
            ):
                vload = min(vload, mask_h)
            vload = max(vload, self.baseline_vertical_px * 0.58)
            
            display_major = max(colour_major, self.baseline_vertical_px)
            display_minor = max(40.0, min(colour_minor, vload))
            display_angle = 0.0
            if live_colour_ok:
                self._last_floor_shape = (colour_major, colour_minor, 0.0, self._frame_idx)
        elif self.baseline_vertical_px:
            vload = self.baseline_vertical_px
            if size_ok_colour:
                
                display_major = colour_major
                display_minor = colour_minor
                display_angle = angle
            else:
                display_major = display_minor = self.baseline_vertical_px
                display_angle = 0.0

        major, minor, angle = display_major, display_minor, display_angle
        rnd = minor / max(major, 1.0)
        ecc = float(np.sqrt(max(0.0, 1.0 - rnd * rnd)))

        trusted = self._position_trusted(cx, cy, hint)
        size_ok = (
            self.baseline_vertical_px is None
            or size_ok_colour
            or (0.55 * self.baseline_vertical_px <= max(major, minor) <= 1.35 * self.baseline_vertical_px)
            or measure_comp
        )
        trusted = trusted and size_ok

        raw_comp = 0.0
        if self.baseline_vertical_px and self.baseline_vertical_px > 0 and clearly_squashed and trusted:
            raw_comp = max(0.0, (self.baseline_vertical_px - vload) / self.baseline_vertical_px * 100.0)
            if raw_comp > 42.0:
                raw_comp = 40.0
            if raw_comp < report_min:
                raw_comp = 0.0
        if raw_comp == 0.0 and self.baseline_vertical_px:
            vload = self.baseline_vertical_px
            ecc = float(np.sqrt(max(0.0, 1.0 - (display_minor / max(display_major, 1.0)) ** 2)))

        
        if hand_held and trusted and self.baseline_vertical_px:
            dent = self._hand_dent_pct(frame, cx, cy, self.baseline_vertical_px * 0.5)
            if dent >= 12.0:
                raw_comp = dent
                vload = self.baseline_vertical_px * (1.0 - raw_comp / 100.0)
                display_major = self.baseline_vertical_px
                display_minor = max(40.0, vload)
                display_angle = 0.0
                major, minor = display_major, display_minor
                ecc = float(np.sqrt(max(0.0, 1.0 - (minor / max(major, 1.0)) ** 2)))
            else:
                raw_comp = 0.0
                vload = self.baseline_vertical_px
                display_major = display_minor = (
                    colour_major if size_ok_colour else self.baseline_vertical_px
                )
                if size_ok_colour:
                    display_minor = colour_minor
                display_angle = 0.0 if not size_ok_colour else angle
                major, minor = display_major, display_minor
                ecc = float(np.sqrt(max(0.0, 1.0 - (minor / max(major, 1.0)) ** 2)))

        
        if raw_comp > 0 and self._px_per_mm:
            dia_mm = vload / self._px_per_mm
        else:
            dia_mm = self.known_mm
        if trusted and 40 < cx < self._frame_w - 40:
            self._track_hint = (cx, cy, max(display_major, display_minor) * 0.5)

        return VideoFrameResult(
            detected=bool(bm.get("detected")) and trusted,
            status="compressing" if raw_comp >= IMPACT_STATUS_MIN_PCT else ("tracking" if trusted else "partial"),
            cx=cx, cy=cy,
            major_px=display_major, minor_px=display_minor, angle=display_angle,
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
            ellipse={"cx": cx, "cy": cy, "a": display_major / 2, "b": display_minor / 2, "angle": display_angle},
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
        self.bounce_counter = BounceCounter(self._frame_h, self._fps, floor_frac=self._floor_frac)
        self._contact = ContactPhaseTracker(self._frame_h, floor_frac=self._floor_frac)
        logger.info("Baseline locked: %.1f px (%.3f px/mm)", bpx, ppm)
        return True

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None) -> VideoFrameResult:
        if self._frame_h == 0:
            self._frame_h, self._frame_w = frame.shape[:2]
            if self.bounce_counter is None:
                self.bounce_counter = BounceCounter(self._frame_h, self._fps, floor_frac=self._floor_frac)
            if self._contact is None:
                self._contact = ContactPhaseTracker(self._frame_h, floor_frac=self._floor_frac)

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

        hand_press_mode = self.bounce_counter is None and self.ball_type.startswith("pickleball")
        if self.bounce_counter and r.major_px > 10:
            hint = self._yolo_hint(frame)
            track_cy = r.cy
            if hint is not None and r.cy < self._frame_h * 0.45 and hint.center[1] > self._frame_h * 0.55:
                track_cy = hint.center[1]
            reliable = r.confidence >= 0.65 and track_cy > self._frame_h * self._floor_frac
            self.bounce_counter.update(self._frame_idx, track_cy, reliable=reliable)

        if r.major_px > 10 and r.cx > 0 and (
            hand_press_mode or (r.confidence >= 0.65 and r.diameter_mm > self.known_mm * 0.45)
        ):
            if (
                np.isfinite(r.cx)
                and np.isfinite(r.cy)
                and 25 < r.cx < self._frame_w - 25
                and 25 < r.cy < self._frame_h - 25
            ):
                self._last_cy = r.cy
                self._last_cx = r.cx

        self._frame_idx += 1
        return r

    @property
    def bounce_count(self) -> int:
        return self.bounce_counter.bounces if self.bounce_counter else 0
