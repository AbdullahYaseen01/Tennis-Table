"""High-accuracy ball detection and compression measurement for web + export.

Includes low-quality video pipeline: upscale, denoise, multi-strategy detection,
sub-pixel edge refinement, and temporal smoothing.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from app.vision.geometry import fit_ellipse_from_edges, prepare_gray

KNOWN_DIAMETER_MM = 67.0
GOLF_DIAMETER_MM = 42.7

BASELINE_TARGET_FRAMES = 60
MIN_BASELINE_GOOD_FRAMES = 40
MIN_BASELINE_LOW_QUALITY = 22
BASELINE_OUTLIER_SIGMA = 2.5
COMPRESSION_MEDIAN_WINDOW = 5
TEMPORAL_HEIGHT_WINDOW = 7
REST_ROUNDNESS_MIN = 0.82
REST_ROUNDNESS_GOLF = 0.68
REST_ASPECT_MIN = 0.88
REST_ASPECT_MAX = 1.12
REST_ASPECT_GOLF_MIN = 0.65
REST_ASPECT_GOLF_MAX = 1.35

LOW_QUALITY_MAX_WIDTH = 520
LOW_QUALITY_MAX_PIXELS = 350_000
ENHANCE_TARGET_WIDTH = 640


def _roundness_min(ball_type: str) -> float:
    return REST_ROUNDNESS_GOLF if ball_type in ("golf", "white") else REST_ROUNDNESS_MIN


def _aspect_bounds(ball_type: str) -> tuple[float, float]:
    if ball_type in ("golf", "white"):
        return REST_ASPECT_GOLF_MIN, REST_ASPECT_GOLF_MAX
    return REST_ASPECT_MIN, REST_ASPECT_MAX


COLOR_PROFILES: dict[str, dict] = {
    "tennis": {
        "hsv_lower": np.array([18, 45, 70]),
        "hsv_upper": np.array([48, 255, 255]),
        "diameter_mm": KNOWN_DIAMETER_MM,
        "use_lab": True,
    },
    "golf": {
        "hsv_lower": np.array([0, 0, 140]),
        "hsv_upper": np.array([180, 90, 255]),
        "diameter_mm": GOLF_DIAMETER_MM,
        "use_lab": False,
    },
    "white": {
        "hsv_lower": np.array([0, 0, 140]),
        "hsv_upper": np.array([180, 90, 255]),
        "diameter_mm": KNOWN_DIAMETER_MM,
        "use_lab": False,
    },
}

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_LG = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))


def assess_frame_quality(width: int, height: int) -> dict:
    pixels = width * height
    low = width < LOW_QUALITY_MAX_WIDTH or pixels < LOW_QUALITY_MAX_PIXELS
    scale = 1.0
    if low and width > 0:
        scale = min(3.0, max(1.5, ENHANCE_TARGET_WIDTH / width))
    return {
        "low_quality": low,
        "enhance_scale": scale,
        "width": width,
        "height": height,
    }


def _robust_median(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) + 1e-6
    inliers = arr[np.abs(arr - med) <= BASELINE_OUTLIER_SIGMA * 1.4826 * mad]
    return float(np.median(inliers)) if len(inliers) >= 3 else med


def _min_ball_px(frame_w: int, low_quality: bool = False) -> float:
    base = max(18.0, min(150.0, frame_w * 0.10))
    return base * 0.65 if low_quality else base


def _min_baseline_frames(frame_w: int) -> int:
    if frame_w < LOW_QUALITY_MAX_WIDTH:
        return MIN_BASELINE_LOW_QUALITY
    return MIN_BASELINE_GOOD_FRAMES


def enhance_frame(frame: np.ndarray, quality: dict | None = None) -> tuple[np.ndarray, float]:
    """Upscale, denoise, and boost contrast for low-quality sources."""
    h, w = frame.shape[:2]
    q = quality or assess_frame_quality(w, h)
    scale = q["enhance_scale"] if q["low_quality"] else 1.0
    out = frame

    if scale > 1.01:
        nw, nh = int(w * scale), int(h * scale)
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_CUBIC)

    if q["low_quality"]:
        out = cv2.bilateralFilter(out, d=5, sigmaColor=35, sigmaSpace=35)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        ch, cs, cv = cv2.split(hsv)
        cv = _clahe_strong.apply(cv)
        out = cv2.cvtColor(cv2.merge([ch, cs, cv]), cv2.COLOR_HSV2BGR)
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)
    else:
        out = _preprocess(out)

    return out, scale


def _preprocess(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = _clahe.apply(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def _circularity(contour: np.ndarray) -> float:
    a = cv2.contourArea(contour)
    p = cv2.arcLength(contour, True)
    if p <= 0:
        return 0.0
    return float(4.0 * np.pi * a / (p * p))


def _build_masks(proc: np.ndarray, profile: dict, ball_type: str) -> list[np.ndarray]:
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    masks: list[np.ndarray] = []

    m1 = cv2.inRange(hsv, profile["hsv_lower"], profile["hsv_upper"])
    masks.append(m1)

    if profile.get("use_lab"):
        lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)
        _, a_ch, _ = cv2.split(lab)
        masks.append(cv2.bitwise_and(m1, cv2.inRange(a_ch, 110, 255)))

    # Relaxed saturation for faded / compressed video
    loose = profile["hsv_lower"].copy()
    loose[1] = max(0, int(loose[1]) - 30)
    loose[2] = max(0, int(loose[2]) - 25)
    masks.append(cv2.inRange(hsv, loose, profile["hsv_upper"]))

    if ball_type in ("golf", "white") or ball_type == "tennis":
        _, _, v = cv2.split(hsv)
        bright = cv2.inRange(v, 120, 255)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        _, adapt = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(cv2.bitwise_and(bright, adapt))

    return masks


def _best_contour_from_mask(mask: np.ndarray, min_area: float, frame_w: int, frame_h: int | None = None):
    fh = frame_h or max(mask.shape[0], 1)
    max_area = frame_w * fh * 0.22
    max_dim = max(frame_w, fh) * 0.45

    k = KERNEL_LG if frame_w < LOW_QUALITY_MAX_WIDTH else KERNEL
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 2 or h < 2 or w > max_dim or h > max_dim:
            continue
        circ = _circularity(c)
        if circ < 0.35:
            continue
        aspect = min(w, h) / max(w, h)
        edge_penalty = 0.45 if (x <= 2 or y <= 2 or (x + w) >= (frame_w - 2)) else 1.0
        score = a * circ * aspect * edge_penalty
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None
    x, y, w, h = cv2.boundingRect(best)
    return float(x + w / 2), float(y + h / 2), float(w), float(h), best


def detect_on_frame(
    proc: np.ndarray,
    *,
    ball_type: str = "tennis",
    min_area: float = 1000.0,
    frame_w: int | None = None,
) -> tuple[float, float, float, float, np.ndarray] | None:
    """Run mask pipeline on an already-enhanced frame."""
    fw = frame_w or proc.shape[1]
    low_q = fw < LOW_QUALITY_MAX_WIDTH or (proc.shape[0] * fw) < LOW_QUALITY_MAX_PIXELS
    profile = COLOR_PROFILES.get(ball_type, COLOR_PROFILES["tennis"])
    eff_min = max(200.0, min_area * (0.5 if low_q else 1.0))

    best_hit = None
    best_score = -1.0
    for mask in _build_masks(proc, profile, ball_type):
        hit = _best_contour_from_mask(mask, eff_min, fw, proc.shape[0])
        if hit is None:
            continue
        _, _, _, _, contour = hit
        score = cv2.contourArea(contour) * _circularity(contour)
        if score > best_score:
            best_score = score
            best_hit = hit

    if best_hit is None and ball_type == "tennis":
        white = COLOR_PROFILES["golf"]
        for mask in _build_masks(proc, white, "golf"):
            hit = _best_contour_from_mask(mask, max(150.0, eff_min * 0.4), fw, proc.shape[0])
            if hit is not None:
                return hit
    return best_hit


def detect(
    frame: np.ndarray,
    *,
    ball_type: str = "tennis",
    min_area: float = 1000.0,
    quality: dict | None = None,
) -> tuple[float, float, float, float, np.ndarray] | None:
    h, w = frame.shape[:2]
    q = quality or assess_frame_quality(w, h)
    proc, _ = enhance_frame(frame, q)
    return detect_on_frame(proc, ball_type=ball_type, min_area=min_area, frame_w=proc.shape[1])


def _scale_detection(
    hit: tuple[float, float, float, float, np.ndarray], inv_scale: float,
) -> tuple[float, float, float, float, np.ndarray]:
    cx, cy, bw, bh, contour = hit
    contour = (contour.astype(np.float64) * inv_scale).astype(np.int32)
    return cx * inv_scale, cy * inv_scale, bw * inv_scale, bh * inv_scale, contour


def measure_ellipse(contour: np.ndarray) -> dict | None:
    if contour is None or len(contour) < 5:
        return None
    try:
        (ex, ey), (ea, eb), ang = cv2.fitEllipse(contour)
    except cv2.error:
        return None

    major = float(max(ea, eb))
    minor = float(min(ea, eb))
    return {
        "cx": float(ex), "cy": float(ey),
        "major_px": major, "minor_px": minor,
        "angle": float(ang),
        "roundness": round(minor / max(major, 1.0), 3),
        "ellipse": {
            "cx": float(ex), "cy": float(ey),
            "a": major / 2, "b": minor / 2,
            "angle": float(ang),
        },
    }


def measure_refined(
    frame: np.ndarray,
    contour: np.ndarray,
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    *,
    low_quality: bool,
    use_subpixel: bool = False,
) -> dict | None:
    """Optional sub-pixel edge refine (off by default for speed on long videos)."""
    ell = measure_ellipse(contour)
    if ell is None:
        return None
    if not use_subpixel or not low_quality or ell["roundness"] >= 0.88:
        return ell

    try:
        gray, grad = prepare_gray(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        refined, _, _ = fit_ellipse_from_edges(
            gray, grad,
            (cx, cy), (bw, bh), ell["angle"],
            mask=mask, two_pass=False,
        )
        if refined is None:
            return ell
        (ex, ey), (ea, eb), ang = refined
        major, minor = float(max(ea, eb)), float(min(ea, eb))
        return {
            "cx": float(ex), "cy": float(ey),
            "major_px": major, "minor_px": minor,
            "angle": float(ang),
            "roundness": round(minor / max(major, 1.0), 3),
            "ellipse": {
                "cx": float(ex), "cy": float(ey),
                "a": major / 2, "b": minor / 2,
                "angle": float(ang),
            },
            "refined": True,
        }
    except Exception:
        return ell


def is_valid_measurement(
    meas: dict,
    *,
    frame_w: int,
    baseline_minor_px: float | None = None,
    for_baseline: bool = False,
    ball_type: str = "tennis",
    low_quality: bool = False,
) -> bool:
    bx = meas.get("bbox_x", 0)
    by = meas.get("bbox_y", 0)
    bbw = meas.get("bbox_w", 0)
    touches_edge = bx <= 2 or by <= 2 or (bx + bbw) >= (frame_w - 2)

    min_ball = _min_ball_px(frame_w, low_quality)
    height_px = meas.get("height_px") or meas.get("minor_px", 0)
    roundness = meas.get("roundness", 0)
    aspect = height_px / max(meas.get("width_px") or meas.get("major_px", 1), 1.0)

    if height_px < min_ball:
        return False
    if touches_edge and not low_quality:
        return False
    if touches_edge and low_quality and (bx <= 0 or (bx + bbw) >= frame_w - 1):
        return False

    asp_min, asp_max = _aspect_bounds(ball_type)
    rnd_min = _roundness_min(ball_type) - (0.12 if low_quality else 0.0)

    if for_baseline:
        return asp_min <= aspect <= asp_max and roundness >= rnd_min

    if baseline_minor_px and baseline_minor_px > 0:
        if height_px < 0.30 * baseline_minor_px:
            return False
    return roundness >= (0.42 if low_quality else 0.50)


def analyze_ball_in_frame(
    frame: np.ndarray,
    *,
    ball_type: str = "tennis",
    baseline_minor_px: float | None = None,
    px_per_mm: float | None = None,
) -> dict:
    if frame is None:
        return {"detected": False, "status": "searching"}

    h_img, w_img = frame.shape[:2]
    quality = assess_frame_quality(w_img, h_img)
    low_q = quality["low_quality"]
    min_ball = _min_ball_px(w_img, low_q)
    profile = COLOR_PROFILES.get(ball_type, COLOR_PROFILES["tennis"])

    enhanced, scale = enhance_frame(frame, quality)
    inv = 1.0 / scale

    d = detect_on_frame(
        enhanced, ball_type=ball_type,
        min_area=max(180.0, min_ball * min_ball * 0.30),
        frame_w=enhanced.shape[1],
    )
    if d is None:
        return {"detected": False, "status": "searching", "low_quality_mode": low_q}

    ecx, ecy, ebw, ebh, econtour = d
    ell = measure_refined(enhanced, econtour, ecx, ecy, ebw, ebh, low_quality=low_q)
    if ell is None:
        return {"detected": False, "status": "searching", "low_quality_mode": low_q}

    if scale != 1.0:
        for key in ("cx", "cy", "major_px", "minor_px"):
            ell[key] *= inv
        e = ell["ellipse"]
        ell["ellipse"] = {**e, "cx": e["cx"] * inv, "cy": e["cy"] * inv, "a": e["a"] * inv, "b": e["b"] * inv}

    cx, cy, bw, bh = ell["cx"], ell["cy"], ell["major_px"], ell["minor_px"]
    bx, by, bbw, bbh = cv2.boundingRect((econtour.astype(np.float64) * inv).astype(np.int32))
    bh = float(bbh)
    bw = float(bbw)

    meas = {
        **ell,
        "bbox_x": bx, "bbox_y": by, "bbox_w": bbw, "bbox_h": bbh,
        "width_px": bw, "height_px": bh,
    }

    if not is_valid_measurement(
        meas, frame_w=w_img, baseline_minor_px=baseline_minor_px,
        ball_type=ball_type, low_quality=low_q,
    ):
        return {
            "detected": False,
            "status": "partial",
            "ellipse": ell["ellipse"],
            "roundness": ell["roundness"],
            "low_quality_mode": low_q,
        }

    ref_mm = profile["diameter_mm"]
    if px_per_mm is None or px_per_mm <= 0:
        px_per_mm = (baseline_minor_px / ref_mm) if baseline_minor_px else bh / ref_mm

    load_px = float(bh)
    compression_pct = 0.0
    if baseline_minor_px and baseline_minor_px > 0:
        compression_pct = max(0.0, (baseline_minor_px - load_px) / baseline_minor_px * 100.0)

    return {
        "detected": True,
        "status": "compressing" if compression_pct > 5 else "tracking",
        "quality": "good" if ell["roundness"] >= _roundness_min(ball_type) else "ok",
        "low_quality_mode": low_q,
        "enhanced": low_q,
        "subpixel": ell.get("refined", False),
        "cx": cx, "cy": cy,
        "major_px": round(ell["major_px"], 1),
        "minor_px": round(ell["minor_px"], 1),
        "width_px": float(bw),
        "height_px": float(bh),
        "diameter_mm": round(load_px / px_per_mm, 1),
        "compression_pct": round(compression_pct, 1),
        "roundness": ell["roundness"],
        "ellipse": ell["ellipse"],
    }


def compute_baseline_from_video(cap: cv2.VideoCapture, ball_type: str = "tennis") -> tuple[float, float]:
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    ref_mm = COLOR_PROFILES.get(ball_type, COLOR_PROFILES["tennis"])["diameter_mm"]
    minors: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = analyze_ball_in_frame(frame, ball_type=ball_type)
        if not r.get("detected"):
            continue
        aspect = r["height_px"] / max(r.get("width_px") or r["major_px"], 1.0)
        asp_min, asp_max = _aspect_bounds(ball_type)
        rnd_min = _roundness_min(ball_type) - (0.12 if r.get("low_quality_mode") else 0.0)
        if r.get("roundness", 0) >= rnd_min and asp_min <= aspect <= asp_max:
            minors.append(float(r["height_px"]))

    if not minors:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = analyze_ball_in_frame(frame, ball_type=ball_type)
            if r.get("detected") and r.get("height_px"):
                minors.append(float(r["height_px"]))

    if not minors:
        return 305.0, 305.0 / ref_mm

    return _robust_median(minors), _robust_median(minors) / ref_mm


def compute_baseline_from_results(frame_results: list[dict], ball_type: str = "tennis") -> dict:
    ref_mm = COLOR_PROFILES.get(ball_type, COLOR_PROFILES["tennis"])["diameter_mm"]
    minors, majors, roundness_vals = [], [], []
    low_q = any(r.get("low_quality_mode") for r in frame_results)
    min_required = MIN_BASELINE_LOW_QUALITY if low_q else MIN_BASELINE_GOOD_FRAMES
    rnd_min = _roundness_min(ball_type) - (0.12 if low_q else 0.0)

    for r in frame_results:
        if not r.get("detected"):
            continue
        minor = r.get("height_px") or r.get("minor_px")
        major = r.get("width_px") or r.get("major_px")
        if minor is None or major is None:
            continue
        aspect = minor / max(major, 1.0)
        rnd = float(r.get("roundness", 0))
        asp_min, asp_max = _aspect_bounds(ball_type)
        if aspect < asp_min or aspect > asp_max or rnd < rnd_min:
            continue
        minors.append(float(minor))
        majors.append(float(major))
        roundness_vals.append(rnd)

    good = len(minors)
    total = len(frame_results)

    if good < min_required:
        return {
            "ok": False,
            "reason": "not_enough_stable_frames",
            "good_frames": good,
            "target_frames": BASELINE_TARGET_FRAMES,
            "min_required": min_required,
            "quality_pct": round(good / max(total, 1) * 100, 1),
            "low_quality_mode": low_q,
        }

    clean_minors = []
    med = float(np.median(minors))
    mad = float(np.median(np.abs(np.array(minors) - med))) + 1e-6
    for v in minors:
        if abs(v - med) <= BASELINE_OUTLIER_SIGMA * 1.4826 * mad:
            clean_minors.append(v)
    if len(clean_minors) < min_required:
        clean_minors = minors

    baseline_minor = _robust_median(clean_minors)
    baseline_major = _robust_median(majors[: len(clean_minors)] if len(majors) >= len(clean_minors) else majors)
    px_per_mm = baseline_minor / ref_mm
    stability = 100.0 - min(100.0, float(np.std(clean_minors) / max(baseline_minor, 1.0) * 100.0 * 4))
    confidence = min(100.0, (good / BASELINE_TARGET_FRAMES) * 50 + stability * 0.5 + (10 if low_q else 0))

    return {
        "ok": True,
        "baseline_h_px": round(baseline_minor, 1),
        "baseline_minor_px": round(baseline_minor, 1),
        "baseline_major_px": round(baseline_major, 1),
        "baseline_w_px": round(baseline_major, 1),
        "baseline_mm": round(ref_mm, 1),
        "px_per_mm": round(px_per_mm, 4),
        "good_frames": good,
        "frames_used": len(clean_minors),
        "quality_pct": round(good / max(total, 1) * 100, 1),
        "stability_pct": round(max(0.0, stability), 1),
        "confidence_pct": round(confidence, 1),
        "avg_roundness": round(float(np.mean(roundness_vals)), 3) if roundness_vals else 0.0,
        "ball_type": ball_type,
        "low_quality_mode": low_q,
    }


class CompressionSmoother:
    def __init__(self, window: int = COMPRESSION_MEDIAN_WINDOW) -> None:
        self._hist: deque[float] = deque(maxlen=window)
        self.max_compression = 0.0
        self.max_raw = 0.0

    def update(self, raw_pct: float) -> float:
        self._hist.append(raw_pct)
        self.max_raw = max(self.max_raw, raw_pct)
        smoothed = float(np.median(self._hist))
        self.max_compression = max(self.max_compression, smoothed, raw_pct)
        return smoothed

    def reset(self) -> None:
        self._hist.clear()
        self.max_compression = 0.0
        self.max_raw = 0.0


class TemporalHeightFilter:
    """Median-filter ball height across frames — fills brief drop-outs on noisy video."""

    def __init__(self, window: int = TEMPORAL_HEIGHT_WINDOW) -> None:
        self._heights: deque[float] = deque(maxlen=window)
        self._missed = 0

    def update(self, height_px: float | None, detected: bool) -> float | None:
        if detected and height_px is not None and height_px > 0:
            self._heights.append(float(height_px))
            self._missed = 0
            return float(np.median(self._heights))
        self._missed += 1
        if self._heights and self._missed <= 3:
            return float(np.median(self._heights))
        return None

    def reset(self) -> None:
        self._heights.clear()
        self._missed = 0
