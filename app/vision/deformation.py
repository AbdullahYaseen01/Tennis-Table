from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.ball_profiles import get_profile

N_ANGLES = 360


@dataclass
class DeformationResult:
    valid: bool = False
    reason: str = "init"
    cx: float = 0.0
    cy: float = 0.0
    r_baseline: float = 0.0
    radii: np.ndarray | None = None
    contour: np.ndarray | None = None
    arc_mask: np.ndarray | None = None
    deform_pct: float = 0.0
    threshold_px: float = 0.0


def _masks(frame: np.ndarray, ball_type: str) -> tuple[np.ndarray, np.ndarray]:
    profile = get_profile(ball_type)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = profile.hsv_lower.copy()
    hi = profile.hsv_upper.copy()
    lo[1] = max(int(lo[1]), 90)
    lo[2] = max(int(lo[2]), 110)
    yellow = cv2.inRange(hsv, lo, hi)
    skin = cv2.inRange(hsv, (0, 30, 45), (25, 175, 255))
    skin |= cv2.inRange(hsv, (160, 30, 45), (180, 175, 255))
    yellow = cv2.bitwise_and(yellow, cv2.bitwise_not(skin))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k)
    yellow = cv2.morphologyEx(
        yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2
    )
    return yellow, skin


def _select_blob(mask: np.ndarray, cx: float, cy: float, r: float) -> tuple[np.ndarray | None, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big: list[tuple[float, np.ndarray]] = []
    min_area = max(700.0, (r * 0.45) ** 2)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        d = float(np.hypot(m["m10"] / m["m00"] - cx, m["m01"] / m["m00"] - cy))
        if d > r * 1.4:
            continue
        big.append((area, c))
    if not big:
        return None, 0
    big.sort(key=lambda x: x[0], reverse=True)
    return big[0][1], len(big)


def _polar_radii(contour: np.ndarray, cx: float, cy: float) -> np.ndarray:
    pts = contour.reshape(-1, 2).astype(np.float64)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    ang = (np.degrees(np.arctan2(dy, dx)).astype(int) % N_ANGLES)
    rad = np.hypot(dx, dy)
    radii = np.full(N_ANGLES, np.nan)
    for a, rr in zip(ang, rad):
        if np.isnan(radii[a]) or rr > radii[a]:
            radii[a] = rr
    idx = np.arange(N_ANGLES)
    good = ~np.isnan(radii)
    if good.sum() < N_ANGLES * 0.5:
        return radii
    xp = np.concatenate([idx[good], idx[good] + N_ANGLES])
    fp = np.concatenate([radii[good], radii[good]])
    radii = np.interp(idx, xp, fp)
    k = 5
    pad = np.concatenate([radii[-k:], radii, radii[:k]])
    radii = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    return radii


def _largest_arc(flags: np.ndarray) -> np.ndarray:
    out = np.zeros_like(flags)
    if not flags.any():
        return out
    n = len(flags)
    doubled = np.concatenate([flags, flags])
    best_len = best_start = 0
    cur_len = cur_start = 0
    for i in range(2 * n):
        if doubled[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    best_len = min(best_len, n)
    for i in range(best_start, best_start + best_len):
        out[i % n] = True
    return out


def analyze_deformation(
    frame: np.ndarray,
    ball_type: str,
    cx: float,
    cy: float,
    approx_radius: float,
    *,
    require_skin: bool = True,
) -> DeformationResult:
    if approx_radius < 20:
        return DeformationResult(reason="no-baseline")
    yellow, skin = _masks(frame, ball_type)
    contour, n_blobs = _select_blob(yellow, cx, cy, approx_radius)
    if contour is None:
        return DeformationResult(reason="no-ball")
    if n_blobs > 2:
        return DeformationResult(reason="multi-blob")

    m = cv2.moments(contour)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    radii = _polar_radii(contour, cx, cy)
    if np.isnan(radii).any():
        return DeformationResult(reason="sparse-contour")

    r_baseline = float(np.percentile(radii, 78))
    if not (approx_radius * 0.5 <= r_baseline <= approx_radius * 1.7):
        return DeformationResult(reason="implausible-radius")

    deficit = np.maximum(0.0, r_baseline - radii)
    sigma = 1.4826 * float(np.median(np.abs(deficit - np.median(deficit)))) + 1e-6
    threshold = max(3.0 * sigma, 0.03 * r_baseline, 4.0)

    flags = _largest_arc(deficit > threshold)

    if require_skin and flags.any():
        ys, xs = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
        idx = np.where(flags)[0]
        a0, a1 = np.deg2rad(idx.min()), np.deg2rad(idx.max())
        ang_img = np.arctan2(ys - cy, xs - cx)
        ang_img = np.where(ang_img < 0, ang_img + 2 * np.pi, ang_img)
        dist = np.hypot(xs - cx, ys - cy)
        band = (dist >= r_baseline * 0.9) & (dist <= r_baseline * 1.25)
        band &= (ang_img >= min(a0, a1)) & (ang_img <= max(a0, a1))
        tot = int(band.sum())
        frac = float((skin > 0)[band].sum()) / tot if tot > 50 else 0.0
        if frac < 0.06:
            flags[:] = False

    peak = float(deficit[flags].max()) if flags.any() else 0.0
    deform_pct = min(45.0, peak / r_baseline * 100.0)
    if deform_pct < 3.0:
        flags[:] = False
        deform_pct = 0.0

    return DeformationResult(
        valid=True,
        reason="ok",
        cx=cx,
        cy=cy,
        r_baseline=r_baseline,
        radii=radii,
        contour=contour.reshape(-1, 2),
        arc_mask=flags,
        deform_pct=deform_pct,
        threshold_px=threshold,
    )


class DeformationSmoother:
    """Rolling-median smoothing of deformation to suppress single-frame jitter."""

    def __init__(self, window: int = 5, hold: int = 6) -> None:
        self._pct: list[float] = []
        self._win = window
        self._hold_max = hold
        self._last: DeformationResult | None = None
        self._hold = 0

    def update(self, res: DeformationResult) -> DeformationResult:
        if not res.valid:
            if self._last is not None and self._hold > 0:
                self._hold -= 1
                return self._last
            return res
        self._pct.append(res.deform_pct)
        if len(self._pct) > self._win:
            self._pct.pop(0)
        res.deform_pct = float(np.median(self._pct))
        if res.deform_pct < 3.0 and res.arc_mask is not None:
            res.arc_mask[:] = False
        self._last = res
        self._hold = self._hold_max
        return res


def deformed_contour_points(res: DeformationResult) -> np.ndarray | None:
    """Pixel points of the true deformed arc only, for overlay drawing."""
    if not res.valid or res.arc_mask is None or not res.arc_mask.any():
        return None
    idx = np.where(res.arc_mask)[0]
    ang = np.deg2rad(idx.astype(np.float64))
    r = res.radii[idx]
    xs = res.cx + r * np.cos(ang)
    ys = res.cy + r * np.sin(ang)
    return np.stack([xs, ys], axis=1).astype(np.int32)
