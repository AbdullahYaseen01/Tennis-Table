from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.ball_profiles import get_profile

N_ANG = 180


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
    lo[1] = max(int(lo[1]) - 18, 62)
    lo[2] = max(int(lo[2]) - 30, 80)
    yellow = cv2.inRange(hsv, lo, hi)
    if ball_type.startswith("pickleball") or ball_type == "tennis":
        yellow |= cv2.inRange(hsv, (16, 50, 60), (50, 255, 255))
    skin = cv2.inRange(hsv, (0, 35, 45), (25, 180, 255))
    skin |= cv2.inRange(hsv, (160, 35, 45), (180, 180, 255))
    skin = cv2.bitwise_and(skin, cv2.bitwise_not(yellow))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k)
    # Light close only — a heavy close fills the finger dent and hides real press.
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    skin = cv2.dilate(skin, k, iterations=1)
    return yellow, skin


def _find_ball(frame: np.ndarray, ball_type: str, hint: tuple[float, float] | None):
    h, w = frame.shape[:2]
    yellow, skin = _masks(frame, ball_type)
    mask = cv2.bitwise_and(yellow, cv2.bitwise_not(skin))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core = cv2.erode(mask, k, iterations=1)
    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    yel_cnts, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = max(600.0, (min(h, w) * 0.07) ** 2)
    best = None
    best_score = -1e18
    n_ok = 0
    for cset in (contours, yel_cnts):
        for c in cset:
            area = float(cv2.contourArea(c))
            if area < min_area:
                continue
            peri = cv2.arcLength(c, True) + 1e-6
            circ = 4.0 * np.pi * area / (peri * peri)
            if circ < 0.22:
                continue
            (x, y), rad = cv2.minEnclosingCircle(c)
            if y > h * 0.88 or y < h * 0.10 or x < w * 0.10 or x > w * 0.90:
                continue
            if rad < min(h, w) * 0.05:
                continue
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(rad) or rad < 4:
                continue
            x0, y0 = max(0, int(x - rad)), max(0, int(y - rad))
            x1, y1 = min(w, int(x + rad)), min(h, int(y + rad))
            disk = yellow[y0:y1, x0:x1]
            if disk.size == 0:
                continue
            yy, xx = np.ogrid[y0:y1, x0:x1]
            inside = (xx - x) ** 2 + (yy - y) ** 2 <= (rad * 0.85) ** 2
            yel_frac = float((disk > 0)[inside].mean()) if inside.any() else 0.0
            if yel_frac < 0.32:
                continue
            hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            sat = float(hsv[:, :, 1][inside].mean()) if inside.any() else 0.0
            if sat < 75:
                continue
            n_ok += 1
            score = area * circ
            if hint is not None:
                score -= float(np.hypot(x - hint[0], y - hint[1])) * 8.0
            if score > best_score:
                best_score, best = score, (float(x), float(y), float(rad), yellow, skin, c)
        if best is not None:
            break
    if best is None or n_ok > 4:
        return None
    return best


def _polar_radii(mask: np.ndarray, cx: float, cy: float, r_max: float, skin: np.ndarray | None = None) -> np.ndarray:
    h, w = mask.shape[:2]
    radii = np.zeros(N_ANG, dtype=np.float64)
    if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(r_max) or r_max < 4:
        return radii
    steps = max(int(r_max * 1.15), 50)
    ts = np.linspace(0.18 * r_max, 1.28 * r_max, steps)
    for i in range(N_ANG):
        ang = 2.0 * np.pi * i / N_ANG
        ca, sa = np.cos(ang), np.sin(ang)
        last = 0.0
        hit_skin_in = False
        for t in ts:
            x = int(round(cx + t * ca))
            y = int(round(cy + t * sa))
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            if t < 0.95 * r_max and skin is not None and skin[y, x] > 0:
                hit_skin_in = True
            if mask[y, x] > 0:
                last = t
            elif last > 0 and t > last + 5:
                break
        if hit_skin_in and last > 0:
            last = min(last, 0.82 * r_max)
        radii[i] = last if np.isfinite(last) else 0.0
    good = radii > 1
    if good.sum() < N_ANG * 0.4:
        return radii
    idx = np.arange(N_ANG)
    xp = np.concatenate([idx[good], idx[good] + N_ANG])
    fp = np.concatenate([radii[good], radii[good]])
    radii = np.interp(idx, xp, fp)
    pad = np.r_[radii[-5:], radii, radii[:5]]
    return np.convolve(pad, np.ones(11) / 11.0, mode="valid")


def _skin_peak(cx: float, cy: float, r_new: float, skin: np.ndarray) -> tuple[int | None, np.ndarray]:
    strength = np.zeros(N_ANG, dtype=np.float64)
    ys, xs = np.where(skin > 0)
    if len(xs) < 25:
        return None, strength
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    dist = np.sqrt(dx * dx + dy * dy)
    ring = (dist >= r_new * 0.88) & (dist <= r_new * 1.38)
    if not np.any(ring):
        return None, strength
    ang = (np.arctan2(dy[ring], dx[ring]) + 2 * np.pi) % (2 * np.pi)
    bins = (ang / (2 * np.pi) * N_ANG).astype(np.int32) % N_ANG
    np.add.at(strength, bins, 1.0)
    pad = np.r_[strength[-8:], strength, strength[:8]]
    strength = np.convolve(pad, np.ones(17) / 17.0, mode="valid")
    peak = int(np.argmax(strength))
    if strength[peak] < 2.5:
        return None, strength
    return peak, strength


def _largest_arc(zone: np.ndarray) -> np.ndarray:
    n = len(zone)
    out = np.zeros(n, dtype=bool)
    if not zone.any():
        return out
    ext = np.r_[zone, zone]
    best_len = best_start = 0
    i = 0
    while i < len(ext):
        if not ext[i]:
            i += 1
            continue
        j = i
        while j < len(ext) and ext[j]:
            j += 1
        if j - i > best_len:
            best_len, best_start = j - i, i
        i = j
    best_len = min(best_len, n)
    for t in range(best_start, best_start + best_len):
        out[t % n] = True
    if out.sum() > n // 3:
        mid = (best_start + best_len // 2) % n
        dang = np.minimum(np.abs(np.arange(n) - mid), n - np.abs(np.arange(n) - mid))
        out = dang <= 22
    return out


def analyze_deformation(
    frame: np.ndarray,
    ball_type: str,
    cx: float = 0.0,
    cy: float = 0.0,
    approx_radius: float = 0.0,
    *,
    require_skin: bool = True,
) -> DeformationResult:
    hint = (cx, cy) if cx > 0 and cy > 0 else None
    hit = _find_ball(frame, ball_type, hint)
    if hit is None:
        return DeformationResult(reason="no-ball")
    cx, cy, r_enc, yellow, skin, contour = hit
    radii = _polar_radii(yellow, cx, cy, r_enc, skin)
    valid = radii > 0.25 * r_enc
    if valid.sum() < N_ANG * 0.35:
        return DeformationResult(reason="sparse-contour")

    r_new = float(np.percentile(radii[valid], 88))
    r_new = float(np.clip(r_new, 0.82 * r_enc, 1.08 * r_enc))
    if not np.isfinite(r_new) or r_new < 8:
        return DeformationResult(reason="bad-radius")
    deficit = np.maximum(0.0, r_new - radii)
    deficit[~valid] = 0.0

    zone = np.zeros(N_ANG, dtype=bool)
    press_pct = 0.0
    skin_peak, skin_str = _skin_peak(cx, cy, r_new, skin)
    deep_thr = max(0.08 * r_new, 8.0)

    if skin_peak is not None:
        skin_n = skin_str / max(float(skin_str.max()), 1e-6)
        def_n = deficit / max(r_new, 1.0)
        joint = skin_n * def_n
        pad = np.r_[joint[-6:], joint, joint[:6]]
        joint = np.convolve(pad, np.ones(13) / 13.0, mode="valid")
        cand = (skin_str >= max(0.20 * skin_str[skin_peak], 1.2)) & (deficit >= deep_thr) & valid
        if np.any(cand):
            peak = int(np.argmax(np.where(cand, joint, -1.0)))
            dang = np.minimum(
                np.abs(np.arange(N_ANG) - peak),
                N_ANG - np.abs(np.arange(N_ANG) - peak),
            )
            zone = (
                (dang <= 14)
                & (deficit >= max(0.55 * deficit[peak], deep_thr * 0.65))
                & (skin_str >= max(0.18 * skin_str[peak], 1.0))
                & valid
            )
            zone[peak] = True
    elif not require_skin or float(np.percentile(deficit, 95)) >= 0.14 * r_new:
        deep = (deficit >= 0.14 * r_new) & valid
        zone = _largest_arc(deep)
        if not (8 <= int(zone.sum()) <= 40):
            zone[:] = False

    zone = _largest_arc(zone)
    if zone.sum() < 8:
        zone[:] = False
        press_pct = 0.0
    else:
        zdef = deficit[zone]
        press_pct = float(np.clip(np.percentile(zdef, 65) / r_new * 100.0, 0.0, 32.0))
        if not np.isfinite(press_pct) or press_pct < 8.0:
            zone[:] = False
            press_pct = 0.0

    ang = 2.0 * np.pi * np.arange(N_ANG) / N_ANG
    outline = np.column_stack([cx + radii * np.cos(ang), cy + radii * np.sin(ang)])
    outline = np.nan_to_num(outline, nan=0.0, posinf=0.0, neginf=0.0)
    deform_pct = press_pct if np.isfinite(press_pct) else 0.0
    return DeformationResult(
        valid=True,
        reason="ok",
        cx=float(cx) if np.isfinite(cx) else 0.0,
        cy=float(cy) if np.isfinite(cy) else 0.0,
        r_baseline=r_new,
        radii=np.nan_to_num(radii, nan=0.0),
        contour=outline,
        arc_mask=zone,
        deform_pct=deform_pct,
        threshold_px=deep_thr if np.isfinite(deep_thr) else 8.0,
    )


class DeformationSmoother:
    def __init__(self, window: int = 3) -> None:
        self._pct: list[float] = []
        self._win = window

    def update(self, res: DeformationResult) -> DeformationResult:
        if not res.valid:
            self._pct.clear()
            return res
        if not np.isfinite(res.deform_pct) or res.deform_pct < 8.0:
            self._pct = [0.0]
            res.deform_pct = 0.0
            if res.arc_mask is not None:
                res.arc_mask[:] = False
            return res
        self._pct.append(res.deform_pct)
        if len(self._pct) > self._win:
            self._pct.pop(0)
        res.deform_pct = float(np.median(self._pct))
        if res.deform_pct < 8.0 and res.arc_mask is not None:
            res.arc_mask[:] = False
            res.deform_pct = 0.0
        return res


def deformed_contour_points(res: DeformationResult) -> np.ndarray | None:
    if not res.valid or res.arc_mask is None or not res.arc_mask.any() or res.radii is None:
        return None
    idx = np.where(res.arc_mask)[0]
    ang = 2.0 * np.pi * idx / N_ANG
    r = res.radii[idx]
    xs = res.cx + r * np.cos(ang)
    ys = res.cy + r * np.sin(ang)
    return np.stack([xs, ys], axis=1).astype(np.int32)
