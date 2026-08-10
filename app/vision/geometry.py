from __future__ import annotations

import cv2
import numpy as np

from app.config import SUBPIXEL_RAYS

def bilinear_sample(image: np.ndarray, x: float, y: float) -> float:
    h, w = image.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return 0.0
    ix, iy = int(x), int(y)
    fx, fy = x - ix, y - iy
    return float(
        image[iy, ix] * (1 - fx) * (1 - fy)
        + image[iy, ix + 1] * fx * (1 - fy)
        + image[iy + 1, ix] * (1 - fx) * fy
        + image[iy + 1, ix + 1] * fx * fy
    )

def prepare_gray(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.8)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    return gray, grad_mag

def _ellipse_radius_at_angle(
    cx: float, cy: float, a: float, b: float, angle_deg: float, px: float, py: float
) -> float:
    
    dx, dy = px - cx, py - cy
    theta = np.arctan2(dy, dx) - np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    denom = (b * cos_t) ** 2 + (a * sin_t) ** 2
    if denom < 1e-9:
        return max(a, b)
    return float(a * b / np.sqrt(denom))

def fit_ellipse(contour: np.ndarray) -> tuple | None:
    if contour is None or len(contour) < 5:
        return None
    try:
        return cv2.fitEllipse(contour)
    except cv2.error:
        return None

def fit_ellipse_robust(
    points: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    max_iterations: int = 5,
    outlier_sigma: float = 2.5,
) -> tuple | None:
    
    pts = points.reshape(-1, 2).astype(np.float64)
    if len(pts) < 8:
        return fit_ellipse(points.astype(np.float32).reshape(-1, 1, 2))

    w = np.ones(len(pts), dtype=np.float64)
    if weights is not None and len(weights) == len(pts):
        w = np.clip(weights.astype(np.float64), 1e-3, None)
        w /= np.max(w)

    mask = np.ones(len(pts), dtype=bool)
    ellipse = None
    for _ in range(max_iterations):
        active = pts[mask]
        if len(active) < 5:
            break
        contour = active.astype(np.float32).reshape(-1, 1, 2)
        ellipse = fit_ellipse(contour)
        if ellipse is None:
            break
        center, axes, angle = ellipse
        cx, cy = float(center[0]), float(center[1])
        a, b = float(max(axes)) / 2, float(min(axes)) / 2
        residuals = []
        for px, py in pts:
            dist = np.hypot(px - cx, py - cy)
            expected = _ellipse_radius_at_angle(cx, cy, a, b, float(angle), px, py)
            residuals.append(abs(dist - expected))
        residuals = np.array(residuals)
        active_res = residuals[mask]
        med = float(np.median(active_res))
        mad = float(np.median(np.abs(active_res - med))) + 1e-6
        threshold = med + outlier_sigma * 1.4826 * mad
        new_mask = (residuals <= threshold) & (w >= np.percentile(w, 8))
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if ellipse is None:
        return None
    return ellipse

def fit_ellipse_ransac(
    points: np.ndarray,
    *,
    iterations: int = 120,
    inlier_threshold_px: float = 2.0,
    min_inlier_ratio: float = 0.55,
) -> tuple | None:
    
    pts = points.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 8:
        return fit_ellipse_robust(points)

    best_ellipse = None
    best_inliers = 0
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        idx = rng.choice(n, size=min(6, n), replace=False)
        sample = pts[idx].astype(np.float32).reshape(-1, 1, 2)
        ellipse = fit_ellipse(sample)
        if ellipse is None:
            continue
        center, axes, angle = ellipse
        cx, cy = float(center[0]), float(center[1])
        a, b = float(max(axes)) / 2, float(min(axes)) / 2
        if a < 5 or b < 5:
            continue
        inliers = 0
        for px, py in pts:
            dist = np.hypot(px - cx, py - cy)
            expected = _ellipse_radius_at_angle(cx, cy, a, b, float(angle), px, py)
            if abs(dist - expected) <= inlier_threshold_px:
                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_ellipse = ellipse

    if best_ellipse is None or best_inliers / n < min_inlier_ratio:
        return fit_ellipse_robust(points)

    
    center, axes, angle = best_ellipse
    cx, cy = float(center[0]), float(center[1])
    a, b = float(max(axes)) / 2, float(min(axes)) / 2
    inlier_pts = []
    for px, py in pts:
        dist = np.hypot(px - cx, py - cy)
        expected = _ellipse_radius_at_angle(cx, cy, a, b, float(angle), px, py)
        if abs(dist - expected) <= inlier_threshold_px:
            inlier_pts.append([px, py])
    if len(inlier_pts) >= 5:
        return fit_ellipse_robust(np.array(inlier_pts, dtype=np.float32).reshape(-1, 1, 2))
    return best_ellipse

def refine_radial_edges(
    gray: np.ndarray,
    grad_mag: np.ndarray,
    center: tuple[float, float],
    axes: tuple[float, float],
    angle_deg: float,
    n_rays: int = SUBPIXEL_RAYS,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    
    cx, cy = center
    major, minor = axes[0] / 2, axes[1] / 2
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    h, w = gray.shape
    points: list[tuple[float, float]] = []
    weights: list[float] = []

    for i in range(n_rays):
        theta = 2 * np.pi * i / n_rays
        ex = major * np.cos(theta)
        ey = minor * np.sin(theta)
        rx = ex * cos_a - ey * sin_a
        ry = ex * sin_a + ey * cos_a
        direction = np.array([rx, ry], dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction /= norm
        r_nom = norm

        radii = np.linspace(r_nom * 0.65, r_nom * 1.35, 120)
        r_mask_edge: float | None = None
        if mask is not None:
            was_inside = False
            for r in radii:
                px = cx + direction[0] * r
                py = cy + direction[1] * r
                if px < 1 or px >= w - 1 or py < 1 or py >= h - 1:
                    continue
                inside = bilinear_sample(mask.astype(np.float32), px, py) > 127
                if was_inside and not inside:
                    r_mask_edge = float(r)
                    break
                was_inside = inside

        search = (
            np.linspace(r_mask_edge * 0.96, r_mask_edge * 1.04, 40)
            if r_mask_edge is not None
            else radii
        )
        grads: list[float] = []
        mags: list[float] = []
        valid_r: list[float] = []
        for r in search:
            px = cx + direction[0] * r
            py = cy + direction[1] * r
            if px < 2 or px >= w - 2 or py < 2 or py >= h - 2:
                continue
            
            dr = 0.5
            g0 = bilinear_sample(grad_mag, px - direction[0] * dr, py - direction[1] * dr)
            g1 = bilinear_sample(grad_mag, px + direction[0] * dr, py + direction[1] * dr)
            grads.append(abs(g1 - g0))
            mags.append(bilinear_sample(grad_mag, px, py))
            valid_r.append(float(r))

        if len(grads) < 7:
            continue

        
        g_arr = np.array(grads)
        r_arr = np.array(valid_r)
        peak_threshold = max(float(np.max(g_arr)) * 0.45, float(np.percentile(g_arr, 85)))
        candidates = np.where(g_arr >= peak_threshold)[0]
        peak_idx = int(candidates[-1])  
        r_peak = float(r_arr[peak_idx])
        w = float(mags[peak_idx])

        
        if 0 < peak_idx < len(grads) - 1:
            g0, g1, g2 = grads[peak_idx - 1], grads[peak_idx], grads[peak_idx + 1]
            denom = 2 * (g0 - 2 * g1 + g2)
            if abs(denom) > 1e-9:
                offset = (g0 - g2) / denom
                step = valid_r[peak_idx + 1] - valid_r[peak_idx - 1]
                r_peak = valid_r[peak_idx] + offset * step * 0.5

        r_grad = r_peak
        if r_mask_edge is not None:
            
            r_peak = 0.32 * r_mask_edge + 0.68 * r_grad
        else:
            r_peak = r_grad

        px = cx + direction[0] * r_peak
        py = cy + direction[1] * r_peak
        points.append((px, py))
        weights.append(w)

    if len(points) < 8:
        return np.array([], dtype=np.float32).reshape(-1, 1, 2), np.array([])

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return pts, np.array(weights, dtype=np.float64)

def fit_ellipse_from_edges(
    gray: np.ndarray,
    grad_mag: np.ndarray,
    center: tuple[float, float],
    axes: tuple[float, float],
    angle_deg: float,
    mask: np.ndarray | None = None,
    *,
    two_pass: bool = True,
) -> tuple[tuple | None, np.ndarray, np.ndarray]:
    
    edge_pts, weights = refine_radial_edges(
        gray, grad_mag, center, axes, angle_deg, SUBPIXEL_RAYS, mask=mask
    )
    if len(edge_pts) < 8:
        return None, edge_pts, weights

    refined = fit_ellipse_robust(edge_pts, weights=weights)
    if refined is None:
        refined = fit_ellipse_ransac(edge_pts, inlier_threshold_px=2.0)

    if refined is not None and two_pass:
        c, ax, ang = refined
        cx, cy = float(c[0]), float(c[1])
        mp, mnp = float(max(ax)), float(min(ax))
        edge_pts2, weights2 = refine_radial_edges(
            gray, grad_mag, (cx, cy), (mp, mnp), float(ang), SUBPIXEL_RAYS, mask=mask
        )
        if len(edge_pts2) >= 8:
            refined2 = fit_ellipse_robust(edge_pts2, weights=weights2)
            if refined2 is not None:
                refined = refined2
                edge_pts, weights = edge_pts2, weights2

    return refined, edge_pts, weights

def equivalent_diameter_px(major_px: float, minor_px: float) -> float:
    
    return float(2.0 * np.sqrt((major_px / 2) * (minor_px / 2)))

def eccentricity(major_px: float, minor_px: float) -> float:
    if major_px <= 1e-6:
        return 0.0
    a, b = major_px / 2, minor_px / 2
    return float(np.sqrt(max(0.0, 1.0 - (b / a) ** 2)))
