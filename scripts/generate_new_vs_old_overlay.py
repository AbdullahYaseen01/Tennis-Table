from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vision.ball_profiles import get_profile

def find_ball(frame: np.ndarray) -> tuple[float, float, float, np.ndarray] | None:
    profile = get_profile("pickleball")
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = profile.hsv_lower.copy()
    hi = profile.hsv_upper.copy()
    lo[1] = max(int(lo[1]), 110)
    lo[2] = max(int(lo[2]), 140)
    mask = cv2.inRange(hsv, lo, hi)
    skin = cv2.inRange(hsv, (0, 30, 40), (25, 170, 255))
    skin |= cv2.inRange(hsv, (160, 30, 40), (180, 170, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    core = cv2.erode(mask, k, iterations=2)
    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best, best_score = None, -1.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 2500:
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < 0.40:
            continue
        (x, y), rad = cv2.minEnclosingCircle(c)
        if y > h * 0.92 or rad < 40:
            continue
        score = area * circ
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(rad), c)
    return best

def deform_stats(contour: np.ndarray, cx: float, cy: float, r_new: float) -> tuple[float, np.ndarray]:
    
    pts = contour.reshape(-1, 2).astype(np.float64)
    dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    
    inward = dist < r_new * 0.92
    if inward.sum() < 8:
        deficit = np.maximum(0.0, r_new - dist)
        score = float(np.percentile(deficit, 60)) / r_new * 100.0
        return max(0.0, min(40.0, score)), inward
    mean_in = float(np.mean(dist[inward])) if inward.any() else r_new
    dent = (1.0 - mean_in / r_new) * 100.0
    
    deficit = np.maximum(0.0, r_new - dist)
    score = float(np.percentile(deficit, 60)) / r_new * 100.0
    dent = max(dent * 0.7, score)
    return max(0.0, min(40.0, dent)), inward

def grade_of(pct: float) -> tuple[str, tuple[int, int, int]]:
    if pct < 12:
        return "GRADE A — GOOD / NEW", (0, 200, 80)
    if pct < 22:
        return "GRADE B — USED / MONITOR", (0, 200, 255)
    return "GRADE C — REPLACE", (0, 60, 255)

def draw_overlay(frame: np.ndarray, *, title: str) -> tuple[np.ndarray, float, str] | None:
    hit = find_ball(frame)
    if hit is None:
        return None
    cx, cy, r_fit, contour = hit
    
    r_new = r_fit  
    
    
    dent_pct, inward = deform_stats(contour, cx, cy, r_new)
    grade, gcolor = grade_of(dent_pct)

    out = frame.copy()
    
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (456, 168), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, out, 0.38, 0, out)

    
    cv2.circle(out, (int(cx), int(cy)), int(r_new), (255, 200, 0), 3, cv2.LINE_AA)
    
    cv2.drawContours(out, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
    
    pts = contour.reshape(-1, 2)
    dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    for i, p in enumerate(pts[::2]):
        if dist[i * 2] < r_new * 0.93:
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1, cv2.LINE_AA)

    
    dent_pts = pts[dist < r_new * 0.93]
    if len(dent_pts) >= 6:
        hull = cv2.convexHull(dent_pts.astype(np.int32))
        layer = out.copy()
        cv2.fillConvexPoly(layer, hull, (0, 0, 180))
        cv2.addWeighted(layer, 0.25, out, 0.75, 0, out)

    cv2.drawMarker(out, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)

    
    y0 = 36
    cv2.putText(out, title, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(out, "CYAN = New ball circumference (reference)", (20, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(out, "GREEN = Used ball outline", (20, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, "RED = Deformed parts (vs new ball)", (20, y0 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"Deformation: {dent_pct:.1f}%", (20, y0 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, grade, (20, y0 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.58, gcolor, 2, cv2.LINE_AA)

    
    cv2.rectangle(out, (out.shape[1] - 168, 12), (out.shape[1] - 12, 70), gcolor, -1)
    cv2.putText(out, grade.split("—")[0].strip(), (out.shape[1] - 158, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    return out, dent_pct, grade

def main() -> int:
    video = ROOT / "data" / "samples" / "testing ball.mp4"
    out_dir = ROOT / "data" / "reports" / "client-overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    
    picks = [
        (40, "1) Near-round / light contact"),
        (83, "2) Clear press — deformed vs new circle"),
        (140, "3) Strong squeeze — deformed parts marked"),
        (200, "4) Compression peak region"),
        (463, "5) Late press — new vs used overlay"),
        (482, "6) Hard thumb press — grade example"),
    ]

    cap = cv2.VideoCapture(str(video))
    results = []
    for fi, title in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"skip {fi}")
            continue
        drawn = draw_overlay(frame, title=title)
        if drawn is None:
            print(f"no ball @ {fi}")
            continue
        img, pct, grade = drawn
        path = out_dir / f"overlay-f{fi}.jpg"
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        results.append((fi, pct, grade, path))
        print(f"wrote {path.name}: {pct:.1f}% | {grade}")
    cap.release()

    
    if len(results) >= 4:
        thumbs = []
        for fi, pct, grade, path in results[:6]:
            im = cv2.imread(str(path))
            im = cv2.resize(im, (360, 640))
            thumbs.append(im)
        while len(thumbs) < 6:
            thumbs.append(np.zeros_like(thumbs[0]))
        row1 = np.hstack(thumbs[0:3])
        row2 = np.hstack(thumbs[3:6])
        collage = np.vstack([row1, row2])
        
        header = np.zeros((90, collage.shape[1], 3), dtype=np.uint8)
        header[:] = (28, 28, 28)
        cv2.putText(header, "NEW BALL vs USED BALL OVERLAY — Deformation Detection Demo", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(header, "Cyan circle = brand-new circumference | Green = used outline | Red = deformed parts | Grades A/B/C", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        collage = np.vstack([header, collage])
        collage_path = out_dir / "CLIENT-new-vs-old-overlay-DEMO.jpg"
        cv2.imwrite(str(collage_path), collage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"\nCLIENT COLLAGE: {collage_path}")

    
    legend = np.zeros((420, 720, 3), dtype=np.uint8)
    legend[:] = (24, 24, 24)
    cv2.putText(legend, "3-GRADE DEFORMATION SYSTEM (New vs Old Overlay)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
    rows = [
        ((0, 200, 80), "GRADE A — GOOD / NEW", "Deformation < 12%  |  outline matches new circle"),
        ((0, 200, 255), "GRADE B — USED / MONITOR", "Deformation 12–22%  |  visible indent vs new circle"),
        ((0, 60, 255), "GRADE C — REPLACE", "Deformation > 22%  |  large mismatch / deep dent"),
    ]
    y = 110
    for color, name, desc in rows:
        cv2.rectangle(legend, (40, y - 28), (90, y + 12), color, -1)
        cv2.putText(legend, name, (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        cv2.putText(legend, desc, (110, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
        y += 90
    cv2.putText(legend, "Method: Overlay brand-new ball circumference on used ball,", (40, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(legend, "detect inward mismatch, mark deformed parts in red, assign grade.", (40, 398), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    legend_path = out_dir / "CLIENT-3-grade-legend.jpg"
    cv2.imwrite(str(legend_path), legend, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"GRADE LEGEND: {legend_path}")

    
    import shutil
    dl = Path.home() / "Downloads" / "pickleball-new-vs-old-overlay"
    dl.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.jpg"):
        shutil.copy2(p, dl / p.name)
    print(f"\nCopied to: {dl}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
