"""High-accuracy tennis ball tracker + annotated video export.

Tuned for bright ball on grass footage. Mask-seeded ROI + Hough circle
refinement, temporal smoothing, trajectory trail, and clean overlays.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KNOWN_DIAMETER_MM = 67.0  # ITF tennis ball spec

HSV_LOWER = np.array([26, 90, 120])
HSV_UPPER = np.array([45, 255, 255])
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def detect_ball(frame: np.ndarray) -> tuple[float, float, float] | None:
    """Return (cx, cy, radius) via colour-seeded ROI + Hough refinement."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1500:
            continue
        if area > best_area:
            best_area = area
            best = c
    if best is None:
        return None

    (sx, sy), sr = cv2.minEnclosingCircle(best)
    h, w = frame.shape[:2]
    pad = int(sr * 1.8)
    x0, y0 = max(0, int(sx - pad)), max(0, int(sy - pad))
    x1, y1 = min(w, int(sx + pad)), min(h, int(sy + pad))
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return (sx, sy, sr)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1.2, 100,
        param1=80, param2=40,
        minRadius=int(sr * 0.6), maxRadius=int(sr * 1.5),
    )
    if circles is not None:
        cx, cy, r = circles[0][0]
        return (x0 + float(cx), y0 + float(cy), float(r))
    return (sx, sy, sr)


class Smoother:
    """Median + EMA smoothing for stable center and radius."""

    def __init__(self, window: int = 5, alpha: float = 0.5) -> None:
        self.cx: deque[float] = deque(maxlen=window)
        self.cy: deque[float] = deque(maxlen=window)
        self.r: deque[float] = deque(maxlen=window)
        self.alpha = alpha
        self._ema: tuple[float, float, float] | None = None

    def update(self, cx: float, cy: float, r: float) -> tuple[float, float, float]:
        self.cx.append(cx)
        self.cy.append(cy)
        self.r.append(r)
        mx = float(np.median(self.cx))
        my = float(np.median(self.cy))
        mr = float(np.median(self.r))
        if self._ema is None:
            self._ema = (mx, my, mr)
        else:
            a = self.alpha
            ex, ey, er = self._ema
            self._ema = (a * mx + (1 - a) * ex, a * my + (1 - a) * ey, a * mr + (1 - a) * er)
        return self._ema


def draw_panel(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    pad, line_h = 12, 30
    width = 330
    height = len(lines) * line_h + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + width, 10 + height), (0, 210, 255), 2)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10 + pad, 10 + pad + (i + 1) * line_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def export(video_path: Path, out_avi: Path) -> tuple[Path, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_avi.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_avi), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")

    # Robust scale: use median radius across a first pass for stable px/mm
    radii_pass: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = detect_ball(frame)
        if res:
            radii_pass.append(res[2])
    ref_radius = float(np.median(radii_pass)) if radii_pass else 180.0
    px_per_mm = (ref_radius * 2) / KNOWN_DIAMETER_MM

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    smoother = Smoother(window=5, alpha=0.5)
    trail: deque[tuple[int, int]] = deque(maxlen=25)
    frame_idx = 0
    detected = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = detect_ball(frame)
        t = frame_idx / fps

        if res:
            detected += 1
            cx, cy, r = smoother.update(*res)
            dia_mm = (2 * r) / px_per_mm
            trail.append((int(cx), int(cy)))

            # Trajectory trail
            for j in range(1, len(trail)):
                alpha = j / len(trail)
                cv2.line(frame, trail[j - 1], trail[j],
                         (0, int(120 + 135 * alpha), 255), 2, cv2.LINE_AA)

            cv2.circle(frame, (int(cx), int(cy)), int(r), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255),
                           cv2.MARKER_CROSS, 22, 2)
            cv2.putText(frame, f"D = {dia_mm:.1f} mm",
                        (int(cx - r), max(34, int(cy - r - 14))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            status = ("TRACKING", (0, 255, 0))
            dia_line = (f"Diameter: {dia_mm:.1f} mm", (0, 255, 255))
        else:
            status = ("SEARCHING", (0, 165, 255))
            dia_line = ("Diameter: --", (200, 200, 200))

        draw_panel(frame, [
            ("TENNIS BALL TRACKING", (0, 255, 180)),
            (f"Time: {t:5.2f} s", (240, 240, 240)),
            dia_line,
            (f"Status: {status[0]}", status[1]),
        ])

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 40 == 0:
            print(f"  {frame_idx}/{total} frames", flush=True)

    cap.release()
    writer.release()
    rate = detected / max(frame_idx, 1) * 100
    print(f"Detection rate: {rate:.1f}%  |  ref diameter {ref_radius*2:.0f}px = {KNOWN_DIAMETER_MM}mm")
    return out_avi, frame_idx, rate


def main() -> int:
    video = ROOT / "data" / "samples" / "Latest Test Video.mp4"
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    if not video.exists():
        print(f"Video not found: {video}")
        return 1

    stem = video.stem.replace(" ", "-").lower()
    out_avi = ROOT / "data" / "reports" / f"{stem}-tracked.avi"
    print(f"Input:  {video}\nOutput: {out_avi}\n")
    export(video, out_avi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
