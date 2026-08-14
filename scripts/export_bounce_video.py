from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import IS_VERCEL
from app.vision.deformation import (
    DeformationResult,
    DeformationSmoother,
    analyze_deformation,
    deformed_contour_points,
)
from app.vision.video_measure import (
    VideoBallAnalyzer,
    VideoFrameResult,
    IMPACT_STATUS_MIN_PCT,
)

VERCEL_MAX_EDGE = 720

def _lock_press_baseline(video_path: Path, ball_type: str, known_mm: float) -> tuple[float, float] | None:
    cap = cv2.VideoCapture(str(video_path))
    radii: list[float] = []
    any_r: list[float] = []
    for _ in range(18):
        ok, frame = cap.read()
        if not ok:
            break
        if max(frame.shape[0], frame.shape[1]) > VERCEL_MAX_EDGE:
            s = VERCEL_MAX_EDGE / max(frame.shape[0], frame.shape[1])
            frame = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        res = analyze_deformation(frame, ball_type, require_skin=True)
        if res.valid and res.r_baseline > 20:
            any_r.append(res.r_baseline)
            if res.deform_pct < 8.0:
                radii.append(res.r_baseline)
    cap.release()
    use = radii if len(radii) >= 3 else any_r
    if len(use) < 2:
        return None
    r = float(np.median(radii))
    bpx = r * 2.0
    return bpx, bpx / known_mm

def find_hand_press_ball(
    frame: np.ndarray,
    baseline_px: float,
    last_xy: tuple[float, float] | None,
) -> tuple[float, float, float] | None:
    
    from app.vision.ball_profiles import get_profile

    profile = get_profile("pickleball")
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = profile.hsv_lower.copy()
    hi = profile.hsv_upper.copy()
    lo[1] = max(int(lo[1]), 120)
    lo[2] = max(int(lo[2]), 150)
    mask = cv2.inRange(hsv, lo, hi)
    skin = cv2.inRange(hsv, (0, 35, 40), (25, 180, 255))
    skin |= cv2.inRange(hsv, (160, 35, 40), (180, 180, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    core = cv2.erode(mask, k, iterations=3)
    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (baseline_px * 0.45) ** 2
    cands: list[tuple[float, float, float, float, float, float]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < 0.50:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        x = bx + bw * 0.5
        y = by + bh * 0.5
        maj = float(max(bw, bh))
        if maj < baseline_px * 0.55 or maj > baseline_px * 1.45:
            continue
        if y > h * 0.90:
            continue
        
        rad_i = int(max(bw, bh) * 0.45)
        x0c, y0c = max(0, int(x) - rad_i), max(0, int(y) - rad_i)
        x1c, y1c = min(w, int(x) + rad_i), min(h, int(y) + rad_i)
        disk = np.zeros((y1c - y0c, x1c - x0c), dtype=np.uint8)
        cv2.circle(disk, (int(x) - x0c, int(y) - y0c), rad_i, 255, -1)
        skin_roi = skin[y0c:y1c, x0c:x1c]
        yel_roi = mask[y0c:y1c, x0c:x1c]
        if disk.size == 0:
            continue
        inside = disk > 0
        skin_frac = float(skin_roi[inside].astype(bool).sum()) / max(1, int(inside.sum()))
        yel_frac = float(yel_roi[inside].astype(bool).sum()) / max(1, int(inside.sum()))
        if skin_frac > 0.28 or yel_frac < 0.30:
            continue
        score = area * (circ ** 2) * yel_frac
        cands.append((score, float(x), float(y), maj, circ, float(min(bw, bh))))
    if not cands:
        return None
    cands.sort(reverse=True)
    
    return (cands[0][1], cands[0][2], cands[0][3])

def draw_panel(frame, lines, *, width: int = 440) -> int:
    pad, line_h = 14, 34
    height = len(lines) * line_h + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + width, 10 + height), (0, 210, 255), 2)
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text, (10 + pad, 10 + pad + (i + 1) * line_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
        )
    return 10 + height

def draw_bar(frame, x, y, w, h, frac, color):
    frac = float(np.clip(frac, 0, 1))
    cv2.putText(frame, "Compression", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cv2.rectangle(frame, (x, y), (x + int(w * frac), y + h), color, -1)

def _draw_yellow_outline(frame, cx: int, cy: int, radius: int, ball_type: str) -> bool:
    
    from app.vision.ball_profiles import get_profile

    profile = get_profile(ball_type)
    h, w = frame.shape[:2]
    pad = int(max(radius * 1.2, 80))
    x0, y0 = max(0, cx - pad), max(0, cy - pad)
    x1, y1 = min(w, cx + pad), min(h, cy + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lo = profile.hsv_lower.copy()
    hi = profile.hsv_upper.copy()
    lo[1] = max(int(lo[1]), 100)
    lo[2] = max(int(lo[2]), 130)
    mask = cv2.inRange(hsv, lo, hi)
    skin = cv2.inRange(hsv, (0, 25, 50), (25, 160, 255))
    skin |= cv2.inRange(hsv, (160, 25, 50), (180, 160, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_fill, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False
    cx_l, cy_l = float(cx - x0), float(cy - y0)
    best, best_score = None, -1e18
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circ = 4.0 * np.pi * area / (peri * peri)
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        dist = float(np.hypot(m["m10"] / m["m00"] - cx_l, m["m01"] / m["m00"] - cy_l))
        if dist > radius * 0.55:
            continue
        score = area * circ - dist * 40.0
        if score > best_score:
            best_score, best = score, c
    if best is None:
        return False
    best = best + np.array([[[x0, y0]]], dtype=best.dtype)
    cv2.drawContours(frame, [best], -1, (0, 255, 0), 3, cv2.LINE_AA)
    return True

def draw_measurement(
    frame,
    r: VideoFrameResult,
    baseline_radius: int,
    frame_h: int,
    *,
    ball_type: str = "tennis",
    hand_press: bool = False,
) -> None:
    has_shape = r.major_px > 10 and r.cx > 0 and r.cy > 0
    if not has_shape:
        return
    cx, cy = int(r.cx), int(r.cy)
    if cy < 15 or cy > frame_h - 15:
        return

    rx = max(1, int(r.major_px / 2))
    ry = max(1, int(r.minor_px / 2))
    
    if hand_press:
        if r.compression_pct >= IMPACT_STATUS_MIN_PCT:
            cv2.circle(frame, (cx, cy), baseline_radius, (255, 180, 0), 2, cv2.LINE_AA)
        live_r = max(8, int(baseline_radius * (1.0 - min(r.compression_pct, 40.0) / 100.0)))
        cv2.ellipse(frame, (cx, cy), (baseline_radius, live_r), 0, 0, 360, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    else:
        if r.compression_pct >= IMPACT_STATUS_MIN_PCT:
            cv2.circle(frame, (cx, cy), baseline_radius, (255, 180, 0), 2, cv2.LINE_AA)
        outlined = False
        if ball_type.startswith("pickleball"):
            outlined = _draw_yellow_outline(frame, cx, cy, max(rx, ry, baseline_radius), ball_type)
        if not outlined:
            cv2.ellipse(
                frame, (cx, cy), (rx, ry), r.angle, 0, 360, (0, 255, 0), 3, cv2.LINE_AA,
            )
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if r.edge_points is not None and len(r.edge_points) > 0 and not outlined:
            for pt in r.edge_points[::8]:
                cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 2, (0, 200, 255), -1)

    comp = r.compression_pct
    color = (0, 0, 255) if comp >= IMPACT_STATUS_MIN_PCT else (0, 255, 255) if comp >= 8 else (0, 255, 0)
    label_r = max(rx, ry, baseline_radius)
    cv2.putText(
        frame, f"{comp:.1f}% compressed",
        (max(8, cx - 110), max(40, cy - label_r - 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"D = {r.diameter_mm:.1f} mm",
        (max(8, cx - 75), min(frame_h - 12, cy + label_r + 30)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )

def draw_deformation(frame, res: DeformationResult, baseline_mm: float, frame_h: int) -> None:
    """Draw validated deformation only: baseline circle, true contour, deformed arc."""
    cx, cy = int(res.cx), int(res.cy)
    r = int(res.r_baseline)
    cv2.circle(frame, (cx, cy), r, (0, 255, 255), 1, cv2.LINE_AA)
    if res.contour is not None and len(res.contour) > 3:
        cv2.polylines(frame, [res.contour.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    arc = deformed_contour_points(res)
    if arc is not None and len(arc) > 1:
        cv2.polylines(frame, [arc], False, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    comp = res.deform_pct
    color = (0, 0, 255) if comp >= IMPACT_STATUS_MIN_PCT else (0, 255, 255) if comp >= 8 else (0, 255, 0)
    cv2.putText(
        frame, f"{comp:.1f}% deformed", (max(8, cx - 110), max(40, cy - r - 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA,
    )
    dia = baseline_mm * (1.0 - comp / 100.0) if comp > 0 else baseline_mm
    cv2.putText(
        frame, f"D = {dia:.1f} mm", (max(8, cx - 75), min(frame_h - 12, cy + r + 30)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _smooth_comp(history: deque[float]) -> float:
    if not history:
        return 0.0
    vals = [v for v in history if v >= 0]
    if not vals:
        return 0.0
    if vals[-1] > 0:
        return float(np.median(vals[-3:] if len(vals) >= 3 else vals))
    return float(np.median(vals))

def export(
    video_path: Path,
    out_path: Path,
    *,
    ball_type: str = "tennis",
    use_yolo: bool | None = None,
    fast_mode: bool | None = None,
    fixed_baseline_px: float | None = None,
    fixed_px_per_mm: float | None = None,
) -> dict:
    if fast_mode is None:
        fast_mode = IS_VERCEL
    hand_press = ball_type.startswith("pickleball") or any(
        k in video_path.stem.lower() for k in ("testing", "hand", "press", "squeeze")
    )
    if use_yolo is None:
        
        use_yolo = False if hand_press else (not fast_mode)
    analyzer = VideoBallAnalyzer(ball_type=ball_type, use_yolo=use_yolo, fast_mode=fast_mode)

    print("  Locking baseline...", flush=True)
    locked = False
    if hand_press:
        press_base = _lock_press_baseline(video_path, ball_type, analyzer.known_mm)
        if press_base:
            bpx, ppm = press_base
            analyzer.baseline_vertical_px = bpx
            analyzer._px_per_mm = ppm
            analyzer.calibrator.pixels_per_mm = ppm
            analyzer.baseline_locked = True
            analyzer.bounce_counter = None
            locked = True
    if not locked:
        locked = analyzer.lock_baseline_from_video(str(video_path))
        if hand_press:
            analyzer.bounce_counter = None
    if not locked and fixed_baseline_px and fixed_baseline_px > 0:
        analyzer.baseline_vertical_px = fixed_baseline_px
        analyzer._px_per_mm = fixed_px_per_mm or (fixed_baseline_px / analyzer.known_mm)
        analyzer.calibrator.pixels_per_mm = analyzer._px_per_mm
        analyzer.baseline_locked = True
        from app.vision.video_measure import BounceCounter, ContactPhaseTracker

        analyzer.bounce_counter = BounceCounter(analyzer._frame_h or 720, analyzer._fps, floor_frac=analyzer._floor_frac)
        analyzer._contact = ContactPhaseTracker(analyzer._frame_h or 720, floor_frac=analyzer._floor_frac)
    if not analyzer.baseline_locked:
        raise RuntimeError("Could not lock baseline from video")

    baseline_px = analyzer.baseline_vertical_px or 320.0
    baseline_mm = analyzer.known_mm
    baseline_radius = int(baseline_px / 2)
    floor_frac = analyzer._floor_frac
    ball_label = ball_type.replace("_", " ").upper()

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analyzer._fps = fps
    analyzer._frame_h = h
    analyzer._frame_w = w
    analyzer._frame_idx = 0

    scale = 1.0
    if fast_mode and max(w, h) > VERCEL_MAX_EDGE:
        scale = VERCEL_MAX_EDGE / max(w, h)
        w_out, h_out = int(w * scale), int(h * scale)
        if analyzer.baseline_vertical_px:
            analyzer.baseline_vertical_px *= scale
        if analyzer._px_per_mm:
            analyzer._px_per_mm *= scale
            analyzer.calibrator.pixels_per_mm = analyzer._px_per_mm
        analyzer._frame_h = h_out
        analyzer._frame_w = w_out
    else:
        w_out, h_out = w, h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_out, h_out))
    if not writer.isOpened():
        out_path = out_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w_out, h_out))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")

    last_good: VideoFrameResult | None = None
    hold = 0
    frame_idx = 0
    max_comp = 0.0
    max_comp_frame = 0
    detected = 0
    press_frames = 0
    confs: list[float] = []
    rest_radii: list[float] = []
    comp_hist: deque[float] = deque(maxlen=5)
    deform_smoother = DeformationSmoother()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_AREA)

        r = analyzer.process_frame(frame, frame_idx / fps) if not hand_press else VideoFrameResult(
            detected=False, status="searching", baseline_locked=True, baseline_mm=baseline_mm
        )

        deform_res: DeformationResult | None = None
        if hand_press:
            hint_cx = last_good.cx if last_good else 0.0
            hint_cy = last_good.cy if last_good else 0.0
            dr = analyze_deformation(
                frame, ball_type, hint_cx, hint_cy, baseline_px * 0.5, require_skin=True
            )
            deform_res = deform_smoother.update(dr)
            dent = deform_res.deform_pct if deform_res.valid else 0.0
            cx_b = deform_res.cx if deform_res.valid else 0.0
            cy_b = deform_res.cy if deform_res.valid else 0.0
            r_px = deform_res.r_baseline * 2 if deform_res.valid else baseline_px
            r = VideoFrameResult(
                detected=bool(deform_res.valid),
                status="compressing" if dent >= IMPACT_STATUS_MIN_PCT else ("tracking" if deform_res.valid else "searching"),
                cx=cx_b,
                cy=cy_b,
                major_px=r_px,
                minor_px=r_px * (1.0 - dent / 100.0) if dent > 0 else r_px,
                angle=0.0,
                vertical_load_px=r_px * (1.0 - dent / 100.0),
                diameter_mm=baseline_mm * (1.0 - dent / 100.0) if dent > 0 else baseline_mm,
                compression_pct=dent,
                raw_compression_pct=dent,
                baseline_locked=True,
                baseline_mm=baseline_mm,
                confidence=0.90 if deform_res.valid else 0.0,
                eccentricity=float(np.sqrt(max(0.0, 1.0 - ((1.0 - dent / 100.0) ** 2)))),
                yolo_active=False,
            )

        has_shape = (
            r.major_px > 10
            and r.cx > 0
            and r.confidence >= 0.55
            and (r.diameter_mm > baseline_mm * 0.40 or hand_press)
        )

        if has_shape:
            last_good = r
            hold = 4
            detected += 1
            raw = float(r.raw_compression_pct)
            comp_hist.append(raw)
            confs.append(r.confidence)
            if raw >= 8.0:
                press_frames += 1
            elif deform_res is not None and deform_res.valid and deform_res.r_baseline > 0:
                rest_radii.append(deform_res.r_baseline)
            if raw >= IMPACT_STATUS_MIN_PCT:
                max_comp = max(max_comp, raw)
                if raw >= max_comp - 0.05:
                    max_comp_frame = frame_idx
        elif last_good is not None and hold > 0 and not hand_press:
            hold -= 1
            r = VideoFrameResult(
                detected=True,
                status="tracking",
                cx=last_good.cx,
                cy=last_good.cy,
                major_px=last_good.major_px,
                minor_px=last_good.minor_px,
                angle=last_good.angle,
                vertical_load_px=last_good.vertical_load_px,
                diameter_mm=baseline_mm,
                compression_pct=0.0,
                raw_compression_pct=0.0,
                baseline_locked=True,
                baseline_mm=baseline_mm,
                confidence=0.60,
                yolo_active=last_good.yolo_active,
            )
            has_shape = True
            comp_hist.append(0.0)

        live_comp = float(r.raw_compression_pct) if has_shape else 0.0
        display = VideoFrameResult(
            **{**r.__dict__, "compression_pct": live_comp, "raw_compression_pct": live_comp}
        )

        if hand_press:
            if deform_res is not None and deform_res.valid:
                draw_deformation(frame, deform_res, baseline_mm, h_out)
            elif r.cx > 0 and r.cy > 0:
                cv2.circle(frame, (int(r.cx), int(r.cy)), baseline_radius, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.drawMarker(frame, (int(r.cx), int(r.cy)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(
                    frame, "0.0% deformed",
                    (max(8, int(r.cx) - 110), max(40, int(r.cy) - baseline_radius - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2, cv2.LINE_AA,
                )
        else:
            draw_measurement(
                frame, display, baseline_radius, h_out, ball_type=ball_type, hand_press=hand_press
            )

        if live_comp >= IMPACT_STATUS_MIN_PCT:
            cv2.putText(frame, "IMPACT", (w_out // 2 - 90, 90), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

        mode = "CV-FAST" if fast_mode else ("YOLO+CV" if r.yolo_active else "CV")
        bounce_count = analyzer.bounce_count
        if live_comp >= IMPACT_STATUS_MIN_PCT:
            status = ("IMPACT — COMPRESSING", (0, 0, 255))
        elif has_shape or (last_good and hold > 0):
            status = ("TRACKING", (0, 255, 0))
        elif not r.baseline_locked:
            status = ("CALIBRATING BASELINE", (0, 200, 255))
        else:
            status = ("SEARCHING", (0, 165, 255))

        t = frame_idx / fps
        panel_bottom = draw_panel(frame, [
            (f"{ball_label} COMPRESSION TEST", (0, 255, 180)),
            (f"Time: {t:5.2f} s   Frame: {frame_idx:4d}  [{mode}]", (240, 240, 240)),
            (f"Baseline diameter: {baseline_mm:.1f} mm", (0, 255, 255)),
            (f"Live compression:  {live_comp:5.1f} %", (0, 255, 255) if live_comp < IMPACT_STATUS_MIN_PCT else (0, 0, 255)),
            (f"Peak compression:  {max_comp:5.1f} %", (0, 255, 255)),
            (f"Bounces detected:  {bounce_count}", (240, 240, 240)),
            (f"Confidence: {r.confidence:.2f}   ecc: {r.eccentricity:.2f}" if has_shape else "Confidence: —", (200, 200, 200)),
            (f"Status: {status[0]}", status[1]),
        ])
        bar_scale = max(20.0, max_comp * 1.25, IMPACT_STATUS_MIN_PCT)
        draw_bar(
            frame, 24, panel_bottom + 28, 380, 18,
            live_comp / bar_scale,
            (0, 0, 255) if live_comp >= IMPACT_STATUS_MIN_PCT else (0, 220, 0),
        )

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 40 == 0:
            print(f"  {frame_idx}/{total}", flush=True)

    cap.release()
    writer.release()

    det_rate = round(detected / max(frame_idx, 1) * 100, 1)
    mean_conf = float(np.mean(confs)) * 100.0 if confs else 0.0
    if rest_radii:
        arr = np.asarray(rest_radii, dtype=np.float64)
        stab = max(0.0, 100.0 - float(np.std(arr) / (np.mean(arr) + 1e-6) * 400.0))
    else:
        stab = det_rate
    metrics = {
        "baseline_mm": baseline_mm,
        "max_compression_pct": round(max_comp, 1),
        "max_compression_frame": max_comp_frame,
        "bounce_count": 0 if hand_press else analyzer.bounce_count,
        "press_frames": press_frames,
        "frames": frame_idx,
        "detected_frames": detected,
        "detection_rate_pct": det_rate,
        "confidence_pct": round(mean_conf, 1),
        "stability_pct": round(stab, 1),
        "output": str(out_path),
        "fps": round(fps, 2),
        "mode": "hand_press" if hand_press else "bounce",
    }

    print(
        f"Baseline: {baseline_mm:.1f} mm | Peak outline deform: {max_comp:.1f}% @ frame {max_comp_frame} | "
        f"Bounces: {analyzer.bounce_count} | Detected: {detected}/{frame_idx} ({det_rate}%) | "
        f"Engine: {'CV-FAST' if fast_mode else ('YOLO+CV' if analyzer.yolo_active else 'CV')}"
    )
    return metrics

def main() -> int:
    video = ROOT / "data" / "samples" / "latest testing video.mp4"
    ball_type = "tennis"
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    if len(sys.argv) > 2:
        ball_type = sys.argv[2]
    if not video.exists():
        print(f"Video not found: {video}")
        return 1

    stem = video.stem.replace(" ", "-").replace("---", "-").lower()
    out = ROOT / "data" / "reports" / f"{stem}-tracked.mp4"
    print(f"Input:  {video}\nOutput: {out}\nBall:   {ball_type}\n")
    export(video, out, ball_type=ball_type)
    print(f"\nOutput video: {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
