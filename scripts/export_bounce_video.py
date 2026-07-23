"""High-accuracy bounce compression tracker — colour detect + YOLO ROI + sub-pixel refine."""
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
from app.vision.video_measure import VideoBallAnalyzer, VideoFrameResult, FLOOR_CONTACT_Y_FRAC

VERCEL_MAX_EDGE = 960


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


def draw_measurement(frame, r: VideoFrameResult, baseline_radius: int, frame_h: int) -> None:
    has_shape = r.major_px > 10 and r.cx > 0 and r.cy > 0
    if not has_shape:
        return
    cx, cy = int(r.cx), int(r.cy)
    if cy < 15 or cy > frame_h - 15:
        return

    cv2.circle(frame, (cx, cy), baseline_radius, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.ellipse(
        frame,
        (cx, cy),
        (max(1, int(r.major_px / 2)), max(1, int(r.minor_px / 2))),
        r.angle,
        0, 360,
        (0, 255, 0), 3, cv2.LINE_AA,
    )
    cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    if r.edge_points is not None and len(r.edge_points) > 0:
        for pt in r.edge_points[::8]:
            cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 2, (0, 200, 255), -1)

    comp = r.compression_pct
    color = (0, 0, 255) if comp > 8 else (0, 255, 255) if comp > 3 else (0, 255, 0)
    cv2.putText(
        frame, f"{comp:.1f}% compressed",
        (max(8, cx - 110), max(40, cy - baseline_radius - 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"D = {r.diameter_mm:.1f} mm",
        (max(8, cx - 75), min(frame_h - 12, cy + baseline_radius + 30)),
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
    if use_yolo is None:
        use_yolo = not fast_mode

    analyzer = VideoBallAnalyzer(ball_type=ball_type, use_yolo=use_yolo, fast_mode=fast_mode)

    print("  Locking baseline from in-air frames...", flush=True)
    locked = analyzer.lock_baseline_from_video(str(video_path))
    if not locked and fixed_baseline_px and fixed_baseline_px > 0:
        analyzer.baseline_vertical_px = fixed_baseline_px
        analyzer._px_per_mm = fixed_px_per_mm or (fixed_baseline_px / analyzer.known_mm)
        analyzer.calibrator.pixels_per_mm = analyzer._px_per_mm
        analyzer.baseline_locked = True
        from app.vision.video_measure import BounceCounter, ContactPhaseTracker

        analyzer.bounce_counter = BounceCounter(analyzer._frame_h or 720, analyzer._fps)
        analyzer._contact = ContactPhaseTracker(analyzer._frame_h or 720)
    if not analyzer.baseline_locked:
        raise RuntimeError("Could not lock baseline from video")

    baseline_px = analyzer.baseline_vertical_px or 320.0
    baseline_mm = analyzer.known_mm
    baseline_radius = int(baseline_px / 2)

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
    comp_hist: deque[float] = deque(maxlen=5)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_AREA)

        r = analyzer.process_frame(frame, frame_idx / fps)
        has_shape = (
            r.major_px > 10
            and r.cx > 0
            and r.confidence >= 0.65
            and (r.diameter_mm > baseline_mm * 0.45 or r.cy > h_out * 0.62)
        )

        if has_shape:
            last_good = r
            hold = 4
            detected += 1
            comp_hist.append(r.raw_compression_pct)
            raw = r.raw_compression_pct
            if r.cy > h_out * FLOOR_CONTACT_Y_FRAC and r.compression_pct > 0:
                max_comp = max(max_comp, raw)
                if raw >= max_comp - 0.05:
                    max_comp_frame = frame_idx
        elif last_good is not None and hold > 0:
            hold -= 1
            r = VideoFrameResult(
                detected=False,
                status="tracking",
                cx=last_good.cx,
                cy=last_good.cy,
                major_px=last_good.major_px,
                minor_px=last_good.minor_px,
                angle=last_good.angle,
                vertical_load_px=last_good.vertical_load_px,
                diameter_mm=last_good.diameter_mm,
                compression_pct=0.0,
                raw_compression_pct=0.0,
                baseline_locked=True,
                baseline_mm=baseline_mm,
                yolo_active=last_good.yolo_active,
            )
            comp_hist.append(0.0)
            detected += 1

        live_comp = _smooth_comp(comp_hist) if has_shape or hold > 0 else 0.0
        display = VideoFrameResult(
            **{**r.__dict__, "compression_pct": live_comp if has_shape else r.compression_pct}
        )
        if has_shape:
            display.compression_pct = live_comp

        draw_measurement(frame, display, baseline_radius, h_out)

        if live_comp > 8:
            cv2.putText(frame, "IMPACT", (w_out // 2 - 90, 90), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

        mode = "CV-FAST" if fast_mode else ("YOLO+CV" if r.yolo_active else "CV")
        bounce_count = analyzer.bounce_count
        if live_comp > 8:
            status = ("IMPACT — COMPRESSING", (0, 0, 255))
        elif has_shape or (last_good and hold > 0):
            status = ("TRACKING", (0, 255, 0))
        elif not r.baseline_locked:
            status = ("CALIBRATING BASELINE", (0, 200, 255))
        else:
            status = ("SEARCHING", (0, 165, 255))

        t = frame_idx / fps
        panel_bottom = draw_panel(frame, [
            ("TENNIS BALL COMPRESSION TEST", (0, 255, 180)),
            (f"Time: {t:5.2f} s   Frame: {frame_idx:4d}  [{mode}]", (240, 240, 240)),
            (f"Baseline diameter: {baseline_mm:.1f} mm", (0, 255, 255)),
            (f"Live compression:  {live_comp:5.1f} %", (0, 255, 255) if live_comp <= 8 else (0, 0, 255)),
            (f"Peak compression:  {max_comp:5.1f} %", (0, 255, 255)),
            (f"Bounces detected:  {bounce_count}", (240, 240, 240)),
            (f"Confidence: {r.confidence:.2f}   ecc: {r.eccentricity:.2f}" if has_shape else "Confidence: —", (200, 200, 200)),
            (f"Status: {status[0]}", status[1]),
        ])
        bar_scale = max(20.0, max_comp * 1.25, 12.0)
        draw_bar(
            frame, 24, panel_bottom + 28, 380, 18,
            live_comp / bar_scale,
            (0, 0, 255) if live_comp > 8 else (0, 220, 0),
        )

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 40 == 0:
            print(f"  {frame_idx}/{total}", flush=True)

    cap.release()
    writer.release()

    det_rate = round(detected / max(frame_idx, 1) * 100, 1)
    print(
        f"Baseline: {baseline_mm:.1f} mm | Peak: {max_comp:.1f}% @ frame {max_comp_frame} | "
        f"Bounces: {analyzer.bounce_count} | Detected: {detected}/{frame_idx} ({det_rate}%) | "
        f"Engine: {'CV-FAST' if fast_mode else ('YOLO+CV' if analyzer.yolo_active else 'CV')}"
    )
    return {
        "baseline_mm": baseline_mm,
        "max_compression_pct": round(max_comp, 1),
        "bounce_count": analyzer.bounce_count,
        "frames": frame_idx,
        "detected_frames": detected,
        "detection_rate_pct": det_rate,
        "output": str(out_path),
    }


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
