"""High-accuracy bounce compression tracker + annotated video."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vision.bounce_measure import (
    CompressionSmoother,
    TemporalHeightFilter,
    analyze_ball_in_frame,
    assess_frame_quality,
    compute_baseline_from_video,
)

KNOWN_DIAMETER_MM = 67.0


def compute_baseline(video_path: Path, ball_type: str = "tennis") -> tuple[float, float]:
    cap = cv2.VideoCapture(str(video_path))
    baseline_px, px_per_mm = compute_baseline_from_video(cap, ball_type=ball_type)
    cap.release()
    return baseline_px, px_per_mm


def draw_panel(frame, lines):
    pad, line_h, width = 14, 32, 380
    height = len(lines) * line_h + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + width, 10 + height), (0, 210, 255), 2)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10 + pad, 10 + pad + (i + 1) * line_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw_bar(frame, x, y, w, h, frac, color):
    frac = float(np.clip(frac, 0, 1))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cv2.rectangle(frame, (x, y), (x + int(w * frac), y + h), color, -1)


def export(
    video_path: Path,
    out_avi: Path,
    *,
    ball_type: str = "tennis",
    fixed_baseline_px: float | None = None,
    fixed_px_per_mm: float | None = None,
):
    if fixed_baseline_px and fixed_px_per_mm:
        baseline_px = fixed_baseline_px
        px_per_mm = fixed_px_per_mm
    else:
        baseline_px, px_per_mm = compute_baseline(video_path, ball_type=ball_type)
    baseline_mm = baseline_px / px_per_mm

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    quality = assess_frame_quality(w, h)
    low_q = quality["low_quality"]

    out_avi.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_avi), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")

    smoother = CompressionSmoother()
    height_filter = TemporalHeightFilter()
    max_comp = 0.0
    bounce_count = 0
    in_compression = False
    frame_idx = 0
    detected_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = analyze_ball_in_frame(
            frame,
            ball_type=ball_type,
            baseline_minor_px=baseline_px,
            px_per_mm=px_per_mm,
        )
        t = frame_idx / fps
        live_comp = 0.0
        valid = r.get("detected", False)

        filtered_h = height_filter.update(r.get("height_px") if valid else None, valid)
        if filtered_h is not None and baseline_px > 0:
            raw_comp = max(0.0, (baseline_px - filtered_h) / baseline_px * 100.0)
            live_comp = smoother.update(raw_comp)
            max_comp = smoother.max_compression
            valid = True
            detected_frames += 1

            if live_comp > 10.0 and not in_compression:
                in_compression = True
                bounce_count += 1
            elif live_comp < 4.0 and in_compression:
                in_compression = False

        if r.get("ellipse") and valid:
            ell = r["ellipse"]
            cv2.ellipse(
                frame,
                ((ell["cx"], ell["cy"]), (ell["a"] * 2, ell["b"] * 2), ell["angle"]),
                (0, 255, 0), 2 if low_q else 3, cv2.LINE_AA,
            )
            ex, ey = int(r.get("cx", ell["cx"])), int(r.get("cy", ell["cy"]))
            color = (0, 0, 255) if live_comp > 15 else (0, 255, 0)
            cv2.putText(frame, f"Compression {live_comp:.1f}%",
                        (max(5, ex - 60), max(34, ey - int(ell.get("b", 20)) - 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65 if low_q else 0.75, color, 2, cv2.LINE_AA)

        mode_tag = "LQ+ENH" if low_q else "HQ"
        status = ("IMPACT - COMPRESSING", (0, 0, 255)) if live_comp > 6 else \
                 ("TRACKING", (0, 255, 0)) if valid else ("SEARCHING", (0, 165, 255))

        draw_panel(frame, [
            ("BALL COMPRESSION TEST", (0, 255, 180)),
            (f"Time: {t:5.2f} s  [{mode_tag}]", (240, 240, 240)),
            (f"Baseline diameter: {baseline_mm:.1f} mm", (0, 255, 255)),
            (f"Live compression:  {live_comp:.1f} %", (0, 255, 255) if live_comp <= 15 else (0, 0, 255)),
            (f"Max compression:   {max_comp:.1f} %", (0, 255, 255)),
            (f"Bounces detected:  {bounce_count}", (240, 240, 240)),
            (f"Status: {status[0]}", status[1]),
        ])
        draw_bar(frame, 24, 10 + 7 * 32 + 20, 330, 16, live_comp / 40.0,
                 (0, 0, 255) if live_comp > 15 else (0, 220, 0))

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 40 == 0:
            print(f"  {frame_idx}/{total}", flush=True)

    cap.release()
    writer.release()
    print(f"Baseline: {baseline_mm:.1f} mm | Max compression: {max_comp:.1f}% | Bounces: {bounce_count} | Detected: {detected_frames}/{frame_idx}")
    return {
        "baseline_mm": round(baseline_mm, 1),
        "max_compression_pct": round(max_comp, 1),
        "bounce_count": bounce_count,
        "frames": frame_idx,
        "detected_frames": detected_frames,
        "detection_rate_pct": round(detected_frames / max(frame_idx, 1) * 100, 1),
        "low_quality_mode": low_q,
        "ball_type": ball_type,
    }


def main() -> int:
    video = ROOT / "data" / "samples" / "Tennis Ball Bounce - Trim.mp4"
    ball_type = "tennis"
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    if len(sys.argv) > 2:
        ball_type = sys.argv[2]
    if not video.exists():
        print(f"Video not found: {video}")
        return 1
    stem = video.stem.replace(" ", "-").replace("---", "-").lower()
    out = ROOT / "data" / "reports" / f"{stem}-tracked.avi"
    print(f"Input:  {video}\nOutput: {out}\nBall:   {ball_type}\n")
    export(video, out, ball_type=ball_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
