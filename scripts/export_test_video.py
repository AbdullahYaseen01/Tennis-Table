from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from app.config import COLOR_PROFILES, REFERENCE_DIAMETERS_MM, SAMPLES_DIR

def resize_keep_aspect(frame: np.ndarray, max_side: int = 720) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def detect_ball_fast(frame: np.ndarray, ball_type: str = "tennis") -> dict | None:
    
    profile = COLOR_PROFILES.get(ball_type, COLOR_PROFILES["tennis"])
    lower = np.array(profile["hsv_lower"], dtype=np.uint8)
    upper = np.array(profile["hsv_upper"], dtype=np.uint8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = frame.shape[:2]
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200 or area > h * w * 0.55:
            continue
        if len(c) < 5:
            continue
        if area > best_area:
            best_area = area
            best = c

    if best is None:
        return None

    try:
        ellipse = cv2.fitEllipse(best)
    except cv2.error:
        return None

    (cx, cy), (axis_a, axis_b), angle = ellipse
    major = max(axis_a, axis_b)
    minor = min(axis_a, axis_b)
    if minor < 15:
        return None
    return {
        "center": (float(cx), float(cy)),
        "major_px": float(major),
        "minor_px": float(minor),
        "angle": float(angle),
        "area": float(best_area),
    }

def draw_panel(frame: np.ndarray, lines: list[str], x: int = 12, y: int = 12) -> None:
    
    pad = 10
    line_h = 26
    width = max(len(s) for s in lines) * 11 + pad * 2
    height = len(lines) * line_h + pad * 2
    x2 = min(frame.shape[1] - 4, x + width)
    y2 = min(frame.shape[0] - 4, y + height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 200, 255), 2)

    for i, text in enumerate(lines):
        color = (0, 255, 180) if i == 0 else (240, 240, 240)
        if "Max compression" in text or "Baseline" in text:
            color = (0, 255, 255)
        cv2.putText(
            frame,
            text,
            (x + pad, y + pad + (i + 1) * line_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

def export_annotated(
    video_path: Path,
    output_path: Path,
    report: dict | None = None,
    ball_type: str = "tennis",
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError("Cannot read first frame")

    first = resize_keep_aspect(first, 720)
    h, w = first.shape[:2]
    known_mm = REFERENCE_DIAMETERS_MM.get(ball_type, 67.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    
    if output_path.suffix.lower() == ".mp4":
        avi_path = output_path.with_suffix(".avi")
    else:
        avi_path = output_path

    writer = cv2.VideoWriter(
        str(avi_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h)
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
        )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed for {avi_path}")
    output_path = avi_path

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    
    rest_px: list[float] = []
    baseline_px: float | None = None
    ppm: float | None = None
    max_comp = 0.0
    frame_idx = 0
    detected = 0

    report = report or {}
    baseline_mm_report = report.get("baseline_mm")
    max_comp_report = report.get("max_compression_pct")
    rest_err = report.get("rest_error_pct")
    det_rate = report.get("detection_rate_pct")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = resize_keep_aspect(frame, 720)
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))

        t = frame_idx / fps
        meas = detect_ball_fast(frame, ball_type)
        live_comp = 0.0
        dia_mm = None

        if meas:
            detected += 1
            cx, cy = meas["center"]
            major, minor = meas["major_px"], meas["minor_px"]
            cv2.ellipse(
                frame,
                (int(cx), int(cy)),
                (int(major / 2), int(minor / 2)),
                meas["angle"],
                0,
                360,
                (0, 255, 0),
                2,
            )

            
            if t < 2.0 and minor > 20:
                rest_px.append(minor)
                if len(rest_px) >= 12 and ppm is None:
                    med = float(np.median(rest_px))
                    ppm = med / known_mm
                    baseline_px = med

            if ppm and baseline_px:
                dia_mm = minor / ppm
                live_comp = max(0.0, (baseline_px - minor) / baseline_px * 100.0)
                max_comp = max(max_comp, live_comp)

            label = f"Ball  {minor:.0f}x{major:.0f}px"
            if dia_mm is not None:
                label += f"  D={dia_mm:.1f}mm  Comp={live_comp:.1f}%"
            cv2.putText(
                frame,
                label,
                (int(cx - 80), max(30, int(cy - major / 2 - 12))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        
        bl = baseline_mm_report if baseline_mm_report is not None else (
            round(baseline_px / ppm, 1) if (baseline_px and ppm) else None
        )
        mc = max_comp_report if max_comp_report is not None else round(max_comp, 1)
        lines = [
            "TENNIS BALL CONDITION TEST",
            f"Video: {video_path.name}",
            f"Time: {t:5.2f}s   Frame: {frame_idx}",
            f"Baseline: {bl if bl is not None else 'â€”'} mm",
            f"Max compression: {mc if mc is not None else 'â€”'} %",
            f"Live compression: {live_comp:.1f}%",
            f"Rest error vs 67mm: {rest_err:+.1f}%" if rest_err is not None else "Rest error: â€”",
            f"Detection rate: {det_rate}%" if det_rate is not None else f"Detected frames: {detected}",
            "Status: COMPLETE" if report.get("test_completed") else "Status: PROCESSING",
        ]
        draw_panel(frame, lines, x=12, y=12)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 40 == 0:
            print(f"  wrote {frame_idx}/{total or '?'} frames", flush=True)

    cap.release()
    writer.release()

    size_kb = output_path.stat().st_size / 1024
    print(f"Done: {frame_idx} frames -> {output_path} ({size_kb:.0f} KB)")
    if frame_idx < 5 or size_kb < 50:
        raise RuntimeError("Output video looks empty â€” writer failed")
    return output_path

def main() -> int:
    video = SAMPLES_DIR / "Test video.mp4"
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    if not video.exists():
        print(f"Video not found: {video}")
        return 1

    stem = video.stem.replace(" ", "-").lower()
    out = ROOT / "data" / "reports" / f"{stem}-annotated.mp4"
    if len(sys.argv) > 2:
        out = Path(sys.argv[2])

    report_path = ROOT / "data" / "reports" / f"{stem}-test-report.json"
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    print(f"Input:  {video}")
    print(f"Output: {out}")
    export_annotated(video, out, report=report)
    return 0

if __name__ == "__main__":
    sys.exit(main())
