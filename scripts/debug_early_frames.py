from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.vision.bounce_measure import analyze_ball_in_frame
from app.vision.video_measure import VideoBallAnalyzer

video = ROOT / "data" / "samples" / "latest testing video.mp4"
a = VideoBallAnalyzer(use_yolo=True)
a.lock_baseline_from_video(str(video))
cap = cv2.VideoCapture(str(video))
for idx in range(45):
    ok, f = cap.read()
    if not ok:
        break
    a._yolo_age = 0  
    r = a.process_frame(f)
    bm = analyze_ball_in_frame(
        f, ball_type="tennis",
        baseline_minor_px=a.baseline_vertical_px,
        px_per_mm=a._px_per_mm,
    )
    hint = a._yolo.detect_roi(f) if a._yolo else None
    yolo = "none" if hint is None else f"{hint.center[0]:.0f},{hint.center[1]:.0f} r={hint.radius:.0f}"
    bmc = bm.get("cx") or 0
    bmcy = bm.get("cy") or 0
    print(
        f"{idx:3d} out=({r.cx:5.0f},{r.cy:5.0f}) bm=({bmc:5.0f},{bmcy:5.0f}) "
        f"D={r.diameter_mm:4.1f} maj={r.major_px:.0f} min={r.minor_px:.0f} "
        f"comp={r.compression_pct:4.1f} yolo={yolo}"
    )
    if idx in (30, 35, 40):
        out = ROOT / "data" / "reports" / f"dbg-early-{idx}.jpg"
        cv2.circle(f, (int(r.cx), int(r.cy)), int(a.baseline_vertical_px / 2), (255, 180, 0), 2)
        cv2.drawMarker(f, (int(r.cx), int(r.cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if hint:
            cv2.circle(f, (int(hint.center[0]), int(hint.center[1])), int(hint.radius), (255, 0, 255), 1)
        cv2.imwrite(str(out), f)
cap.release()
