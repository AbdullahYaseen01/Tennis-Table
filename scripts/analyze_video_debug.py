from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.vision.video_measure import VideoBallAnalyzer

video = ROOT / "data" / "samples" / "latest testing video.mp4"
frames = [73, 84, 87, 94, 97, 108, 125]

analyzer = VideoBallAnalyzer(ball_type="tennis", use_yolo=True)
analyzer.lock_baseline_from_video(str(video))
print(f"Baseline px: {analyzer.baseline_vertical_px}")

cap = cv2.VideoCapture(str(video))
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx in frames:
        r = analyzer.process_frame(frame, idx / 30.0)
        print(
            f"frame {idx:3d}  det={r.detected}  st={r.status:12s}  "
            f"cx={r.cx:6.1f} cy={r.cy:6.1f}  maj={r.major_px:5.1f} min={r.minor_px:5.1f}  "
            f"comp={r.compression_pct:5.1f}%  bounces={analyzer.bounce_count}"
        )
        out = ROOT / "data" / "reports" / f"dbg-{idx}.jpg"
        cv2.circle(frame, (int(r.cx), int(r.cy)), int(analyzer.baseline_vertical_px / 2), (255, 180, 0), 2)
        cv2.ellipse(frame, (int(r.cx), int(r.cy)), (int(r.major_px/2), int(r.minor_px/2)), r.angle, 0, 360, (0, 255, 0), 2)
        cv2.imwrite(str(out), frame)
    else:
        analyzer.process_frame(frame, idx / 30.0)
    idx += 1
cap.release()
print(f"Final bounces: {analyzer.bounce_count}")
