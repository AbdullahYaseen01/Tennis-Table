from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.vision.ball_profiles import PICKLEBALL_DIAMETER_MM, get_profile
from app.vision.bounce_measure import analyze_ball_in_frame, compute_baseline_from_video
from app.vision.video_measure import VideoBallAnalyzer

REPORT_DIR = ROOT / "data" / "reports"

def _auto_tune_hsv(frames: list[np.ndarray], ball_type: str) -> dict | None:
    
    hs, ss, vs = [], [], []
    for frame in frames[:40]:
        h_img, w_img = frame.shape[:2]
        cx, cy = w_img // 2, h_img // 2
        r = min(w_img, h_img) // 5
        x0, y0 = max(0, cx - r), max(0, cy - r)
        crop = frame[y0 : y0 + 2 * r, x0 : x0 + 2 * r]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        bright = v > 100
        if np.count_nonzero(bright) < 50:
            continue
        hs.extend(h[bright].ravel().tolist())
        ss.extend(s[bright].ravel().tolist())
        vs.extend(v[bright].ravel().tolist())
    if len(hs) < 100:
        return None
    h_arr, s_arr, v_arr = np.array(hs), np.array(ss), np.array(vs)
    return {
        "ball_type": ball_type,
        "hsv_lower": [
            int(max(0, np.percentile(h_arr, 5) - 8)),
            int(max(40, np.percentile(s_arr, 5) - 20)),
            int(max(80, np.percentile(v_arr, 5) - 25)),
        ],
        "hsv_upper": [
            int(min(179, np.percentile(h_arr, 95) + 8)),
            255,
            255,
        ],
        "samples": len(hs),
    }

def validate_video(video_path: Path, ball_type: str = "pickleball") -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    bpx, ppm = compute_baseline_from_video(cap, ball_type=ball_type)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    analyzer = VideoBallAnalyzer(ball_type=ball_type, use_yolo=True, fast_mode=False)
    if bpx > 0:
        analyzer.baseline_vertical_px = bpx
        analyzer._px_per_mm = ppm
        analyzer.baseline_locked = True
        analyzer.known_mm = PICKLEBALL_DIAMETER_MM

    detected = 0
    compressed_frames = 0
    max_comp = 0.0
    max_comp_frame = 0
    sample_frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if len(sample_frames) < 50:
            sample_frames.append(frame.copy())

        r = analyzer.process_frame(frame, frame_idx / fps)
        bm = analyze_ball_in_frame(
            frame,
            ball_type=ball_type,
            baseline_minor_px=analyzer.baseline_vertical_px,
            px_per_mm=analyzer._px_per_mm,
        )
        if r.detected or bm.get("detected") or bm.get("ellipse"):
            detected += 1
        if r.compression_pct > max_comp:
            max_comp = r.compression_pct
            max_comp_frame = frame_idx
        if r.compression_pct > 8:
            compressed_frames += 1
        frame_idx += 1

    cap.release()
    tune = _auto_tune_hsv(sample_frames, ball_type)
    profile = get_profile(ball_type)

    report = {
        "video": video_path.name,
        "ball_type": ball_type,
        "profile": profile.name,
        "diameter_mm": profile.diameter_mm,
        "frames": frame_idx,
        "detected_frames": detected,
        "detection_rate_pct": round(detected / max(frame_idx, 1) * 100, 1),
        "baseline_px": round(bpx, 1) if bpx else None,
        "px_per_mm": round(ppm, 4) if ppm else None,
        "baseline_mm": round(bpx / ppm, 1) if bpx and ppm else profile.diameter_mm,
        "max_compression_pct": round(max_comp, 1),
        "max_compression_frame": max_comp_frame,
        "compressed_frames": compressed_frames,
        "bounces": analyzer.bounce_count,
        "resolution": f"{w}x{h}",
        "fps": round(fps, 2),
        "suggested_hsv_tune": tune,
    }
    return report

def main() -> int:
    ball_type = "pickleball"
    video = ROOT / "data" / "samples" / "pickleball test.mp4"

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        video = Path(args[0])
    if "--variant" in sys.argv:
        idx = sys.argv.index("--variant")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "indoor":
            ball_type = "pickleball_indoor"

    if not video.exists():
        print(f"Video not found: {video}")
        print("\nAdd your pickleball bounce video to:")
        print(f"  {ROOT / 'data' / 'samples' / 'pickleball test.mp4'}")
        print("\nOr run:")
        print("  python scripts/train_pickleball.py path/to/your_video.mp4")
        return 1

    print(f"Training / validating: {ball_type}")
    print(f"Video: {video}\n")

    report = validate_video(video, ball_type=ball_type)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / f"pickleball-training-{video.stem.replace(' ', '-')}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 50)
    print("PICKLEBALL TRAINING REPORT")
    print("=" * 50)
    print(f"  Frames:           {report['frames']}")
    print(f"  Detection rate:   {report['detection_rate_pct']}%")
    print(f"  Baseline:         {report['baseline_mm']} mm ({report['baseline_px']} px)")
    print(f"  Peak compression: {report['max_compression_pct']}% @ frame {report['max_compression_frame']}")
    print(f"  Bounces:          {report['bounces']}")
    print(f"  Report saved:     {out_json}")

    if report["suggested_hsv_tune"]:
        t = report["suggested_hsv_tune"]
        print(f"\n  Suggested HSV (from video): lower={t['hsv_lower']} upper={t['hsv_upper']}")

    if report["detection_rate_pct"] < 70:
        print("\n  WARNING: Detection rate low — use dark background, neon pickleball, fixed camera.")
        return 1

    print("\n  OK — run export:")
    print(f'  python scripts/export_bounce_video.py "{video}" {ball_type}')
    return 0

if __name__ == "__main__":
    sys.exit(main())
