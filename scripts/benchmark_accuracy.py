"""Benchmark measurement accuracy on sample video or file."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from app.config import REFERENCE_DIAMETERS_MM, SAMPLES_DIR
from app.vision.calibration import Calibrator
from app.vision.camera import FrameSource
from app.vision.detect import BallDetector


def benchmark_video(path: Path, ball_type: str = "tennis", n_frames: int = 90) -> dict:
    cal = Calibrator()
    det = BallDetector(cal)
    det.set_ball_type(ball_type)

    known_mm = REFERENCE_DIAMETERS_MM.get(ball_type, 67.0)
    src = FrameSource(str(path))

    minors: list[float] = []
    majors: list[float] = []
    confs: list[float] = []
    frames_collected: list = []

    for _ in range(n_frames):
        ok, frame, ts = src.read()
        if not ok or frame is None:
            break
        frames_collected.append(frame)
        m = det.detect(frame, ts)
        if m:
            minors.append(m.minor_mm)
            majors.append(m.major_mm)
            confs.append(m.detection_confidence)

    src.release()

    # Auto-calibrate scale from median pixel diameter if not calibrated
    if not cal.is_ready() or cal.pixels_per_mm is None:
        d_px = det.measure_median_diameter_px(frames_collected[:60], axis="minor")
        if d_px:
            cal.calibrate_scale_from_ball_diameter(d_px, known_mm)
            # Re-run detection with correct scale
            minors, majors, confs = [], [], []
            src2 = FrameSource(str(path))
            for _ in range(n_frames):
                ok, frame, ts = src2.read()
                if not ok:
                    break
                m = det.detect(frame, ts)
                if m:
                    minors.append(m.minor_mm)
                    majors.append(m.major_mm)
                    confs.append(m.detection_confidence)
            src2.release()

    if len(minors) < 10:
        return {"error": "Insufficient detections", "detections": len(minors)}

    # Resting frames only (first ~2 s at 30 fps)
    rest_n = min(60, len(minors))
    rest_minors = minors[:rest_n]
    rest_arr = np.array(rest_minors)

    minor_arr = np.array(minors)
    major_arr = np.array(majors)
    mean_minor = float(np.mean(minor_arr))
    mean_major = float(np.mean(major_arr))
    std_minor = float(np.std(minor_arr))
    err_minor = (mean_minor - known_mm) / known_mm * 100
    err_major = (mean_major - known_mm) / known_mm * 100

    val = cal.validate_with_measurement((mean_minor + mean_major) / 2, known_mm)

    return {
        "video": str(path),
        "ball_type": ball_type,
        "known_mm": known_mm,
        "detections": len(minors),
        "rest_detections": len(rest_minors),
        "mean_minor_mm": mean_minor,
        "mean_major_mm": mean_major,
        "rest_mean_minor_mm": float(np.mean(rest_arr)),
        "rest_std_minor_mm": float(np.std(rest_arr)),
        "rest_jitter_pct": float(np.std(rest_arr)) / known_mm * 100,
        "rest_error_pct": (float(np.mean(rest_arr)) - known_mm) / known_mm * 100,
        "std_minor_mm": std_minor,
        "jitter_pct": std_minor / known_mm * 100,
        "error_minor_pct": err_minor,
        "error_major_pct": err_major,
        "validation_error_pct": val.error_pct,
        "mean_confidence": float(np.mean(confs)),
        "yolo_enabled": det._use_yolo,
        "scale_px_per_mm": cal.pixels_per_mm,
    }


def main() -> int:
    video = SAMPLES_DIR / "sample_test.mp4"
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])

    print(f"Accuracy benchmark: {video}\n")
    result = benchmark_video(video)

    if "error" in result:
        print(f"FAILED: {result}")
        return 1

    print(f"  Detections:      {result['detections']}")
    print(f"  Known diameter:  {result['known_mm']:.2f} mm")
    print(f"  Scale:           {result['scale_px_per_mm']:.3f} px/mm")
    print(f"  Resting mean:    {result['rest_mean_minor_mm']:.3f} mm")
    print(f"  Resting jitter:  {result['rest_std_minor_mm']:.4f} mm ({result['rest_jitter_pct']:.3f}%)")
    print(f"  Resting error:   {result['rest_error_pct']:+.3f}%")
    print(f"  Full-video mean: {result['mean_minor_mm']:.3f} mm")
    print(f"  Full-video err:  {result['validation_error_pct']:+.3f}%")
    print(f"  Mean confidence: {result['mean_confidence']:.3f}")
    print(f"  YOLO ROI:        {result['yolo_enabled']}")
    print()
    ok = abs(result["rest_error_pct"]) < 2.0 and result["rest_jitter_pct"] < 1.5
    print("PASS" if ok else "DONE (check calibration for your video)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
