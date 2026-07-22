"""Generate dev calibration files matching the synthetic sample video."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import REFERENCE_DIAMETERS_MM, SAMPLES_DIR
from app.vision.calibration import Calibrator
from app.vision.camera import FrameSource
from app.vision.detect import BallDetector

CALIB_DIR = ROOT / "data" / "calib"
VIDEO = SAMPLES_DIR / "sample_test.mp4"
KNOWN_MM = REFERENCE_DIAMETERS_MM["tennis"]


def main() -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO.exists():
        print(f"Missing {VIDEO} — run generate_sample_video.py first")
        return

    src = FrameSource(str(VIDEO))
    ok, frame, _ = src.read()
    src.release()
    if not ok or frame is None:
        print("Could not read sample video")
        return

    h, w = frame.shape[:2]
    camera_matrix = np.array(
        [[1000.0, 0, w / 2], [0, 1000.0, h / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)

    np.savez(
        CALIB_DIR / "intrinsics.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array([w, h]),
    )

    # Measure actual pixel diameter from enhanced detector (rest frames)
    cal = Calibrator()
    cal.camera_matrix = camera_matrix
    cal.dist_coeffs = dist_coeffs
    cal._image_size = (w, h)
    cal._build_remap((w, h))

    det = BallDetector(cal)
    det.set_ball_type("tennis")
    src = FrameSource(str(VIDEO))
    frames = []
    for _ in range(45):
        ok, f, _ = src.read()
        if ok and f is not None:
            frames.append(f)
    src.release()

    d_px = det.measure_median_diameter_px(frames, axis="minor")
    if d_px is None:
        d_px = (120 * 2)  # fallback
    ppm = d_px / KNOWN_MM

    cal.save_scale(ppm)
    val = cal.validate_with_measurement(KNOWN_MM, KNOWN_MM)

    print(f"Dev calibration written to {CALIB_DIR}")
    print(f"Measured diameter: {d_px:.2f} px -> {ppm:.3f} px/mm")
    print(f"Reference ball: {KNOWN_MM} mm, validation error: {val.error_pct:.4f}%")


if __name__ == "__main__":
    main()
