"""Single-frame and multi-frame analysis for live webcam accuracy."""
from __future__ import annotations

import cv2
import numpy as np

from app.vision.bounce_measure import (
    BASELINE_TARGET_FRAMES,
    MIN_BASELINE_GOOD_FRAMES,
    analyze_ball_in_frame,
    compute_baseline_from_results,
)

# Re-export for backwards compatibility
BASELINE_TARGET_FRAMES = BASELINE_TARGET_FRAMES
MIN_BASELINE_GOOD_FRAMES = MIN_BASELINE_GOOD_FRAMES


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def analyze_jpeg_bytes(
    jpeg: bytes,
    baseline_h_px: float | None = None,
    px_per_mm: float | None = None,
    ball_type: str = "tennis",
) -> dict:
    frame = decode_image(jpeg)
    if frame is None:
        return {"detected": False, "error": "invalid_image", "status": "searching"}
    baseline_minor = baseline_h_px
    return analyze_ball_in_frame(
        frame,
        ball_type=ball_type,
        baseline_minor_px=baseline_minor,
        px_per_mm=px_per_mm,
    )


def compute_baseline_from_frames(frame_results: list[dict], ball_type: str = "tennis") -> dict:
    return compute_baseline_from_results(frame_results, ball_type=ball_type)
