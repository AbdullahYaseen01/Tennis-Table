"""Generate synthetic test video for development without a camera."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_test.mp4"

# Video parameters
WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION_S = 12
BASE_RADIUS_PX = 120  # ~67mm ball at ~3.58 px/mm
PX_PER_MM = (BASE_RADIUS_PX * 2) / 67.0  # reference scale for dev calibration


def draw_ball(frame: np.ndarray, cx: float, cy: float, rx: float, ry: float, angle: float = 0) -> None:
    """Draw a fuzzy yellow-green tennis ball."""
    overlay = frame.copy()
    cv2.ellipse(overlay, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, (40, 200, 220), -1)
    # Fuzz texture
    noise = np.random.randint(-15, 15, overlay.shape, dtype=np.int16)
    ball_mask = (overlay > 0).all(axis=2)
    blended = overlay.astype(np.int16)
    blended[ball_mask] += noise[ball_mask]
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, 255, -1)
    frame[mask > 0] = blended[mask > 0]
    # Edge highlight
    cv2.ellipse(frame, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, (30, 170, 200), 2)


def compression_profile(t: float, total: float) -> float:
    """Return compression factor 0=rest, 1=max compression."""
    # Timeline: 0-2s rest, 2-4s compress, 4-5s hold, 5-12s recover
    if t < 2.0:
        return 0.0
    if t < 4.0:
        return (t - 2.0) / 2.0 * 0.18  # 18% compression
    if t < 5.0:
        return 0.18
    # Exponential recovery
    tr = t - 5.0
    return 0.18 * math.exp(-tr / 1.2)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (WIDTH, HEIGHT))

    cx, cy = WIDTH // 2, HEIGHT // 2
    total_frames = FPS * DURATION_S

    for i in range(total_frames):
        t = i / FPS
        frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)

        comp = compression_profile(t, DURATION_S)
        rx = BASE_RADIUS_PX * (1 + comp * 0.15)  # bulge
        ry = BASE_RADIUS_PX * (1 - comp)  # compressed axis

        draw_ball(frame, cx, cy, rx, ry, angle=0)

        # Timestamp overlay for debugging
        cv2.putText(frame, f"t={t:.2f}s comp={comp*100:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        writer.write(frame)

    writer.release()
    print(f"Wrote {OUTPUT} ({total_frames} frames @ {FPS} fps)")
    print(f"Reference scale: {PX_PER_MM:.3f} px/mm")


if __name__ == "__main__":
    main()
