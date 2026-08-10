from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_test.mp4"

WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION_S = 12
BASE_RADIUS_PX = 120  
PX_PER_MM = (BASE_RADIUS_PX * 2) / 67.0  

def draw_ball(frame: np.ndarray, cx: float, cy: float, rx: float, ry: float, angle: float = 0) -> None:
    
    overlay = frame.copy()
    cv2.ellipse(overlay, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, (40, 200, 220), -1)
    
    noise = np.random.randint(-15, 15, overlay.shape, dtype=np.int16)
    ball_mask = (overlay > 0).all(axis=2)
    blended = overlay.astype(np.int16)
    blended[ball_mask] += noise[ball_mask]
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, 255, -1)
    frame[mask > 0] = blended[mask > 0]
    
    cv2.ellipse(frame, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, (30, 170, 200), 2)

def compression_profile(t: float, total: float) -> float:
    
    
    if t < 2.0:
        return 0.0
    if t < 4.0:
        return (t - 2.0) / 2.0 * 0.18  
    if t < 5.0:
        return 0.18
    
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
        rx = BASE_RADIUS_PX * (1 + comp * 0.15)  
        ry = BASE_RADIUS_PX * (1 - comp)  

        draw_ball(frame, cx, cy, rx, ry, angle=0)

        
        cv2.putText(frame, f"t={t:.2f}s comp={comp*100:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        writer.write(frame)

    writer.release()
    print(f"Wrote {OUTPUT} ({total_frames} frames @ {FPS} fps)")
    print(f"Reference scale: {PX_PER_MM:.3f} px/mm")

if __name__ == "__main__":
    main()
