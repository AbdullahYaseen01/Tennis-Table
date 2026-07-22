"""Optional YOLO sports-ball detector for coarse ROI seeding only."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.config import YOLO_CONFIDENCE, YOLO_ENABLED, YOLO_MODEL

logger = logging.getLogger(__name__)

# COCO class index for "sports ball"
SPORTS_BALL_CLASS = 32


@dataclass
class RoiHint:
    center: tuple[float, float]
    radius: float
    confidence: float


class YoloRoiSeeder:
    """
    Optional ultralytics YOLO — locates sports ball for ROI only.
    Precise measurement remains classical CV (sub-pixel ellipse).
    """

    def __init__(self) -> None:
        self._model = None
        self._available = False
        if YOLO_ENABLED:
            self._try_load()

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(YOLO_MODEL)
            self._available = True
            logger.info("YOLO ROI seeder loaded: %s", YOLO_MODEL)
        except Exception as exc:
            logger.warning("YOLO unavailable (optional): %s", exc)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def detect_roi(self, frame_bgr: np.ndarray) -> RoiHint | None:
        if not self._available or self._model is None:
            return None

        try:
            results = self._model.predict(
                frame_bgr,
                conf=YOLO_CONFIDENCE,
                verbose=False,
                classes=[SPORTS_BALL_CLASS],
            )
        except Exception as exc:
            logger.debug("YOLO predict failed: %s", exc)
            return None

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        # Prefer sports ball class when present
        cls = boxes.cls.cpu().numpy().astype(int)
        sports = np.where(cls == SPORTS_BALL_CLASS)[0]
        if len(sports) > 0:
            best_idx = int(sports[np.argmax(confs[sports])])
        else:
            best_idx = int(np.argmax(confs))
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        conf = float(confs[best_idx])
        x0, y0, x1, y1 = xyxy
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        radius = max(x1 - x0, y1 - y0) / 2
        return RoiHint(center=(cx, cy), radius=radius, confidence=conf)

    def roi_box(self, hint: RoiHint, frame_shape: tuple[int, ...], padding: float = 1.35) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        cx, cy = hint.center
        r = hint.radius * padding
        x0 = max(0, int(cx - r))
        y0 = max(0, int(cy - r))
        x1 = min(w, int(cx + r))
        y1 = min(h, int(cy + r))
        return x0, y0, x1 - x0, y1 - y0
