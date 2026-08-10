from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.vision.ball_profiles import PROFILES, get_profile, reference_diameter_mm

IS_VERCEL = bool(os.environ.get("VERCEL"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CALIB_DIR = DATA_DIR / "calib"
SAMPLES_DIR = DATA_DIR / "samples"
DB_PATH = DATA_DIR / "runs.db"
INTRINSICS_PATH = CALIB_DIR / "intrinsics.npz"
SCALE_PATH = CALIB_DIR / "scale.npz"
VALIDATION_PATH = CALIB_DIR / "validation.npz"

DEFAULT_SOURCE = str(SAMPLES_DIR / "sample_test.mp4")

CHECKERBOARD_COLS = 9
CHECKERBOARD_ROWS = 6
CHECKERBOARD_SQUARE_MM = 25.0

COLOR_PROFILES: dict[str, dict] = {
    k: {
        "hsv_lower": np.array(p.hsv_lower),
        "hsv_upper": np.array(p.hsv_upper),
    }
    for k, p in PROFILES.items()
}

REFERENCE_DIAMETERS_MM = {k: p.diameter_mm for k, p in PROFILES.items()}

SUBPIXEL_RAYS = 360
ROI_PADDING_PX = 50
MORPH_KERNEL_SIZE = 5
MIN_CONTOUR_AREA_PX = 150
EDGE_INLIER_THRESHOLD_PX = 2.0
USE_CLAHE = True

YOLO_ENABLED = True
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.25

TEMPORAL_FUSION_WINDOW = 9
REST_ECCENTRICITY_MAX = 0.15
AUTO_SCALE_ON_BASELINE = True

BASELINE_FRAME_COUNT = 45
COMPRESSION_MEDIAN_WINDOW = 5
BASELINE_OUTLIER_SIGMA = 2.5
LOAD_AXIS = "minor"

COMPRESSION_THRESHOLD_PCT = 2.0
RELEASE_THRESHOLD_PCT = 0.5
MIN_COMPRESSION_PEAK_PCT = 5.0

NUM_SURFACE_ZONES = 8
ZONE_ZSCORE_THRESHOLD = 2.0
SURFACE_ANNULUS_INNER_RATIO = 0.85
SURFACE_ANNULUS_OUTER_RATIO = 1.15

DEFAULT_EXPOSURE = -6
DEFAULT_WB_TEMPERATURE = 4500
