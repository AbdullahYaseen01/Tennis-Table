"""Application constants, tunables, and color profiles."""
from __future__ import annotations

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CALIB_DIR = DATA_DIR / "calib"
SAMPLES_DIR = DATA_DIR / "samples"
DB_PATH = DATA_DIR / "runs.db"
INTRINSICS_PATH = CALIB_DIR / "intrinsics.npz"
SCALE_PATH = CALIB_DIR / "scale.npz"
VALIDATION_PATH = CALIB_DIR / "validation.npz"

# Default video source — change this single value to switch input type.
# Examples: 0 (webcam), "rtsp://...", str(SAMPLES_DIR / "sample_test.mp4")
DEFAULT_SOURCE = str(SAMPLES_DIR / "sample_test.mp4")

# Checkerboard calibration
CHECKERBOARD_COLS = 9
CHECKERBOARD_ROWS = 6
CHECKERBOARD_SQUARE_MM = 25.0

# Ball colour profiles (HSV bounds: H, S, V min/max)
COLOR_PROFILES: dict[str, dict[str, tuple[int, int, int, int, int, int]]] = {
    "tennis": {
        "hsv_lower": (25, 40, 80),
        "hsv_upper": (75, 255, 255),
    },
    "pickleball": {
        "hsv_lower": (20, 60, 120),
        "hsv_upper": (45, 255, 255),
    },
}

# Detection — accuracy tunables
SUBPIXEL_RAYS = 360
ROI_PADDING_PX = 50
MORPH_KERNEL_SIZE = 5
MIN_CONTOUR_AREA_PX = 150
EDGE_INLIER_THRESHOLD_PX = 2.0
USE_CLAHE = True

# Optional YOLO coarse ROI (ultralytics) — precise edges still classical CV
YOLO_ENABLED = True
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.25

# Known reference diameters (mm) for scale-from-ball calibration
REFERENCE_DIAMETERS_MM = {
    "tennis": 67.0,
    "pickleball": 74.0,
}

# Accuracy — temporal fusion & auto-scale
TEMPORAL_FUSION_WINDOW = 9
REST_ECCENTRICITY_MAX = 0.15  # apply fusion when ball is nearly round
AUTO_SCALE_ON_BASELINE = True  # calibrate px/mm from baseline + known ball spec

# Compression
BASELINE_FRAME_COUNT = 45
COMPRESSION_MEDIAN_WINDOW = 5
BASELINE_OUTLIER_SIGMA = 2.5
LOAD_AXIS = "minor"  # axis aligned with applied load: "minor" or "major"

# Recovery state machine thresholds
COMPRESSION_THRESHOLD_PCT = 2.0
RELEASE_THRESHOLD_PCT = 0.5
MIN_COMPRESSION_PEAK_PCT = 5.0

# Surface analysis
NUM_SURFACE_ZONES = 8
ZONE_ZSCORE_THRESHOLD = 2.0
SURFACE_ANNULUS_INNER_RATIO = 0.85
SURFACE_ANNULUS_OUTER_RATIO = 1.15

# Camera defaults
DEFAULT_EXPOSURE = -6
DEFAULT_WB_TEMPERATURE = 4500
