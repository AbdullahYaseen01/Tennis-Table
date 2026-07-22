# Tennis Ball & Pickleball Condition Testing System

A desktop application (Python + OpenCV + PySide6) that measures tennis ball and pickleball physical condition from camera or recorded video: **compression under load**, **recovery after release**, and **per-zone surface wear**.

**Core principle:** accuracy is quantified, not assumed. The system measures its own error against a known reference and displays that figure with every result.

## Setup

```bash
cd tennis-ball-tester
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Generate the sample video and dev calibration (no camera required):

```bash
python scripts/generate_sample_video.py
python scripts/generate_dev_calibration.py
```

Run the app:

```bash
python app/main.py
```

Verify all phases:

```bash
python scripts/verify_phases.py
```

## Accuracy Improvements (v2)

This build adds several layers for higher measurement precision:

| Layer | What it does |
|-------|----------------|
| **360-ray sub-pixel edges** | Sobel gradient + mask boundary + parabolic refinement |
| **Robust ellipse fit** | Iterative outlier rejection + RANSAC fallback |
| **CLAHE preprocessing** | Stable edges under uneven lighting |
| **YOLOv8 ROI (optional)** | Pre-trained sports-ball detector seeds search region only |
| **Ball-based scale** | Calibrate px/mm from a known 67 mm / 74 mm ball at rest |
| **Robust baseline** | MAD outlier rejection on 45 baseline frames |
| **Recovery smoothing** | Light filter before exponential curve fit |

### Recommended calibration order (best accuracy)

1. **Intrinsics** — checkerboard (removes lens distortion)
2. **Scale from ball** — Calibration tab → capture ~20 rest frames → **Calibrate Scale from Ball**
3. **Validate** — place ball in frame, enter known diameter, click Validate

> For uploaded videos, always recalibrate scale on **your** video. The sample video scale does not transfer to other cameras or resolutions.

### Benchmark

```bash
python scripts/benchmark_accuracy.py
python scripts/benchmark_accuracy.py path/to/your/video.mp4
```

Reports jitter (std), mean error vs known diameter, and detection confidence.


Change a single config value in `app/config.py`:

```python
DEFAULT_SOURCE = str(SAMPLES_DIR / "sample_test.mp4")  # file
# DEFAULT_SOURCE = 0                                    # webcam index
# DEFAULT_SOURCE = "rtsp://..."                         # network camera
```

The Live and Calibration tabs also let you enter a source at runtime.

## Calibration Walkthrough

Complete both calibrations before running tests. Open the **Calibration** tab.

### 2a. Lens intrinsics (removes barrel distortion)

1. Print a 9×6 checkerboard with 25 mm squares (see `app/config.py`).
2. Capture ~15 frames at varied angles while the checkerboard fills much of the frame.
3. Click **Compute Intrinsics**. Saved to `data/calib/intrinsics.npz`.
4. Undistortion is applied to every frame before measurement.

### 2b. Real-world scale (pixels → mm)

1. Place the checkerboard at the **exact plane** where the ball will sit.
2. Click **Calibrate Scale from Current Frame**. Saved to `data/calib/scale.npz`.

> Parallax error: if the reference is closer or farther than the ball, absolute mm readings will be wrong. Relative compression % is still valid.

### 2c. Accuracy validation

1. Place an object of known diameter (coin, gauge, new ball spec) in frame.
2. Enter the known diameter and click **Validate Accuracy**.
3. The app reports absolute error in mm and %. This value is stored with each test run.

Target error on a decent setup: **< 1–2%**. Whatever your setup achieves is shown honestly.

## Running a Test

1. Open the **Live Test** tab and click **Open** (default: sample video).
2. Select ball type (`tennis` or `pickleball`) and enter a **Ball ID** to track fatigue over time.
3. **Capture Baseline** — hold the ball at rest for ~1 second (30 frames).
4. **Start Test** — apply load (or let the sample video play through compression).
5. Release the ball; recovery is detected automatically and fitted.
6. **Run Surface Scan** — analyses the front-facing 8 zones (single-view mode).
7. **Save Run** — writes to SQLite (`data/runs.db`).

View results in the **Dashboard** tab: per-run charts, zone scores, and fatigue trends by Ball ID.

## Accuracy Engineering Checklist

| # | Requirement | Phase |
|---|-------------|-------|
| 1 | Lens undistortion on every frame | 2a |
| 2 | Scale reference at the ball plane | 2b |
| 3 | Sub-pixel radial edge refinement | 3 |
| 4 | Ellipse fitting (not Hough) for measurement | 3 |
| 5 | Manual exposure & white balance locked on live cameras | 1 |
| 6 | Median temporal filter (window 3–5) | 4 |
| 7 | Quantified validation stored per run | 2c |
| 8 | Even, diffuse lighting — avoid specular highlights | — |

## Measurement Notes

### Relative vs absolute

- **Compression %** is measured against the ball's own baseline → robust to calibration error. This is the most reliable output.
- **Absolute mm** (diameter, bulge) depends on calibration quality; the reported validation error gives context.

### Surface analysis modes

- **Mode A (single view):** 8 angular sectors of the visible hemisphere. Fast, but front-facing only.
- **Mode B (rotate & capture):** rotate the ball to 8 marked positions for full coverage (recommended for production).

### Pickleball

Pickleball fatigue appears as cracking and out-of-round behaviour, not fuzz wear. The surface module detects edge discontinuities for pickleball profiles.

## Limitations

- A single 2D camera measures the **silhouette edge** and **front-facing surface** only.
- Full-surface fuzz analysis requires rotate-and-capture mode (Mode B).
- Absolute accuracy depends on calibration quality, which the app reports — it does not assume perfection.
- Glossy pickleballs may need diffuse lighting or a polarizer to reduce specular glare.

## Project Structure

```
tennis-ball-tester/
  app/
    main.py              # Entry point
    config.py            # Constants and colour profiles
    ui/                  # PySide6 tabs
    vision/              # Camera, calibration, detection, analysis
    core/                # Pipeline, models, database
  data/
    calib/               # Calibration files (.npz)
    samples/             # sample_test.mp4 for dev
    runs.db              # SQLite (created on first save)
  scripts/               # Sample video & verification
```

## Optional: YOLO Detector

Classical CV (colour segmentation + sub-pixel ellipse) is the primary path. To experiment with YOLO, uncomment `ultralytics` in `requirements.txt` and wire it as an alternate detector — not required for normal operation.
