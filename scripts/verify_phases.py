"""Verify acceptance criteria for all phases."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from app.config import SAMPLES_DIR
from app.core.db import RunDatabase
from app.core.models import TestState
from app.core.pipeline import TestPipeline
from app.vision.calibration import Calibrator
from app.vision.camera import FrameSource
from app.vision.compression import CompressionTracker
from app.vision.detect import BallDetector
from app.vision.recovery import RecoveryAnalyzer
from app.vision.surface import SurfaceAnalyzer


def test_phase1() -> bool:
    print("Phase 1: FrameSource...")
    video = SAMPLES_DIR / "sample_test.mp4"
    assert video.exists(), f"Missing {video}"
    src = FrameSource(str(video))
    ok, frame, ts = src.read()
    src.release()
    assert ok and frame is not None
    assert ts > 0
    print("  PASS: file source returns (ok, frame, timestamp)")
    return True


def test_phase2() -> bool:
    print("Phase 2: Calibration...")
    cal = Calibrator()
    assert cal.is_ready(), "Dev calibration should exist"
    val = cal.get_validation()
    assert val is not None
    print(f"  Validation error: {val.error_pct:.4f}%")
    print("  PASS: calibration loaded with validation")
    return True


def test_phase3() -> bool:
    print("Phase 3: Ball detection...")
    cal = Calibrator()
    det = BallDetector(cal)
    src = FrameSource(str(SAMPLES_DIR / "sample_test.mp4"))
    diameters = []
    for _ in range(60):
        ok, frame, ts = src.read()
        if not ok:
            break
        m = det.detect(frame, ts)
        if m:
            diameters.append(m.minor_mm)
    src.release()
    assert len(diameters) > 30, "Should detect ball in most frames"
    std = float(np.std(diameters[:30]))
    print(f"  Resting diameter std: {std:.4f} mm (n={len(diameters[:30])})")
    assert std < 2.0, f"Diameter jitter too high: {std}"
    print("  PASS: ellipse tracking stable")
    return True


def test_phase4_5() -> bool:
    print("Phase 4-5: Compression & recovery...")
    cal = Calibrator()
    pipe = TestPipeline(cal)
    src = FrameSource(str(SAMPLES_DIR / "sample_test.mp4"))

    pipe.start_baseline_capture()
    compressions = []
    for _ in range(300):
        ok, frame, ts = src.read()
        if not ok:
            break
        _, comp = pipe.process_frame(frame, ts)
        if pipe.compression.baseline_mm and pipe.state == TestState.IDLE:
            break

    assert pipe.compression.baseline_mm is not None
    pipe.start_test()

    for _ in range(300):
        ok, frame, ts = src.read()
        if not ok:
            break
        _, comp = pipe.process_frame(frame, ts)
        if comp:
            compressions.append(comp.compression_pct)
        if pipe.state == TestState.DONE:
            break

    src.release()
    assert max(compressions) > 5, f"Expected compression peak, got max={max(compressions):.1f}%"
    assert pipe.recovery_result is not None
    r = pipe.recovery_result
    print(f"  Max compression: {max(compressions):.1f}%")
    print(f"  Recovery tau={r.tau_s:.3f}s t95={r.t95_s:.3f}s residual={r.residual_pct:.2f}%")
    print("  PASS: compression and recovery")
    return True


def test_phase6() -> bool:
    print("Phase 6: Surface analysis...")
    cal = Calibrator()
    det = BallDetector(cal)
    surf = SurfaceAnalyzer(cal)
    src = FrameSource(str(SAMPLES_DIR / "sample_test.mp4"))
    ok, frame, ts = src.read()
    src.release()
    m = det.detect(frame, ts)
    assert m is not None
    worn = surf.artificially_wear_zone(frame, m, zone_index=2)
    m2 = det.detect(worn, ts)
    assert m2 is not None
    result = surf.analyze_single_view(worn, m2)
    flagged = [z.zone_index for z in result.zones if z.flagged]
    print(f"  Flagged zones after artificial wear: {flagged}")
    assert len(flagged) > 0, "Should flag worn zone"
    print("  PASS: surface zone outlier detection")
    return True


def test_phase7() -> bool:
    print("Phase 7: Database...")
    db = RunDatabase()
    from app.core.models import CompressionSample, TestRunSummary, ZoneResult

    summary = TestRunSummary(
        ball_type="tennis",
        ball_id="test-ball",
        baseline_mm=67.0,
        max_compression_pct=15.0,
        recovery_tau_s=1.0,
        recovery_t95_s=2.5,
        residual_pct=0.5,
        accuracy_error_pct=0.1,
        zone_scores=[ZoneResult(i, 0.5, i == 2) for i in range(8)],
        timeseries=[
            CompressionSample(0.0, 0.0, 0.0, 67.0, "BASELINE"),
            CompressionSample(1.0, 10.0, 1.0, 60.0, "COMPRESSING"),
        ],
    )
    run_id = db.save_run(summary)
    run = db.get_run(run_id)
    assert run is not None
    assert len(run["zone_scores"]) == 8
    assert len(run["timeseries"]) == 2
    print(f"  Saved and loaded run #{run_id}")
    print("  PASS: database storage")
    return True


def main() -> int:
    tests = [test_phase1, test_phase2, test_phase3, test_phase4_5, test_phase6, test_phase7]
    passed = 0
    for t in tests:
        try:
            if t():
                passed += 1
        except Exception as exc:
            print(f"  FAIL: {exc}")
    print(f"\n{passed}/{len(tests)} phase tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
