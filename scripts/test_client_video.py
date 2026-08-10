from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import app.config as _cfg

_cfg.YOLO_ENABLED = False
_cfg.SUBPIXEL_RAYS = 72
_cfg.SUBPIXEL_RAYS_REST = 120

import numpy as np

from app.config import REFERENCE_DIAMETERS_MM, SAMPLES_DIR
from app.core.models import TestState
from app.core.pipeline import TestPipeline
from app.vision.calibration import Calibrator
from app.vision.camera import FrameSource

def _resize_frame(frame: np.ndarray, max_side: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def run_full_test(video_path: Path, ball_type: str = "tennis") -> dict:
    known_mm = REFERENCE_DIAMETERS_MM.get(ball_type, 67.0)
    cal = Calibrator()
    pipe = TestPipeline(cal)
    pipe.set_ball_type(ball_type)
    pipe.set_ball_id("client-test")

    src = FrameSource(str(video_path), loop_file=False)

    all_minors: list[float] = []
    all_confs: list[float] = []
    total_frames = 0
    detected_frames = 0
    compressions: list[float] = []
    baseline_locked_at: float | None = None
    test_started = False

    pipe.start_baseline_capture()

    while True:
        ok, frame, ts = src.read()
        if not ok or frame is None:
            break
        total_frames += 1
        frame = _resize_frame(frame)
        measurement, comp = pipe.process_frame(frame, ts)

        if measurement:
            detected_frames += 1
            all_minors.append(measurement.minor_mm)
            all_confs.append(measurement.detection_confidence)

        if pipe.compression.baseline_mm and baseline_locked_at is None:
            baseline_locked_at = ts

        if (
            pipe.state == TestState.IDLE
            and pipe.compression.baseline_mm
            and not test_started
        ):
            pipe.start_test()
            test_started = True

        if comp and pipe.state not in (TestState.BASELINE, TestState.IDLE):
            compressions.append(comp.compression_pct)

        if pipe.state == TestState.DONE:
            break

        if total_frames % 30 == 0:
            print(f"  â€¦ frame {total_frames}/{281}  state={pipe.state.value}", flush=True)

    src.release()

    recovery = pipe.recovery_result
    rest_n = min(48, len(all_minors))
    rest_minors = np.array(all_minors[:rest_n]) if rest_n else np.array([])

    return {
        "video": video_path.name,
        "tested_at": datetime.now().isoformat(timespec="seconds"),
        "ball_type": ball_type,
        "known_diameter_mm": known_mm,
        "video_frames": total_frames,
        "detection_rate_pct": round(detected_frames / max(total_frames, 1) * 100, 1),
        "mean_confidence": round(float(np.mean(all_confs)), 3) if all_confs else None,
        "scale_px_per_mm": round(cal.pixels_per_mm, 3) if cal.pixels_per_mm else None,
        "rest_mean_minor_mm": round(float(np.mean(rest_minors)), 2) if len(rest_minors) else None,
        "rest_std_mm": round(float(np.std(rest_minors)), 3) if len(rest_minors) > 1 else None,
        "rest_error_pct": round((float(np.mean(rest_minors)) - known_mm) / known_mm * 100, 2)
        if len(rest_minors)
        else None,
        "baseline_mm": round(pipe.compression.baseline_mm, 2) if pipe.compression.baseline_mm else None,
        "baseline_locked_at_s": round(baseline_locked_at, 2) if baseline_locked_at else None,
        "max_compression_pct": round(max(compressions), 1) if compressions else None,
        "recovery_tau_s": round(recovery.tau_s, 3) if recovery else None,
        "recovery_t95_s": round(recovery.t95_s, 3) if recovery else None,
        "recovery_residual_pct": round(recovery.residual_pct, 2) if recovery else None,
        "recovery_fit_confidence": round(recovery.fit_confidence, 2) if recovery else None,
        "test_completed": pipe.state == TestState.DONE,
        "pipeline_state": pipe.state.value,
    }

def save_report(report: dict, stem: str) -> tuple[Path, Path]:
    out_dir = ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}-test-report.json"
    txt_path = out_dir / f"{stem}-test-report.txt"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "TENNIS BALL CONDITION TEST REPORT",
        "=" * 40,
        f"Video:              {report['video']}",
        f"Test date:          {report['tested_at']}",
        f"Ball type:          {report['ball_type']}",
        "",
        "DETECTION",
        f"  Frames analyzed:  {report['video_frames']}",
        f"  Detection rate:   {report['detection_rate_pct']}%",
        f"  Mean confidence:  {report['mean_confidence']}",
        "",
        "CALIBRATION (auto from ball at rest)",
        f"  Scale:            {report['scale_px_per_mm']} px/mm",
        f"  Rest diameter:    {report['rest_mean_minor_mm']} mm  (ITF spec {report['known_diameter_mm']} mm)",
        f"  Rest error:       {report['rest_error_pct']:+.2f}%" if report.get("rest_error_pct") is not None else "",
        f"  Jitter (std):     {report['rest_std_mm']} mm" if report.get("rest_std_mm") else "",
        "",
        "COMPRESSION TEST",
        f"  Baseline:         {report['baseline_mm']} mm",
        f"  Max compression:  {report['max_compression_pct']}%",
        "",
        "RECOVERY",
    ]
    if report.get("recovery_tau_s"):
        lines += [
            f"  Tau (time const): {report['recovery_tau_s']} s",
            f"  t95 (95% recv):   {report['recovery_t95_s']} s",
            f"  Residual:         {report['recovery_residual_pct']}%",
            f"  Fit confidence:   {report['recovery_fit_confidence']}",
        ]
    else:
        lines.append(f"  Not completed (state: {report['pipeline_state']})")

    lines += [
        "",
        f"Test completed:     {'Yes' if report['test_completed'] else 'No'}",
        "",
        "Notes:",
        "  - Scale calibrated automatically from known tennis ball diameter (67 mm).",
        "  - Compression % is relative to the ball's own baseline (most reliable metric).",
        "  - Recovery tau = exponential time constant after load release.",
    ]
    txt = "\n".join(l for l in lines if l is not None)
    txt_path.write_text(txt, encoding="utf-8")
    return txt_path, json_path

def main() -> int:
    video = SAMPLES_DIR / "Test video.mp4"
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])

    if not video.exists():
        print(f"Video not found: {video}")
        return 1

    stem = video.stem.replace(" ", "-").lower()
    print(f"Testing client video: {video}\n", flush=True)
    report = run_full_test(video)
    txt_path, json_path = save_report(report, stem)

    print("\n" + open(txt_path, encoding="utf-8").read())
    print(f"\nReports saved to:\n  {txt_path}\n  {json_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
