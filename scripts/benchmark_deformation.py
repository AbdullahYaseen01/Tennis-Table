"""Synthetic ground-truth benchmark for pickleball hand-press deformation.

There is no labelled real dataset in this project, so this harness renders
pickleball-like frames where the true dent angle and depth are known exactly.
It measures the deformation detector on four axes:

  * press detection   -> precision / recall / F1 (+ false-positive rate)
  * compressed area   -> localization IoU (predicted arc vs true dent sector)
  * measured depth    -> mean absolute error vs the same statistic on GT radii
  * performance       -> latency per frame (ms) and FPS

Run:
  py -3.12 scripts/benchmark_deformation.py --tag baseline
  py -3.12 scripts/benchmark_deformation.py --tag improved

Frames are generated deterministically (fixed seed) so two runs are comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vision.deformation import N_ANG, analyze_deformation

PRESS_MIN_PCT = 8.0
BALL_BGR = (55, 235, 225)
SKIN_BGR = (150, 180, 222)
REPORT_DIR = ROOT / "data" / "reports"


@dataclass
class FrameGT:
    pressed: bool
    depth_pct: float
    sector: np.ndarray  # bool[N_ANG], true compressed angular bins
    detectable: bool = True
    meta: dict = field(default_factory=dict)


def _depth_stat(values: np.ndarray) -> float:
    # Same statistic the detector reports (70th-percentile shortfall over the arc),
    # so depth MAE measures real recovery error rather than a definition gap.
    return float(np.percentile(values, 70)) if len(values) else 0.0


def _sector_from_center(center_bin: int, half: int) -> np.ndarray:
    idx = np.arange(N_ANG)
    dang = np.minimum(np.abs(idx - center_bin), N_ANG - np.abs(idx - center_bin))
    return dang <= half


def _true_radii(R: float, center_bin: int, half: int, depth_frac: float, aspect: float) -> np.ndarray:
    """Ground-truth silhouette radius per angular bin with a smooth cosine dent."""
    idx = np.arange(N_ANG)
    ang = 2.0 * np.pi * idx / N_ANG
    dang = np.minimum(np.abs(idx - center_bin), N_ANG - np.abs(idx - center_bin)).astype(np.float64)
    dent = np.zeros(N_ANG, dtype=np.float64)
    inside = dang <= half
    dent[inside] = depth_frac * 0.5 * (1.0 + np.cos(np.pi * dang[inside] / max(half, 1)))
    rad = R * (1.0 - dent)
    # anisotropic scale to fake camera tilt (ellipse)
    rad = rad * (1.0 / np.sqrt((np.cos(ang) ** 2) + (np.sin(ang) / aspect) ** 2))
    return rad


def _render(
    size: tuple[int, int],
    cx: float,
    cy: float,
    radii: np.ndarray,
    bg_val: int,
    ball_scale: float,
    finger: tuple[float, float] | None,
    noise: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = size
    frame = np.full((h, w, 3), bg_val, dtype=np.uint8)
    if noise:
        frame = np.clip(frame.astype(np.int16) + rng.integers(-noise, noise + 1, frame.shape), 0, 255).astype(np.uint8)
    ang = 2.0 * np.pi * np.arange(N_ANG) / N_ANG
    pts = np.column_stack([cx + radii * np.cos(ang), cy + radii * np.sin(ang)]).astype(np.int32)
    ball_col = tuple(int(np.clip(c * ball_scale, 0, 255)) for c in BALL_BGR)
    cv2.fillPoly(frame, [pts], ball_col, cv2.LINE_AA)
    # subtle shading so it is not a flat disc
    cv2.circle(frame, (int(cx), int(cy)), int(np.max(radii) * 0.55), tuple(int(min(255, c * 1.08)) for c in ball_col), -1, cv2.LINE_AA)
    if finger is not None:
        fx, fy = finger
        fr = int(np.max(radii) * 0.34)
        cv2.circle(frame, (int(fx), int(fy)), fr, SKIN_BGR, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(fx), int(fy)), fr, tuple(int(c * 0.85) for c in SKIN_BGR), 3, cv2.LINE_AA)
    return frame


def build_dataset(seed: int = 20240816) -> list[tuple[np.ndarray, FrameGT]]:
    rng = np.random.default_rng(seed)
    data: list[tuple[np.ndarray, FrameGT]] = []
    H, W = 360, 480
    R = 82.0

    # ---- press clips: ramp 0 -> peak -> 0 across lighting / angle / location ----
    press_cfgs = [
        dict(bg=18, peak=0.26, center_deg=270, half=16, aspect=1.00, ball_scale=1.00, noise=6),
        dict(bg=30, peak=0.22, center_deg=90, half=14, aspect=0.92, ball_scale=0.95, noise=8),
        dict(bg=12, peak=0.30, center_deg=180, half=18, aspect=1.00, ball_scale=1.05, noise=5),
        dict(bg=45, peak=0.20, center_deg=0, half=13, aspect=0.88, ball_scale=0.90, noise=10),
        dict(bg=22, peak=0.28, center_deg=315, half=15, aspect=1.06, ball_scale=1.00, noise=7),
        dict(bg=60, peak=0.18, center_deg=45, half=12, aspect=0.95, ball_scale=0.88, noise=12),
    ]
    for ci, cfg in enumerate(press_cfgs):
        center_bin = int(round((cfg["center_deg"] / 360.0) * N_ANG)) % N_ANG
        cx = W * 0.5 + rng.uniform(-25, 25)
        cy = H * 0.5 + rng.uniform(-20, 20)
        n = 22
        for f in range(n):
            phase = np.sin(np.pi * f / (n - 1))  # 0..1..0
            depth = cfg["peak"] * phase
            radii = _true_radii(R, center_bin, cfg["half"], depth, cfg["aspect"])
            true_def = np.maximum(0.0, np.max(radii) - radii)
            # "Actual compressed area" = the dent core (>=35% of local peak),
            # measured around the dent centre so ellipse/tilt shortfall elsewhere
            # is not counted as compression.
            half_sec = _sector_from_center(center_bin, min(cfg["half"], 34))
            peak_true = float(true_def[half_sec].max()) if half_sec.any() else 0.0
            sector = half_sec & (true_def >= 0.35 * peak_true) if peak_true > 0 else np.zeros(N_ANG, bool)
            depth_pct = float(np.clip(_depth_stat(true_def[sector]) / np.max(radii) * 100.0, 0.0, 32.0)) if sector.any() else 0.0
            pressed = depth_pct >= PRESS_MIN_PCT
            finger = None
            if depth > 0.04:
                theta = 2.0 * np.pi * center_bin / N_ANG
                # finger sits just outside the rim at the dent, pressing inward
                fdist = np.max(radii) * (1.0 - depth) + R * 0.28
                finger = (cx + fdist * np.cos(theta), cy + fdist * np.sin(theta))
            frame = _render((H, W), cx, cy, radii, cfg["bg"], cfg["ball_scale"], finger, cfg["noise"], seed + ci * 100 + f)
            data.append((frame, FrameGT(pressed=pressed, depth_pct=depth_pct if pressed else 0.0,
                                        sector=sector if pressed else np.zeros(N_ANG, bool),
                                        meta={"clip": f"press{ci}", "frame": f, "depth_frac": depth})))

    # ---- clean clips: ball only, and ball + nearby (non-pressing) hand ----
    clean_cfgs = [
        dict(bg=18, aspect=1.00, ball_scale=1.00, noise=6, hand=False),
        dict(bg=35, aspect=0.93, ball_scale=0.95, noise=9, hand=False),
        dict(bg=55, aspect=1.05, ball_scale=0.90, noise=12, hand=False),
        dict(bg=20, aspect=1.00, ball_scale=1.00, noise=7, hand=True),   # skin present, NO dent
        dict(bg=40, aspect=0.95, ball_scale=0.95, noise=10, hand=True),  # skin present, NO dent
        dict(bg=14, aspect=1.00, ball_scale=1.03, noise=5, hand=True),   # skin present, NO dent
    ]
    for ci, cfg in enumerate(clean_cfgs):
        cx0 = W * 0.5 + rng.uniform(-25, 25)
        cy0 = H * 0.5 + rng.uniform(-20, 20)
        n = 16
        for f in range(n):
            cx = cx0 + rng.uniform(-3, 3)
            cy = cy0 + rng.uniform(-3, 3)
            radii = _true_radii(R, 0, 1, 0.0, cfg["aspect"])  # round (no dent)
            finger = None
            if cfg["hand"]:
                theta = rng.uniform(0, 2 * np.pi)
                fdist = np.max(radii) + R * 0.55  # hand near but NOT indenting
                finger = (cx + fdist * np.cos(theta), cy + fdist * np.sin(theta))
            frame = _render((H, W), cx, cy, radii, cfg["bg"], cfg["ball_scale"], finger, cfg["noise"], seed + 9000 + ci * 100 + f)
            data.append((frame, FrameGT(pressed=False, depth_pct=0.0, sector=np.zeros(N_ANG, bool),
                                        meta={"clip": f"clean{ci}", "frame": f, "hand": cfg["hand"]})))
    return data


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else (1.0 if not a.any() and not b.any() else 0.0)


def evaluate(dataset: list[tuple[np.ndarray, FrameGT]], warmup: int = 5) -> dict:
    tp = fp = fn = tn = 0
    depth_abs_err: list[float] = []
    depth_bias: list[float] = []
    ious: list[float] = []
    latencies: list[float] = []
    detected = 0
    hand_fp = 0
    hand_total = 0

    # warmup (JIT/caches) — excluded from latency
    for frame, _ in dataset[:warmup]:
        analyze_deformation(frame, "pickleball", frame.shape[1] / 2, frame.shape[0] / 2, 82.0, require_skin=True)

    for frame, gt in dataset:
        cx0, cy0 = frame.shape[1] / 2, frame.shape[0] / 2
        t0 = time.perf_counter()
        res = analyze_deformation(frame, "pickleball", cx0, cy0, 82.0, require_skin=True)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if res.valid:
            detected += 1
        pred_pressed = bool(res.valid and res.deform_pct >= PRESS_MIN_PCT and res.arc_mask is not None and res.arc_mask.any())

        if gt.meta.get("hand"):
            hand_total += 1
            if pred_pressed:
                hand_fp += 1

        if gt.pressed and pred_pressed:
            tp += 1
            depth_abs_err.append(abs(res.deform_pct - gt.depth_pct))
            depth_bias.append(res.deform_pct - gt.depth_pct)
            ious.append(_iou(res.arc_mask.astype(bool), gt.sector))
        elif gt.pressed and not pred_pressed:
            fn += 1
        elif (not gt.pressed) and pred_pressed:
            fp += 1
        else:
            tn += 1

    total = len(dataset)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    lat = np.array(latencies)
    return {
        "frames": total,
        "positives_gt": tp + fn,
        "negatives_gt": tn + fp,
        "detection_rate_pct": round(detected / total * 100, 1),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "accuracy": round((tp + tn) / total, 3),
        "false_positive_rate": round(fp / (fp + tn), 3) if (fp + tn) else 0.0,
        "hand_present_false_positive_rate": round(hand_fp / hand_total, 3) if hand_total else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "localization_iou_mean": round(float(np.mean(ious)), 3) if ious else 0.0,
        "depth_mae_pct": round(float(np.mean(depth_abs_err)), 2) if depth_abs_err else None,
        "depth_bias_pct": round(float(np.mean(depth_bias)), 2) if depth_bias else None,
        "latency_ms_mean": round(float(lat.mean()), 2),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
        "latency_ms_p90": round(float(np.percentile(lat, 90)), 2),
        "fps_mean": round(1000.0 / float(lat.mean()), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=20240816)
    args = ap.parse_args()

    dataset = build_dataset(args.seed)
    metrics = evaluate(dataset)
    metrics["tag"] = args.tag

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"deformation-benchmark-{args.tag}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 56)
    print(f"DEFORMATION BENCHMARK [{args.tag}]  ({metrics['frames']} frames)")
    print("=" * 56)
    print(f"  detection rate      {metrics['detection_rate_pct']}%")
    print(f"  precision / recall  {metrics['precision']} / {metrics['recall']}   F1 {metrics['f1']}")
    print(f"  accuracy            {metrics['accuracy']}")
    print(f"  false-positive rate {metrics['false_positive_rate']}  (hand-present {metrics['hand_present_false_positive_rate']})")
    print(f"  localization IoU    {metrics['localization_iou_mean']}")
    print(f"  depth MAE / bias    {metrics['depth_mae_pct']} / {metrics['depth_bias_pct']} pct-points")
    print(f"  latency mean/p90    {metrics['latency_ms_mean']} / {metrics['latency_ms_p90']} ms   ({metrics['fps_mean']} FPS)")
    print(f"  confusion TP/FP/FN/TN  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")
    print(f"\n  saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
