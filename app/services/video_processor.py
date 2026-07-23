"""Process uploaded videos into annotated client-ready MP4s."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import IS_VERCEL
from scripts.export_bounce_video import export as export_bounce

logger = logging.getLogger(__name__)

_DATA_ROOT = Path("/tmp") if IS_VERCEL else ROOT
UPLOADS_DIR = _DATA_ROOT / "data" / "uploads"
OUTPUTS_DIR = _DATA_ROOT / "data" / "outputs"


def _to_mp4(avi_path: Path, mp4_path: Path) -> Path:
    """Convert AVI to H.264 MP4 using ffmpeg if available."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-i", str(avi_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(mp4_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        avi_path.unlink(missing_ok=True)
        return mp4_path

    logger.warning("ffmpeg not found — output will be AVI")
    dest = mp4_path.with_suffix(".avi")
    if avi_path != dest:
        shutil.move(str(avi_path), str(dest))
    return dest


def process_video(
    input_path: Path,
    job_id: str,
    on_progress=None,
    *,
    ball_type: str = "tennis",
    fixed_baseline_px: float | None = None,
    fixed_px_per_mm: float | None = None,
) -> tuple[Path, dict]:
    """Run bounce/compression tracker and return (output_mp4_path, results_dict)."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if on_progress:
        on_progress("Analyzing video…")

    out_path = OUTPUTS_DIR / f"{job_id}-tracked.mp4"

    if on_progress:
        on_progress("Tracking ball and measuring compression…")

    metrics = export_bounce(
        input_path,
        out_path,
        ball_type=ball_type,
        fixed_baseline_px=fixed_baseline_px,
        fixed_px_per_mm=fixed_px_per_mm,
        fast_mode=IS_VERCEL,
        use_yolo=not IS_VERCEL,
    )

    if on_progress:
        on_progress("Finalizing video…")

    # OpenCV mp4v usually works on Vercel; convert if we got AVI fallback
    if out_path.suffix.lower() == ".avi" or not out_path.exists():
        avi = out_path.with_suffix(".avi")
        if avi.exists():
            final_path = _to_mp4(avi, OUTPUTS_DIR / f"{job_id}-tracked.mp4")
        else:
            final_path = out_path
    else:
        final_path = out_path

    return final_path, metrics
