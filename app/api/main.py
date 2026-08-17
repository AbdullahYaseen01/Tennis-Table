from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.api.jobs import JobStatus, job_store
from app.config import IS_VERCEL
from app.services.live_frame import (
    analyze_jpeg_bytes,
    compute_baseline_from_frames,
)
from app.services.video_processor import OUTPUTS_DIR, UPLOADS_DIR, process_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Tennis Ball Video Processor",
    description="Upload a bounce/compression test video → get annotated results video.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=1)
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@app.get("/", response_class=HTMLResponse)
def home():
    
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>UI missing</h1>", status_code=500)
    return HTMLResponse(index.read_text(encoding="utf-8"))

@app.get("/health")
def health():
    commit = (
        os.environ.get("RAILWAY_GIT_COMMIT_SHA")
        or os.environ.get("RAILWAY_GIT_COMMIT")
        or os.environ.get("RENDER_GIT_COMMIT")
        or ""
    )
    platform = "render" if os.environ.get("RENDER") or os.environ.get("RENDER_SERVICE_ID") else (
        "railway" if os.environ.get("RAILWAY_ENVIRONMENT") else "local"
    )
    return {
        "ok": True,
        "service": "tennis-ball-tester",
        "commit": commit[:12],
        "platform": platform,
        "nan_guard": True,
    }

@app.get("/api")
def api_info():
    return {
        "service": "Tennis Ball Video Processor",
        "steps": [
            "1. POST /upload  — upload your .mp4 video (multipart form, field name: file)",
            "2. GET  /status/{job_id}  — poll until status is 'completed'",
            "3. GET  /download/{job_id}  — download annotated video",
            "4. GET  /dashboard/{job_id}  — open interactive graphs dashboard",
        ],
        "docs": "/docs",
    }

@app.post("/analyze-frame")
async def analyze_frame(
    file: UploadFile = File(...),
    baseline_h: float | None = Form(None),
    px_per_mm: float | None = Form(None),
    ball_type: str = Form("tennis"),
):
    
    data = await file.read()
    if len(data) < 100:
        raise HTTPException(400, "Frame too small")
    return analyze_jpeg_bytes(
        data, baseline_h_px=baseline_h, px_per_mm=px_per_mm, ball_type=ball_type
    )

@app.post("/analyze-baseline")
async def analyze_baseline(
    files: list[UploadFile] = File(...),
    ball_type: str = Form("tennis"),
):
    
    if not files:
        raise HTTPException(400, "Upload at least one frame")
    if len(files) > 65:
        raise HTTPException(400, "Too many frames (max 65)")

    results = []
    for upload in files:
        data = await upload.read()
        if len(data) < 100:
            continue
        results.append(analyze_jpeg_bytes(data, ball_type=ball_type))

    summary = compute_baseline_from_frames(results, ball_type=ball_type)
    summary["frames_received"] = len(files)
    return summary

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    ball_type: str = Form("tennis"),
    baseline_h_px: float | None = Form(None),
    px_per_mm: float | None = Form(None),
    confidence_pct: float | None = Form(None),
):
    
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format. Allowed: {', '.join(ALLOWED_EXT)}")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    job = job_store.create(Path("_placeholder"), file.filename)

    save_path = UPLOADS_DIR / f"{job.id}{ext}"
    content = await file.read()
    if len(content) < 1000:
        raise HTTPException(400, "File too small — upload a valid video")
    save_path.write_bytes(content)
    job_store.update(
        job.id,
        input_path=save_path,
        process_opts={
            "ball_type": ball_type,
            "baseline_h_px": baseline_h_px,
            "px_per_mm": px_per_mm,
            "confidence_pct": confidence_pct,
        },
    )

    if IS_VERCEL:
        logger.info("Job %s processing inline (Vercel): %s", job.id, file.filename)
        try:
            payload = _process_job_sync(job.id)
            return JSONResponse(status_code=200, content=payload)
        except Exception as exc:
            logger.exception("Job %s failed", job.id)
            job_store.update(job.id, status=JobStatus.FAILED, error=str(exc), progress="Failed")
            raise HTTPException(500, f"Video processing failed: {exc}") from exc

    _executor.submit(_run_job, job.id)
    logger.info("Job %s queued: %s", job.id, file.filename)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.id,
            "status": "queued",
            "message": "Video uploaded. Poll GET /status/{job_id} then download.",
            "status_url": f"/status/{job.id}",
            "download_url": f"/download/{job.id}",
        },
    )

def _process_job_sync(job_id: str) -> dict:
    
    job = job_store.get(job_id)
    if job is None:
        raise ValueError("Job not found")

    job_store.update(job_id, status=JobStatus.PROCESSING, progress="Starting…")

    def on_progress(msg: str) -> None:
        job_store.update(job_id, progress=msg)

    opts = job.process_opts or {}
    out_path, metrics = process_video(
        job.input_path,
        job_id,
        on_progress=on_progress,
        ball_type=opts.get("ball_type", "tennis"),
        fixed_baseline_px=opts.get("baseline_h_px"),
        fixed_px_per_mm=opts.get("px_per_mm"),
    )
    if opts.get("confidence_pct") is not None:
        metrics["confidence_pct"] = opts["confidence_pct"]

    job_store.update(
        job_id,
        status=JobStatus.COMPLETED,
        output_path=out_path,
        results=metrics,
        progress="Done",
    )
    logger.info("Job %s completed: %s", job_id, metrics)
    job = job_store.get(job_id)
    payload = job.to_dict() if job else {"job_id": job_id, "status": "completed", "results": metrics}
    payload["download_url"] = f"/download/{job_id}"
    if metrics.get("dashboard"):
        payload["dashboard_url"] = f"/dashboard/{job_id}"
    return payload

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()

@app.get("/dashboard/{job_id}")
def get_dashboard(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"Not ready — status: {job.status.value}")

    dashboard = (job.results or {}).get("dashboard")
    if not dashboard:
        raise HTTPException(404, "Dashboard not available for this job")
    path = Path(dashboard)
    if not path.exists():
        raise HTTPException(404, "Dashboard file missing on server")

    return FileResponse(path=path, media_type="text/html; charset=utf-8")

@app.get("/download/{job_id}")
def download_video(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED or job.output_path is None:
        raise HTTPException(409, f"Not ready — status: {job.status.value}")
    if not job.output_path.exists():
        raise HTTPException(404, "Output file missing on server")

    stem = Path(job.original_filename).stem.replace(" ", "-")
    filename = f"{stem}-TRACKED{job.output_path.suffix}"
    return FileResponse(
        path=job.output_path,
        media_type="video/mp4",
        filename=filename,
    )

def _run_job(job_id: str) -> None:
    job = job_store.get(job_id)
    if job is None:
        return

    job_store.update(job_id, status=JobStatus.PROCESSING, progress="Starting…")

    def on_progress(msg: str) -> None:
        job_store.update(job_id, progress=msg)

    try:
        opts = job.process_opts or {}
        out_path, metrics = process_video(
            job.input_path,
            job_id,
            on_progress=on_progress,
            ball_type=opts.get("ball_type", "tennis"),
            fixed_baseline_px=opts.get("baseline_h_px"),
            fixed_px_per_mm=opts.get("px_per_mm"),
        )
        if opts.get("confidence_pct") is not None:
            metrics["confidence_pct"] = opts["confidence_pct"]
        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            output_path=out_path,
            results=metrics,
            progress="Done",
        )
        logger.info("Job %s completed: %s", job_id, metrics)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        job_store.update(job_id, status=JobStatus.FAILED, error=str(exc), progress="Failed")
