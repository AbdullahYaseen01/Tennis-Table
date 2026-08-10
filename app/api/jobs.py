from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from app.config import IS_VERCEL

_JOBS_ROOT = Path("/tmp/data/jobs") if IS_VERCEL else None

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: str
    input_path: Path
    output_path: Path | None = None
    original_filename: str = ""
    progress: str = ""
    error: str | None = None
    results: dict[str, Any] = field(default_factory=dict)
    process_opts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "created_at": self.created_at,
            "original_filename": self.original_filename,
            "progress": self.progress,
            "error": self.error,
            "results": self.results,
            "download_ready": self.status == JobStatus.COMPLETED and self.output_path is not None,
        }

class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        if _JOBS_ROOT:
            _JOBS_ROOT.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path | None:
        return (_JOBS_ROOT / f"{job_id}.json") if _JOBS_ROOT else None

    def _serialize(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id,
            "status": job.status.value,
            "created_at": job.created_at,
            "input_path": str(job.input_path),
            "output_path": str(job.output_path) if job.output_path else None,
            "original_filename": job.original_filename,
            "progress": job.progress,
            "error": job.error,
            "results": job.results,
            "process_opts": job.process_opts,
        }

    def _deserialize(self, data: dict[str, Any]) -> Job:
        return Job(
            id=data["id"],
            status=JobStatus(data["status"]),
            created_at=data["created_at"],
            input_path=Path(data["input_path"]),
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            original_filename=data.get("original_filename", ""),
            progress=data.get("progress", ""),
            error=data.get("error"),
            results=data.get("results") or {},
            process_opts=data.get("process_opts") or {},
        )

    def _persist(self, job: Job) -> None:
        path = self._job_path(job.id)
        if path is None:
            return
        path.write_text(json.dumps(self._serialize(job)), encoding="utf-8")

    def _load_disk(self, job_id: str) -> Job | None:
        path = self._job_path(job_id)
        if path is None or not path.exists():
            return None
        try:
            return self._deserialize(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def create(self, input_path: Path, original_filename: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(timezone.utc).isoformat(),
            input_path=input_path,
            original_filename=original_filename,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
            job = self._load_disk(job_id)
            if job is not None:
                self._jobs[job_id] = job
            return job

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id) or self._load_disk(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)
            self._jobs[job_id] = job
            self._persist(job)

job_store = JobStore()
