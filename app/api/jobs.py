"""In-memory job store for video processing tasks."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


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
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)


job_store = JobStore()
