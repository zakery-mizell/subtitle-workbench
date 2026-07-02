from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str | None = None
    result: Any | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    artifacts: list[str] = field(default_factory=list)


class ProgressReporter:
    """Reports staged progress for a job.

    A stage covers the range [weight_done, next stage's weight_done); `tick`
    interpolates within the current stage.
    """

    def __init__(self, job: Job, lock: threading.Lock) -> None:
        self._job = job
        self._lock = lock
        self._stage_start = 0.0
        self._stage_end = 1.0

    def stage(self, name: str, weight_done: float, weight_end: float = 1.0, message: str | None = None) -> None:
        with self._lock:
            self._stage_start = max(0.0, min(1.0, weight_done))
            self._stage_end = max(self._stage_start, min(1.0, weight_end))
            self._job.stage = name
            self._job.progress = self._stage_start
            self._job.message = message

    def tick(self, fraction_of_stage: float, message: str | None = None) -> None:
        fraction = max(0.0, min(1.0, fraction_of_stage))
        with self._lock:
            self._job.progress = self._stage_start + (self._stage_end - self._stage_start) * fraction
            if message is not None:
                self._job.message = message


class JobRegistry:
    """Minimal in-process job store with a single worker thread.

    Jobs run one at a time so GPU/RAM-heavy mastering never competes with
    itself. State survives only for the process lifetime, which matches the
    single-user local deployment.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mastering")

    def submit(self, fn: Callable[[Job, ProgressReporter], Any]) -> str:
        job = Job(id=f"j_{uuid.uuid4().hex[:12]}")
        with self._lock:
            self._jobs[job.id] = job
        reporter = ProgressReporter(job, self._lock)

        def run() -> None:
            with self._lock:
                job.status = "running"
            try:
                result = fn(job, reporter)
            except Exception as exc:  # noqa: BLE001 - job errors must surface via status
                traceback.print_exc()
                with self._lock:
                    job.status = "error"
                    job.error = str(exc) or exc.__class__.__name__
                return
            with self._lock:
                job.status = "done"
                job.progress = 1.0
                job.result = result

        self._executor.submit(run)
        return job.id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def purge_expired(self, ttl_seconds: float, delete_artifact: Callable[[str], None] | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [
                job
                for job in self._jobs.values()
                if job.status in ("done", "error") and now - job.created_at > ttl_seconds
            ]
            for job in expired:
                del self._jobs[job.id]
        if delete_artifact:
            for job in expired:
                for artifact in job.artifacts:
                    delete_artifact(artifact)


registry = JobRegistry()
