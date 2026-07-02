import time
import unittest

from backend.app.jobs import JobRegistry


def wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class JobRegistryTests(unittest.TestCase):
    def test_successful_job_lifecycle(self) -> None:
        registry = JobRegistry()

        def work(job, reporter):
            reporter.stage("half", 0.5, 1.0, "halfway")
            return {"value": 42}

        job_id = registry.submit(work)
        self.assertTrue(wait_for(lambda: registry.get(job_id).status == "done"))
        job = registry.get(job_id)
        self.assertEqual(job.result, {"value": 42})
        self.assertEqual(job.progress, 1.0)

    def test_failed_job_reports_error(self) -> None:
        registry = JobRegistry()

        def work(job, reporter):
            raise RuntimeError("boom")

        job_id = registry.submit(work)
        self.assertTrue(wait_for(lambda: registry.get(job_id).status == "error"))
        self.assertEqual(registry.get(job_id).error, "boom")

    def test_progress_reporter_interpolates_within_stage(self) -> None:
        registry = JobRegistry()
        checkpoints = []

        def work(job, reporter):
            reporter.stage("denoise", 0.2, 0.6)
            reporter.tick(0.5)
            checkpoints.append(job.progress)
            return None

        job_id = registry.submit(work)
        self.assertTrue(wait_for(lambda: registry.get(job_id).status == "done"))
        self.assertAlmostEqual(checkpoints[0], 0.4)

    def test_purge_expired_removes_finished_jobs_and_artifacts(self) -> None:
        registry = JobRegistry()
        deleted = []

        def work(job, reporter):
            job.artifacts.append("/tmp/fake-artifact")
            return None

        job_id = registry.submit(work)
        self.assertTrue(wait_for(lambda: registry.get(job_id).status == "done"))
        registry.get(job_id).created_at -= 10_000.0
        registry.purge_expired(ttl_seconds=1.0, delete_artifact=deleted.append)
        self.assertIsNone(registry.get(job_id))
        self.assertEqual(deleted, ["/tmp/fake-artifact"])

    def test_running_jobs_are_not_purged(self) -> None:
        registry = JobRegistry()

        def work(job, reporter):
            time.sleep(0.2)
            return None

        job_id = registry.submit(work)
        registry.purge_expired(ttl_seconds=0.0)
        self.assertIsNotNone(registry.get(job_id))
        self.assertTrue(wait_for(lambda: registry.get(job_id).status == "done"))


if __name__ == "__main__":
    unittest.main()
