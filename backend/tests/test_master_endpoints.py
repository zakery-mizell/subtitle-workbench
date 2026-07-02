import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from backend.app import main
from backend.app.mastering.schemas import (
    CutRegionModel,
    DenoiseReport,
    HighPassReport,
    HumReport,
    LevelerReport,
    LoudnessStats,
    MasteringReport,
    MasteringResult,
)


def fake_result() -> MasteringResult:
    stats = LoudnessStats(integrated_lufs=-23.0, lra=10.0, true_peak_dbtp=-3.0, noise_floor_dbfs=-60.0)
    after = LoudnessStats(integrated_lufs=-16.0, lra=6.0, true_peak_dbtp=-1.0, noise_floor_dbfs=-68.0)
    return MasteringResult(
        token="m_test123",
        output_filename="clip.master.wav",
        output_format="wav",
        duration_before=10.0,
        duration_after=9.5,
        report=MasteringReport(
            before=stats,
            after=after,
            hum=HumReport(detected=False),
            denoise=DenoiseReport(applied=False),
            high_pass=HighPassReport(applied=False),
            leveler=LevelerReport(applied=False),
        ),
        cut_list=[CutRegionModel(start=1.0, end=1.5, reason="filler", label="um")],
    )


def wait_for_done(client: TestClient, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in ("done", "error"):
            return payload
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


class MasterEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)

    def test_invalid_params_json_is_rejected(self) -> None:
        response = self.client.post(
            "/api/master",
            files={"audio": ("clip.wav", b"RIFF0000WAVE", "audio/wav")},
            data={"params_json": '{"loudness": {"target_lufs": "very loud"}}'},
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_words_json_is_rejected(self) -> None:
        response = self.client.post(
            "/api/master",
            files={"audio": ("clip.wav", b"RIFF0000WAVE", "audio/wav")},
            data={"params_json": "{}", "words_json": '"not a list"'},
        )
        self.assertEqual(response.status_code, 422)

    def test_master_job_runs_to_completion(self) -> None:
        with mock.patch.object(main, "run_mastering", return_value=fake_result()):
            response = self.client.post(
                "/api/master",
                files={"audio": ("clip.wav", b"RIFF0000WAVE", "audio/wav")},
                data={"params_json": '{"loudness": {"preset": "podcast"}}'},
            )
            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]
            payload = wait_for_done(self.client, job_id)

        self.assertEqual(payload["status"], "done")
        result = payload["result"]
        self.assertEqual(result["token"], "m_test123")
        self.assertEqual(result["report"]["after"]["integrated_lufs"], -16.0)
        self.assertEqual(len(result["cut_list"]), 1)

    def test_failed_job_reports_error_status(self) -> None:
        with mock.patch.object(main, "run_mastering", side_effect=RuntimeError("decode failed")):
            response = self.client.post(
                "/api/master",
                files={"audio": ("clip.wav", b"RIFF0000WAVE", "audio/wav")},
                data={"params_json": "{}"},
            )
            job_id = response.json()["job_id"]
            payload = wait_for_done(self.client, job_id)

        self.assertEqual(payload["status"], "error")
        self.assertIn("decode failed", payload["error"])

    def test_unknown_job_returns_404(self) -> None:
        self.assertEqual(self.client.get("/api/jobs/j_missing").status_code, 404)

    def test_unknown_master_token_returns_404(self) -> None:
        self.assertEqual(self.client.get("/api/master/m_zzz/audio").status_code, 404)
        self.assertEqual(self.client.get("/api/master/m_zzz/waveform").status_code, 404)

    def test_cut_list_export_formats(self) -> None:
        with mock.patch.object(main, "run_mastering", return_value=fake_result()):
            response = self.client.post(
                "/api/master",
                files={"audio": ("clip.wav", b"RIFF0000WAVE", "audio/wav")},
                data={"params_json": "{}"},
            )
            wait_for_done(self.client, response.json()["job_id"])

        as_json = self.client.get("/api/master/m_test123/cut-list")
        self.assertEqual(as_json.status_code, 200)
        self.assertEqual(as_json.json()["cut_list"][0]["label"], "um")

        as_labels = self.client.get("/api/master/m_test123/cut-list", params={"format": "audacity"})
        self.assertEqual(as_labels.status_code, 200)
        self.assertEqual(as_labels.text, "1.000000\t1.500000\tum\n")


if __name__ == "__main__":
    unittest.main()
