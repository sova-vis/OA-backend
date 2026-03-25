import time
import unittest

try:
    from fastapi.testclient import TestClient
    from oa_main_pipeline.api import create_app
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]


class _NoWarmupService:
    def warmup(self) -> None:
        return

    def evaluate(self, request, debug: bool = False):  # pragma: no cover
        raise NotImplementedError()


@unittest.skipIf(TestClient is None or create_app is None, "fastapi testclient not available")
class Phase6HealthReadyTests(unittest.TestCase):
    def setUp(self) -> None:
        app = create_app(service=_NoWarmupService())  # type: ignore[arg-type]
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/oa-level/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("status"), "ok")
        self.assertTrue(str(payload.get("service") or ""))

    def test_ready_endpoint_returns_json(self) -> None:
        # Warmup runs in a background thread; allow a short window for it to complete.
        deadline = time.time() + 2.0
        last = None
        while time.time() < deadline:
            last = self.client.get("/oa-level/ready")
            if last.status_code == 200:
                break
            time.sleep(0.05)
        self.assertIsNotNone(last)
        payload = last.json()  # type: ignore[union-attr]
        self.assertIn("ready", payload)
        self.assertIn("warmup_done", payload)
