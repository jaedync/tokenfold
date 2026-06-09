"""Two-tier auth tests.

Human routes (/, /api/stats, /api/stats/version, /api/rate-limits) protected
by HTTP Basic Auth when DASHBOARD_PASSWORD is set; open when unset.

Machine route (/api/ha) protected by X-API-Key always (regardless of
DASHBOARD_PASSWORD).

/health is unconditionally open.
"""

import unittest
from unittest.mock import patch

import app.config
from app.tests._support import TempDBTestCase

HUMAN_ROUTES = ["/", "/api/stats", "/api/stats/version", "/api/rate-limits"]


class DashboardAuthEnforcedTest(TempDBTestCase):
    """When DASHBOARD_PASSWORD is set, human routes require Basic Auth."""

    def setUp(self):
        super().setUp()

    # --- 1. No credentials → 401 with WWW-Authenticate: Basic header ---

    def test_no_creds_slash_returns_401(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/")
        self.assertEqual(r.status_code, 401)
        self.assertIn("Basic", r.headers.get("WWW-Authenticate", ""))

    def test_no_creds_api_stats_returns_401(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/api/stats")
        self.assertEqual(r.status_code, 401)
        self.assertIn("Basic", r.headers.get("WWW-Authenticate", ""))

    def test_no_creds_api_stats_version_returns_401(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/api/stats/version")
        self.assertEqual(r.status_code, 401)
        self.assertIn("Basic", r.headers.get("WWW-Authenticate", ""))

    def test_no_creds_api_rate_limits_returns_401(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/api/rate-limits")
        self.assertEqual(r.status_code, 401)
        self.assertIn("Basic", r.headers.get("WWW-Authenticate", ""))

    # --- 2. Correct credentials → 200 ---

    def test_correct_creds_slash_returns_200(self):
        with patch.object(app.config, "DASHBOARD_USER", "admin"), \
             patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/", auth=("admin", "secret"))
        self.assertEqual(r.status_code, 200)

    def test_correct_creds_api_stats_returns_200(self):
        with patch.object(app.config, "DASHBOARD_USER", "admin"), \
             patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/api/stats", auth=("admin", "secret"))
        self.assertEqual(r.status_code, 200)

    def test_correct_creds_api_stats_version_returns_200(self):
        with patch.object(app.config, "DASHBOARD_USER", "admin"), \
             patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/api/stats/version", auth=("admin", "secret"))
        self.assertEqual(r.status_code, 200)

    def test_correct_creds_api_rate_limits_returns_200(self):
        with patch.object(app.config, "DASHBOARD_USER", "admin"), \
             patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/api/rate-limits", auth=("admin", "secret"))
        self.assertEqual(r.status_code, 200)

    # --- 3. Wrong password → 401 ---

    def test_wrong_password_returns_401(self):
        with patch.object(app.config, "DASHBOARD_USER", "admin"), \
             patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/api/stats", auth=("admin", "wrongpassword"))
        self.assertEqual(r.status_code, 401)

    # --- 3. Wrong username → 401 (same status, no distinction) ---

    def test_wrong_username_returns_401(self):
        with patch.object(app.config, "DASHBOARD_USER", "admin"), \
             patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/api/stats", auth=("wronguser", "secret"))
        self.assertEqual(r.status_code, 401)

    # --- 4. DASHBOARD_PASSWORD unset → open (no auth needed) ---

    def test_open_when_password_unset_slash(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", ""):
            c = self.client()
            r = c.get("/")
        self.assertEqual(r.status_code, 200)

    def test_open_when_password_unset_api_stats(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", ""):
            c = self.client()
            r = c.get("/api/stats")
        self.assertEqual(r.status_code, 200)

    def test_open_when_password_unset_api_rate_limits(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", ""):
            c = self.client()
            r = c.get("/api/rate-limits")
        self.assertEqual(r.status_code, 200)


class HAMachineAuthTest(TempDBTestCase):
    """GET /api/ha always requires the X-API-Key (machine auth)."""

    def setUp(self):
        super().setUp()

    def test_ha_without_key_returns_401(self):
        c = self.client()
        r = c.get("/api/ha")
        self.assertEqual(r.status_code, 401)

    def test_ha_with_correct_key_returns_200(self):
        # api_key is already patched into app.config.STATS_API_KEY by TempDBTestCase.setUp
        c = self.client()
        r = c.get("/api/ha", headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200)

    def test_ha_with_wrong_key_returns_401(self):
        # api_key is already patched into app.config.STATS_API_KEY by TempDBTestCase.setUp
        c = self.client()
        r = c.get("/api/ha", headers={"X-API-Key": "wrong-key"})
        self.assertEqual(r.status_code, 401)


class IngestAfterRefactorTest(TempDBTestCase):
    """POST /api/ingest still requires X-API-Key after refactoring to shared dep."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _minimal_body(self):
        return {
            "machine": "m",
            "project_dir": "p",
            "session_file": "s.jsonl",
            "cursor": {"last_line_num": 0},
            "events": [
                {
                    "uuid": "u1",
                    "type": "assistant",
                    "timestamp": "2026-06-09T12:00:00Z",
                    "sessionId": "s1",
                    "requestId": "r1",
                    "message": {
                        "model": "claude-opus-4-8",
                        "id": "m1",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }
            ],
        }

    def test_ingest_with_key_returns_200(self):
        c = self.client()
        r = c.post("/api/ingest", json=self._minimal_body(),
                   headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200)

    def test_ingest_without_key_returns_401(self):
        c = self.client()
        r = c.post("/api/ingest", json=self._minimal_body())
        self.assertEqual(r.status_code, 401)


class HealthOpenTest(TempDBTestCase):
    """/health is always open — DASHBOARD_PASSWORD doesn't block it."""

    def test_health_open_when_password_unset(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", ""):
            c = self.client()
            r = c.get("/health")
        self.assertEqual(r.status_code, 200)

    def test_health_open_when_password_set(self):
        with patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            c = self.client()
            r = c.get("/health")
        self.assertEqual(r.status_code, 200)


class EmptyApiKeyFailClosedTest(TempDBTestCase):
    """Regression guard for require_api_key's `not expected` short-circuit.

    When STATS_API_KEY is unset (""), machine routes must reject EVERY request —
    including one sending an empty `X-API-Key: ""` header. Without the
    `not expected or` guard, `compare_digest("", "")` returns True and the route
    would silently open. This is the riskiest untested path: every other machine
    test runs with STATS_API_KEY set, so a future "simplification" dropping the
    guard would stay green without this test.
    """

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _minimal_ingest_body(self):
        return {
            "machine": "m",
            "project_dir": "p",
            "session_file": "s.jsonl",
            "cursor": {"last_line_num": 0},
            "events": [
                {
                    "uuid": "u1",
                    "type": "assistant",
                    "timestamp": "2026-06-09T12:00:00Z",
                    "sessionId": "s1",
                    "requestId": "r1",
                    "message": {
                        "model": "claude-opus-4-8",
                        "id": "m1",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }
            ],
        }

    def test_ha_no_header_rejected_when_key_empty(self):
        with patch.object(app.config, "STATS_API_KEY", ""):
            c = self.client()
            r = c.get("/api/ha")
        self.assertEqual(r.status_code, 401)

    def test_ha_empty_header_rejected_when_key_empty(self):
        """The bypass that must NOT pass: empty key vs empty header."""
        with patch.object(app.config, "STATS_API_KEY", ""):
            c = self.client()
            r = c.get("/api/ha", headers={"X-API-Key": ""})
        self.assertEqual(r.status_code, 401)

    def test_ingest_no_header_rejected_when_key_empty(self):
        with patch.object(app.config, "STATS_API_KEY", ""):
            c = self.client()
            r = c.post("/api/ingest", json=self._minimal_ingest_body())
        self.assertEqual(r.status_code, 401)

    def test_ingest_empty_header_rejected_when_key_empty(self):
        """The bypass that must NOT pass: empty key vs empty header."""
        with patch.object(app.config, "STATS_API_KEY", ""):
            c = self.client()
            r = c.post("/api/ingest", json=self._minimal_ingest_body(),
                       headers={"X-API-Key": ""})
        self.assertEqual(r.status_code, 401)


class DocsRoutesDisabledTest(TempDBTestCase):
    """The auto-generated OpenAPI docs routes must be disabled (404)."""

    def test_openapi_json_returns_404(self):
        c = self.client()
        r = c.get("/openapi.json")
        self.assertEqual(r.status_code, 404)

    def test_docs_returns_404(self):
        c = self.client()
        r = c.get("/docs")
        self.assertEqual(r.status_code, 404)

    def test_redoc_returns_404(self):
        c = self.client()
        r = c.get("/redoc")
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
