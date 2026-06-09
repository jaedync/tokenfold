"""Fix 4: ingest validation — field-length limits and speed/geo coercion."""

import unittest

from app.tests._support import TempDBTestCase


def _arec(uuid="ua", **usage_extra):
    u = {"input_tokens": 1, "output_tokens": 1}
    u.update(usage_extra)
    return {
        "uuid": uuid,
        "type": "assistant",
        "timestamp": "2026-06-09T12:00:00Z",
        "sessionId": "s1",
        "requestId": "r1",
        "message": {
            "model": "claude-opus-4-8",
            "id": "m1",
            "usage": u,
        },
    }


class IngestFieldLengthValidationTest(TempDBTestCase):
    """Pydantic max_length constraints must reject overlong fields with 422."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_overlong_org_name_returns_422(self):
        """POST with 10,000-char org_name must return 422, not 500/stored."""
        body = {
            "machine": "m",
            "project_dir": "p",
            "session_file": "s.jsonl",
            "cursor": {"last_line_num": 0},
            "events": [_arec()],
            "org_name": "x" * 10_000,
        }
        c = self.client()
        r = c.post("/api/ingest", json=body, headers={"X-API-Key": self.api_key})
        self.assertEqual(
            r.status_code, 422,
            f"expected 422 for overlong org_name, got {r.status_code}: {r.text}")


class IngestSpeedCoercionTest(TempDBTestCase):
    """Non-string speed/geo must be coerced to NULL, not raise a 500."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_integer_speed_stored_as_null(self):
        """POST with speed=123 (int) must succeed (200) and store speed as NULL."""
        body = {
            "machine": "m",
            "project_dir": "p",
            "session_file": "s.jsonl",
            "cursor": {"last_line_num": 0},
            "events": [_arec(speed=123)],
        }
        c = self.client()
        r = c.post("/api/ingest", json=body, headers={"X-API-Key": self.api_key})
        self.assertEqual(
            r.status_code, 200,
            f"expected 200 for int speed, got {r.status_code}: {r.text}")
        row = self.conn.execute(
            "SELECT speed FROM events WHERE uuid='ua'"
        ).fetchone()
        self.assertIsNone(
            row["speed"],
            f"speed must be stored as NULL when coerced from int, got {row['speed']!r}")

    def test_dict_service_tier_stored_as_null(self):
        """POST with service_tier={'a': 1} (dict) must succeed (200) with
        service_tier stored as NULL — not 500 the whole batch via a sqlite
        bind InterfaceError (same defect class as the speed/geo bug)."""
        body = {
            "machine": "m",
            "project_dir": "p",
            "session_file": "s.jsonl",
            "cursor": {"last_line_num": 0},
            "events": [_arec(service_tier={"a": 1})],
        }
        c = self.client()
        r = c.post("/api/ingest", json=body, headers={"X-API-Key": self.api_key})
        self.assertEqual(
            r.status_code, 200,
            f"expected 200 for dict service_tier, got {r.status_code}: {r.text}")
        row = self.conn.execute(
            "SELECT service_tier FROM events WHERE uuid='ua'"
        ).fetchone()
        self.assertIsNotNone(row, "event row must have been stored")
        self.assertIsNone(
            row["service_tier"],
            "service_tier must be stored as NULL when coerced from dict, "
            f"got {row['service_tier']!r}")


class IngestEnvelopeLengthBoundsTest(TempDBTestCase):
    """machine/project_dir/session_file must also be length-bounded (422)."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _post(self, **overrides):
        body = {
            "machine": "m",
            "project_dir": "p",
            "session_file": "s.jsonl",
            "cursor": {"last_line_num": 0},
            "events": [_arec()],
        }
        body.update(overrides)
        c = self.client()
        return c.post("/api/ingest", json=body, headers={"X-API-Key": self.api_key})

    def test_overlong_machine_returns_422(self):
        r = self._post(machine="m" * 10_000)
        self.assertEqual(r.status_code, 422,
                         f"expected 422 for overlong machine, got {r.status_code}")

    def test_overlong_project_dir_returns_422(self):
        r = self._post(project_dir="p" * 10_000)
        self.assertEqual(r.status_code, 422,
                         f"expected 422 for overlong project_dir, got {r.status_code}")

    def test_overlong_session_file_returns_422(self):
        r = self._post(session_file="s" * 10_000)
        self.assertEqual(r.status_code, 422,
                         f"expected 422 for overlong session_file, got {r.status_code}")


if __name__ == "__main__":
    unittest.main()
