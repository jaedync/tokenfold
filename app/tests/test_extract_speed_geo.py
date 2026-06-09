import unittest

from app.ingest import _extract_event, EVENT_COLS


def _arec(**usage):
    u = {"input_tokens": 10, "output_tokens": 20}
    u.update(usage)
    return {
        "uuid": "u1", "type": "assistant", "timestamp": "2026-06-09T12:00:00Z",
        "sessionId": "s1", "requestId": "r1",
        "message": {"model": "claude-opus-4-8", "id": "m1", "usage": u},
    }


class ExtractSpeedGeoTest(unittest.TestCase):
    def test_captures_speed_and_geo(self):
        row = _extract_event(_arec(speed="fast", inference_geo="us"), "mach", "proj")
        self.assertEqual(row["speed"], "fast")
        self.assertEqual(row["inference_geo"], "us")

    def test_speed_geo_default_none(self):
        row = _extract_event(_arec(), "mach", "proj")
        self.assertIsNone(row["speed"])
        self.assertIsNone(row["inference_geo"])

    def test_event_cols_wired(self):
        # If these aren't in EVENT_COLS the INSERT silently drops them.
        self.assertIn("speed", EVENT_COLS)
        self.assertIn("inference_geo", EVENT_COLS)


if __name__ == "__main__":
    unittest.main()
