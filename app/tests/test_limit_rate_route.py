"""GET /api/limit-rate: cache, scope and degradation (app/limit_rate.py)."""

import json
import threading
import time
from unittest.mock import patch

from app.tests._support import TempDBTestCase
from app.tests.test_limit_rate_data import insert_event, insert_reading

KEYS = {"generated_at", "period_start", "fit_quality", "usd_per_point", "steps",
        "families", "peak", "cache_read_weight"}


class _Base(TempDBTestCase):

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        import app.limit_rate as lr
        self.lr = lr
        lr.reset_state()
        self.addCleanup(lr.reset_state)
        for name, value in (("DASHBOARD_PASSWORD", ""), ("LOCKED_SCOPE", None)):
            p = patch("app.config." + name, value)
            p.start()
            self.addCleanup(p.stop)
        self.http = self.client()
        self.addCleanup(self.http.close)

    def get(self, qs="scope=personal"):
        r = self.http.get("/api/limit-rate?" + qs)
        return r

    def seed(self, days=4):
        """Steady Opus use: $2 per minute-poll, 1 point per $24 (7 d)."""
        now = time.time()
        start = now - days * 86400
        acc7 = acc5 = 0.0
        w5 = start + 5 * 3600
        t = start
        i = 0
        while t < now:
            insert_event(self.conn, f"e{i}", t + 1, out=80_000)  # $2 at Opus 4.8
            acc7 += 2 / 24.0
            acc5 += 2 / 3.6
            if t >= w5:
                w5 = t + 5 * 3600
                acc5 = 0.0
            insert_reading(self.conn, "seven_day", t + 60, float(int(acc7)), start + 7 * 86400,
                           source="server")
            insert_reading(self.conn, "five_hour", t + 60, float(int(acc5)), w5, source="server")
            t += 120
            i += 1
        self.conn.commit()


class LimitRateRouteTest(_Base):

    def test_shape_and_values_on_seeded_data(self):
        self.seed()
        r = self.get()
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(set(body), KEYS)
        self.assertEqual(body["fit_quality"]["status"], "ok", body["fit_quality"])
        self.assertAlmostEqual(body["usd_per_point"]["seven_day"]["value"], 24.0, delta=1.5)
        self.assertAlmostEqual(body["usd_per_point"]["five_hour"]["value"], 3.6, delta=0.3)
        fam = body["families"][0]
        self.assertEqual(fam["family"], "opus")
        self.assertAlmostEqual(fam["usd_per_point"], 24.0, delta=1.5)
        for key in ("active", "factor", "until", "tz", "hours", "next_start"):
            self.assertIn(key, body["peak"])
        self.assertEqual(body["period_start"] % 60, 0)

    def test_cached_in_memory_and_meta(self):
        self.seed(days=2)
        first = self.get().json()
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='limit_rate_cache'").fetchone()
        self.assertIsNotNone(row)
        stored = json.loads(row[0])
        self.assertEqual(stored["payload"]["usd_per_point"], first["usd_per_point"])
        with patch.object(self.lr, "compute_payload",
                          side_effect=AssertionError("must serve the cache")):
            again = self.get().json()
        self.assertEqual(again["generated_at"], first["generated_at"])

    def test_meta_cache_survives_restart(self):
        self.seed(days=2)
        first = self.get().json()
        self.lr.reset_state()  # a fresh process: memory empty, meta kept
        with patch.object(self.lr, "compute_payload",
                          side_effect=AssertionError("must load meta")):
            again = self.get().json()
        self.assertEqual(again["usd_per_point"], first["usd_per_point"])

    def test_slow_fit_serves_null_then_result(self):
        gate = threading.Event()
        real = self.lr.compute_payload

        def slow(*a, **kw):
            gate.wait(5)
            return real(*a, **kw)

        with patch.object(self.lr, "WAIT_S", 0.05), \
                patch.object(self.lr, "compute_payload", side_effect=slow):
            t0 = time.monotonic()
            body = self.get().json()
            self.assertLess(time.monotonic() - t0, 1.0)
            self.assertEqual(set(body), KEYS)
            self.assertEqual(body["fit_quality"]["status"], "computing")
            self.assertIsNone(body["usd_per_point"]["seven_day"]["value"])
            gate.set()
            self.lr.join_refresh(5)
        self.assertNotEqual(self.get().json()["fit_quality"]["status"], "computing")

    def test_stale_cache_is_served_while_refreshing(self):
        self.seed(days=2)
        first = self.get().json()
        old = time.time() - 2 * self.lr.RECOMPUTE_S
        self.lr._state["entry"] = dict(self.lr._state["entry"], computed_at=old)
        gate = threading.Event()

        def slow_failure(*a):
            gate.wait(5)
            raise RuntimeError("late failure")

        with patch.object(self.lr, "WAIT_S", 0.05), \
                patch.object(self.lr, "compute_payload", side_effect=slow_failure), \
                self.assertLogs("app.limit_rate", "ERROR"):
            body = self.get().json()
            gate.set()
            self.lr.join_refresh(5)
        self.assertEqual(body["fit_quality"]["status"], "stale")
        self.assertIn("min ago", body["fit_quality"]["note"])
        self.assertEqual(body["usd_per_point"], first["usd_per_point"])

    def test_failed_fit_degrades_and_backs_off(self):
        with patch.object(self.lr, "compute_payload", side_effect=RuntimeError("boom")) as m, \
                self.assertLogs("app.limit_rate", "ERROR"):
            r = self.get()
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["fit_quality"]["status"], "error")
            self.assertNotIn("boom", json.dumps(body))
            self.get()
            self.assertEqual(m.call_count, 1)  # no retry storm

    def test_enterprise_scope_is_null_not_error(self):
        r = self.get("scope=enterprise")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(set(body), KEYS)
        self.assertEqual(body["fit_quality"]["status"], "not_available_for_scope")
        self.assertEqual(body["families"], [])

    def test_invalid_scope_400(self):
        self.assertEqual(self.get("scope=bogus").status_code, 400)

    def test_requires_dashboard_auth(self):
        with patch("app.config.DASHBOARD_USER", "u"), \
                patch("app.config.DASHBOARD_PASSWORD", "p"):
            self.assertEqual(self.get().status_code, 401)


class LimitRatePrewarmTest(_Base):
    """App startup starts the fit, so the first request after a deploy finds
    a result instead of "computing"."""

    def test_prewarm_fits_in_the_background(self):
        self.seed(days=2)
        gate = threading.Event()
        real = self.lr.compute_payload

        def slow(*a, **kw):
            gate.wait(5)
            return real(*a, **kw)

        with patch.object(self.lr, "compute_payload", side_effect=slow):
            t0 = time.monotonic()
            thread = self.lr.prewarm()
            self.assertLess(time.monotonic() - t0, 0.5, "prewarm must not block startup")
            self.assertIsNotNone(thread)
            gate.set()
            self.lr.join_refresh(5)
        with patch.object(self.lr, "compute_payload",
                          side_effect=AssertionError("prewarm already fitted")):
            body = self.get().json()
        self.assertEqual(body["fit_quality"]["status"], "ok")

    def test_prewarm_skips_a_fresh_meta_cache(self):
        self.seed(days=2)
        self.get()
        self.lr.reset_state()
        with patch.object(self.lr, "compute_payload",
                          side_effect=AssertionError("cache is fresh")):
            self.assertIsNone(self.lr.prewarm())

    def test_prewarm_skips_an_enterprise_instance(self):
        with patch("app.config.LOCKED_SCOPE", "enterprise"), \
                patch.object(self.lr, "compute_payload",
                             side_effect=AssertionError("no limits here")):
            self.assertIsNone(self.lr.prewarm())

    def test_prewarm_failure_never_raises(self):
        with patch.object(self.lr, "_entry", side_effect=RuntimeError("db gone")), \
                self.assertLogs("app.limit_rate", "ERROR"):
            self.assertIsNone(self.lr.prewarm())


class LifespanPrewarmTest(_Base):
    def test_lifespan_starts_the_limit_rate_fit(self):
        import asyncio
        import app.main as main
        stubs = [patch.object(main, "load_pricing"),
                 patch.object(main.usage_fetcher, "should_run", return_value=False),
                 patch.object(main, "init_light", new=lambda: asyncio.sleep(0)),
                 patch.object(main, "start_watchdog"), patch.object(main, "stop_watchdog"),
                 patch.object(main, "close_conn"),
                 patch.object(main, "_backfill_if_needed"),
                 patch("app.aggregator.trigger_eager_rebuild"),
                 patch("app.aggregator.start_sweeps"), patch("app.aggregator.stop_sweeps"),
                 patch.object(self.lr, "prewarm"), patch("app.prewarm.start"),
                 patch("app.prewarm.join"),
                 patch("app.config.PREWARM", True)]
        for p in stubs:
            p.start()
            self.addCleanup(p.stop)

        async def run():
            async with main.lifespan(main.app):
                pass

        with patch("builtins.print"):
            asyncio.run(run())
        self.lr.prewarm.assert_called_once_with()

    def test_shutdown_waits_for_a_running_fit_before_closing_the_db(self):
        # The fit stores its result through the shared connection; closing
        # that connection under a running fit crashes sqlite.
        import asyncio
        import app.main as main
        order = []
        stubs = [patch.object(main, "load_pricing"),
                 patch.object(main.usage_fetcher, "should_run", return_value=False),
                 patch.object(main, "init_light", new=lambda: asyncio.sleep(0)),
                 patch.object(main, "start_watchdog"), patch.object(main, "stop_watchdog"),
                 patch.object(main, "_backfill_if_needed"),
                 patch("app.aggregator.trigger_eager_rebuild"),
                 patch("app.aggregator.start_sweeps"), patch("app.aggregator.stop_sweeps"),
                 patch.object(self.lr, "prewarm"), patch("app.prewarm.start"),
                 patch("app.prewarm.join"),
                 patch.object(self.lr, "join_refresh", side_effect=lambda t: order.append("join")),
                 patch.object(main, "close_conn", side_effect=lambda: order.append("close"))]
        for p in stubs:
            p.start()
            self.addCleanup(p.stop)

        async def run():
            async with main.lifespan(main.app):
                pass

        with patch("builtins.print"):
            asyncio.run(run())
        self.assertEqual(order, ["join", "close"])
