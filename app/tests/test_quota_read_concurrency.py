"""Quota work must preserve the existing independent-read/event-loop boundary."""
import asyncio
import threading
from unittest.mock import patch

import httpx
from app.tests._support import TempDBTestCase


class QuotaReadConcurrencyTest(TempDBTestCase):
    def test_health_is_responsive_while_rate_limit_cost_work_is_blocked(self):
        from app.main import app
        entered = threading.Event()
        release = threading.Event()

        def cost(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise AssertionError("cost worker release timed out")
            return 0

        async def exercise():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                task = asyncio.create_task(client.get("/api/rate-limits?scope=personal"))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                    health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
                    self.assertEqual(health.status_code, 200)
                    self.assertFalse(task.done())
                finally:
                    release.set()
                    response = await task
                self.assertEqual(response.status_code, 200)

        with patch("app.api.compute_window_cost", side_effect=cost):
            asyncio.run(exercise())

    def test_snapshot_does_not_wait_for_shared_write_transaction(self):
        from app.db import write_txn
        entered = threading.Event()
        release = threading.Event()

        def writer():
            with write_txn() as conn:
                conn.execute("INSERT INTO meta VALUES('pending','1')")
                entered.set()
                release.wait(3)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            response = self.client().get("/api/rate-limit-snapshots?scope=personal")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(thread.is_alive())
        finally:
            release.set()
            thread.join(4)
