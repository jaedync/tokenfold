import time

from app.tests._support import TempDBTestCase
from app.tests.test_summarizer_pricing import insert_assistant


def _tag_enterprise(conn):
    """Bring seeded events in-scope for the enterprise-only window cost filter.

    insert_assistant() leaves plan/org/account NULL; compute_window_cost is now
    fail-closed to verified-enterprise usage (plan, org_name, AND account_email
    must all be set), so seed data must be fully tagged.
    """
    conn.execute(
        "UPDATE events SET plan='enterprise', org_name='Acme', "
        "account_email='test@acme.io'"
    )
    conn.commit()


class WindowCostFastGeoTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_window_honors_fast(self):
        from app.cost_windows import compute_window_cost
        insert_assistant(self.conn, "u1", "r1", inp=1_000_000, speed="fast")  # $10
        _tag_enterprise(self.conn)
        got = compute_window_cost(self.conn, 1781000000.0 - 10, 1781000000.0 + 10)
        self.assertAlmostEqual(got, 10.0, places=2)

    def test_window_mixed_same_model(self):
        from app.cost_windows import compute_window_cost
        insert_assistant(self.conn, "u1", "r1", inp=1_000_000)                            # $5 normal
        insert_assistant(self.conn, "u2", "r2", inp=1_000_000, speed="fast", ts=1781000001.0)  # $10 fast
        _tag_enterprise(self.conn)
        got = compute_window_cost(self.conn, 1781000000.0 - 10, 1781000000.0 + 100)
        self.assertAlmostEqual(got, 15.0, places=2)  # must NOT collapse to 2M tokens @ one rate


class AnthropicOnlyWindowCostTest(TempDBTestCase):
    """Claude subscription gauges must not count other Pi providers.

    The 5h/7d "spent \u00b7 this window" figures describe the Claude limit
    windows; Codex/OpenCode/OpenRouter Pi rows (reported costs) must never
    inflate them. Claude Code CLI rows carry source_client='claude-code' and
    no provider; Pi Anthropic rows carry provider='anthropic'.
    """

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _ins(self, uuid, req, *, source_client="claude-code", provider=None,
             model="claude-opus-4-8", inp=1_000_000, reported=0.0,
             ts=1781000000.0):
        """Personal-scope assistant row with optional Pi provider/reported cost."""
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,"
            "request_id,source_machine,project_dir,model,is_sidechain,agent_id,"
            "input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,"
            "speed,inference_geo,source_client,provider,reported_cost_total) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uuid, "assistant", "2026-06-09T12:00:00Z", ts, "2026-06-09", "s",
             req, "m", "proj", model, 0, None, inp, 0, 0, 0, None, None,
             source_client, provider, reported),
        )
        self.conn.commit()

    def test_default_counts_all_providers_at_reported_costs(self):
        from app.cost_windows import compute_window_cost
        self._ins("u1", "r1")  # claude-code, $5 server-priced
        self._ins("u2", "r2", source_client="pi-agent",
                  provider="openai-codex", reported=3.0)
        got = compute_window_cost(self.conn, 1781000000.0 - 10,
                                  1781000000.0 + 10, scope="personal")
        self.assertAlmostEqual(got, 8.0, places=2)

    def test_anthropic_only_counts_claude_cli_and_pi_anthropic(self):
        from app.cost_windows import compute_window_cost
        self._ins("u1", "r1")  # claude-code, $5
        self._ins("u2", "r2", source_client="pi-agent",
                  provider="anthropic", reported=1.25)
        got = compute_window_cost(self.conn, 1781000000.0 - 10,
                                  1781000000.0 + 10, scope="personal",
                                  anthropic_only=True)
        self.assertAlmostEqual(got, 6.25, places=2)

    def test_legacy_claude_rows_with_null_source_still_count(self):
        """Non-Pi rows (any client) are Claude rows regardless of provider."""
        from app.cost_windows import compute_window_cost
        self._ins("u1", "r1", source_client="claude-desktop")
        self._ins("u2", "r2", source_client="pi-agent",
                  provider="openai-codex", reported=3.0)
        got = compute_window_cost(self.conn, 1781000000.0 - 10,
                                  1781000000.0 + 10, scope="personal",
                                  anthropic_only=True)
        self.assertAlmostEqual(got, 5.0, places=2)

    def test_anthropic_only_excludes_other_pi_providers(self):
        from app.cost_windows import compute_window_cost
        self._ins("u1", "r1")  # claude-code, $5
        for i, prov in enumerate(("openai-codex", "opencode-go",
                                  "openrouter", "huggingface"), 1):
            self._ins(f"u{i+1}", f"r{i+1}", source_client="pi-agent",
                      provider=prov, reported=10.0)
        got = compute_window_cost(self.conn, 1781000000.0 - 10,
                                  1781000000.0 + 10, scope="personal",
                                  anthropic_only=True)
        self.assertAlmostEqual(got, 5.0, places=2)


class StreamedChunksMixedSpeedDedupTest(TempDBTestCase):
    """ONE request whose streamed chunks mix speed=NULL and speed='fast' must
    price as fast ($10) in ALL THREE cost paths.

    Streaming chunks repeat token counts; dedup takes MAX(tokens) per request_id.
    A bare (non-aggregated) speed column in that inner GROUP BY takes an
    ARBITRARY row's value in SQLite — so the same request could price at base in
    one path and fast in another. MAX(speed) (NULL-skipping, deterministic) is
    the required semantics everywhere.
    """

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _seed_streamed_fast_request(self):
        """One enterprise Opus 4.8 request, two chunk rows, MAX input = 1M.

        The NULL-speed chunk is inserted FIRST (lowest rowid): observed SQLite
        behavior picks the FIRST row in scan order for a bare column alongside
        multiple MAX() aggregates, so a bare `speed` column resolves to NULL
        here and prices at base — exposing the bug. MAX(speed) skips NULLs and
        deterministically yields 'fast' regardless of chunk order.
        """
        now = time.time()
        # chunk 1: speed=NULL (field absent in this chunk), partial token count
        insert_assistant(self.conn, "u1", "rX", inp=400_000, speed=None,
                         ts=now - 120)
        # chunk 2: speed='fast', full MAX token count
        insert_assistant(self.conn, "u2", "rX", inp=1_000_000, speed="fast",
                         ts=now - 60)
        _tag_enterprise(self.conn)
        return now

    def test_window_cost_prices_mixed_chunks_as_fast(self):
        from app.cost_windows import compute_window_cost
        now = self._seed_streamed_fast_request()
        got = compute_window_cost(self.conn, now - 3600, now + 10)
        self.assertAlmostEqual(
            got, 10.0, places=2,
            msg=f"mixed NULL/fast chunks must price as fast ($10), got {got}")

    def test_rate_limits_week_cost_prices_mixed_chunks_as_fast(self):
        self._seed_streamed_fast_request()
        import app.aggregator as agg
        agg._cached_data.clear()
        c = self.client()
        rl = c.get("/api/rate-limits").json()["weekly_budget"]
        self.assertAlmostEqual(
            rl["week_cost"], 10.0, places=2,
            msg=f"week_cost must be $10 (fast), got {rl['week_cost']}")

    def test_aggregator_hourly_prices_mixed_chunks_as_fast(self):
        from app.aggregator import _build_hourly
        from app.config import ENTERPRISE_PRED
        self._seed_streamed_fast_request()
        hourly = _build_hourly(self.conn, ENTERPRISE_PRED)
        total = sum(h["cost"] for h in hourly)
        self.assertAlmostEqual(
            total, 10.0, places=2,
            msg=f"hourly cost sum must be $10 (fast), got {total}")
