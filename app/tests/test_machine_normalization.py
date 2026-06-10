"""Machine identity normalization (UX P1-7).

The same physical machine reports under hostname variants — bare hostname,
FQDN with a Tailscale/mDNS suffix, different casing — and was counted as
multiple machines in every chart, pill, and table. Normalization happens at
READ time in the aggregator (database rows are never rewritten):

* lowercase
* strip domain suffixes (first DNS label wins): x.tailnet.ts.net -> x, y.local -> y
* small alias map for variants that differ structurally
"""

import unittest

from app.aggregator import canonical_machine
from app.tests._support import TempDBTestCase


class CanonicalMachineTest(unittest.TestCase):

    def test_strips_tailscale_fqdn(self):
        self.assertEqual(
            canonical_machine("jaedyns-macbook-pro.tailedc58.ts.net"),
            "jaedyns-macbook-pro")

    def test_strips_local_suffix(self):
        self.assertEqual(canonical_machine("ms01arch.local"), "ms01arch")

    def test_lowercases(self):
        self.assertEqual(canonical_machine("MS01ARCH"), "ms01arch")

    def test_alias_map_merges_bare_variant(self):
        # macbook-pro is the same Mac as jaedyns-macbook-pro (alias map)
        self.assertEqual(canonical_machine("macbook-pro"), "jaedyns-macbook-pro")

    def test_alias_applies_after_suffix_strip_and_lowercase(self):
        self.assertEqual(
            canonical_machine("MacBook-Pro.local"), "jaedyns-macbook-pro")

    def test_unknown_name_passes_through(self):
        self.assertEqual(canonical_machine("win11-dev-vm"), "win11-dev-vm")

    def test_empty_and_none_are_safe(self):
        self.assertIsNone(canonical_machine(None))
        self.assertEqual(canonical_machine(""), "")


def _ins(conn, uuid, req, machine, day, ts, inp=1_000_000, prompts=False):
    conn.execute(
        "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,session_id,request_id,"
        "source_machine,project_dir,model,is_sidechain,agent_id,input_tokens,"
        "output_tokens,cache_creation_tokens,cache_read_tokens,account_email,plan,"
        "org_name,is_human_prompt,user_type) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (uuid, "assistant", day + "T12:00:00Z", ts, day, "sess-" + uuid, req,
         machine, "-home-x-proj", "claude-opus-4-8", 0, None, inp, 0, 0, 0,
         "me@personal.dev", "max", None, 0, None))
    conn.commit()


class AggregatorNormalizationTest(TempDBTestCase):
    """Three hostname variants of one Mac collapse to ONE machine everywhere."""

    VARIANTS = ("macbook-pro", "jaedyns-macbook-pro",
                "jaedyns-macbook-pro.tailedc58.ts.net")

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        from datetime import datetime
        from app.aggregator import TZ
        now = datetime.now(TZ)
        self.today = now.strftime("%Y-%m-%d")
        ts = now.timestamp() - 3600
        for i, m in enumerate(self.VARIANTS):
            _ins(self.conn, f"e{i}", f"r{i}", m, self.today, ts + i)
        _ins(self.conn, "e9", "r9", "ms01arch", self.today, ts)

        from app.summarizer import summarize_days
        summarize_days([self.today])
        import app.aggregator as agg
        agg._cached_data.clear()
        self.data = agg._build_dashboard_data_inner("personal")

    def test_machines_list_is_canonical(self):
        self.assertEqual(self.data["machines"],
                         ["jaedyns-macbook-pro", "ms01arch"])

    def test_machine_summary_merged(self):
        names = [m["machine"] for m in self.data["machine_summary"]]
        self.assertEqual(sorted(names), ["jaedyns-macbook-pro", "ms01arch"])
        mac = next(m for m in self.data["machine_summary"]
                   if m["machine"] == "jaedyns-macbook-pro")
        # all three variants' tokens merged: 3 x 1M input
        self.assertEqual(mac["total_tokens"], 3_000_000)
        self.assertEqual(mac["api_calls"], 3)

    def test_machine_daily_cost_merged(self):
        keys = sorted(self.data["machine_daily_cost"].keys())
        self.assertEqual(keys, ["jaedyns-macbook-pro", "ms01arch"])
        mac_series = self.data["machine_daily_cost"]["jaedyns-macbook-pro"]
        ms01_series = self.data["machine_daily_cost"]["ms01arch"]
        self.assertAlmostEqual(sum(mac_series), 3 * sum(ms01_series), places=2)

    def test_machine_last_active_merged(self):
        keys = sorted(self.data["machine_last_active"].keys())
        self.assertEqual(keys, ["jaedyns-macbook-pro", "ms01arch"])

    def test_recent_sessions_use_canonical_name(self):
        machines = {s["machine"] for s in self.data["recent_sessions"]}
        self.assertEqual(machines, {"jaedyns-macbook-pro", "ms01arch"})

    def test_today_machine_summary_merged(self):
        names = [m["machine"] for m in self.data["today"]["machine_summary"]]
        self.assertEqual(sorted(names), ["jaedyns-macbook-pro", "ms01arch"])


if __name__ == "__main__":
    unittest.main()
