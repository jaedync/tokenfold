"""Source-level regression tests for the dashboard template.

The dashboard is a single server-rendered template with inline JS; these
assertions pin the perf/robustness contracts from the UX overhaul:

* Chart.js is self-hosted (no CDN single point of failure) and deferred,
  with all chart code guarded so tables/heatmaps survive a missing library.
* Pollers and listeners are created once, not once-per-render (leaks).
* Charts update in place (no destroy/recreate animation replay on refresh).
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "templates" / "dashboard.html"


class DashboardTemplateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    # ── Chart.js self-hosting (P1-1) ──────────────────────────────────────

    def test_chartjs_is_self_hosted(self):
        """No CDN <script> for chart.js; the bundle ships from /static."""
        self.assertNotIn("cdn.jsdelivr.net", self.tpl)
        self.assertIn('defer src="/static/chart.umd.min.js"', self.tpl)

    def test_chartjs_bundle_exists(self):
        bundle = ROOT / "static" / "chart.umd.min.js"
        self.assertTrue(bundle.is_file(), "static/chart.umd.min.js missing")
        self.assertGreater(bundle.stat().st_size, 100_000)
        self.assertIn("Chart.js", bundle.read_text()[:200])

    def test_chart_usage_is_guarded(self):
        """DOM-only renders must survive a failed chart.js load."""
        self.assertIn("typeof Chart !== 'undefined'", self.tpl)
        self.assertIn("function initChartGlobals()", self.tpl)

    # ── Interval / listener leaks (P1-2) ──────────────────────────────────

    def test_rate_limits_poller_outside_render_mode(self):
        """The /api/rate-limits fetch + 60s poll lives in initRateLimits
        (run once at boot), NOT in renderMode (run on every refresh)."""
        self.assertIn("function initRateLimits()", self.tpl)
        self.assertEqual(self.tpl.count("'/api/rate-limits?scope='"), 1)
        render_start = self.tpl.index("function renderMode(")
        init_start = self.tpl.index("function initRateLimits()")
        fetch_pos = self.tpl.index("'/api/rate-limits?scope='")
        self.assertGreater(init_start, render_start,
                           "initRateLimits should be defined after renderMode")
        self.assertGreater(fetch_pos, init_start,
                           "rate-limits fetch must live inside initRateLimits")

    def test_single_global_keydown_handler(self):
        """One Escape handler for all modals — was one new listener per render."""
        self.assertEqual(self.tpl.count("document.addEventListener('keydown'"), 1)

    def test_env_cycler_interval_is_cleared(self):
        self.assertIn("_envCycleTimer", self.tpl)
        self.assertIn("clearInterval(_envCycleTimer)", self.tpl)

    def test_heatmap_tip_listeners_attached_once(self):
        """Tooltip listeners are delegated in initTips, not re-attached inside
        renderHeatmap/renderHourly on every innerHTML swap."""
        self.assertIn("function initTips()", self.tpl)
        for fn in ("renderHeatmap", "renderHourly"):
            start = self.tpl.index("function %s()" % fn)
            end = self.tpl.index("\n}", start)
            self.assertNotIn("addEventListener", self.tpl[start:end],
                             "%s must not attach listeners per render" % fn)

    # ── Chart animations / update-in-place (P1-3) ─────────────────────────

    def test_chart_animations_disabled(self):
        self.assertIn("Chart.defaults.animation = false", self.tpl)

    def test_charts_update_in_place(self):
        """upsertChart registry replaces the old destroy-everything loop."""
        self.assertIn("function upsertChart(", self.tpl)
        self.assertIn("ex.update('none')", self.tpl)
        self.assertNotIn("_tCharts", self.tpl)

    # ── Information clarity (P1-5 / P1-6 / P1-8) ──────────────────────────

    def test_cost_hero_names_its_window(self):
        """The hero figure must say which window it covers."""
        self.assertIn("costHeroLabel", self.tpl)
        self.assertIn("'Estimated API Cost · ' + windowLabel", self.tpl)
        self.assertIn("'Last 14 Days'", self.tpl)

    def test_personal_scope_api_equivalent_note(self):
        self.assertIn("costHeroNote", self.tpl)
        self.assertIn("TF_SCOPE === 'personal'", self.tpl)
        self.assertIn("not what you're billed", self.tpl)

    def test_yellow_text_token_for_light_surfaces(self):
        """Yellow #f5c518 fails contrast as text on cream; a darkened token
        must exist and the known offenders must use it."""
        self.assertIn("--yellow-text: #9a7a00", self.tpl)
        # Minutes axis on Daily Activity no longer uses raw yellow ticks
        self.assertNotIn("ticks:{color:'#f5c518'}", self.tpl)

    def test_no_sub_055rem_labels(self):
        """Floor micro-labels at 0.55rem (was 0.42-0.5rem)."""
        import re
        for m in re.finditer(r"font-size:\s*0\.(\d+)rem", self.tpl):
            size = float("0." + m.group(1))
            self.assertGreaterEqual(
                size, 0.55,
                "label below 0.55rem floor: %s" % m.group(0))

    # ── P2 batch (information architecture) ───────────────────────────────

    def test_jump_nav_present(self):
        """8,800px page needs in-page navigation (P2-22)."""
        self.assertIn('class="jump-nav"', self.tpl)
        for anchor in ("#cost", "#sessions", "#activity", "#models",
                       "#machines", "#daily", "#reference"):
            self.assertIn('href="%s"' % anchor, self.tpl)

    def test_reference_accordion_demotes_static_content(self):
        """Pricing/Benchmarks/CPH are Anthropic constants, not telemetry —
        they live in a collapsed accordion at the bottom (P2-16)."""
        self.assertIn('<details class="ref-accordion"', self.tpl)
        ref_pos = self.tpl.index('<details class="ref-accordion"')
        self.assertGreater(ref_pos, self.tpl.index('id="daily"'),
                           "Reference must sit below the Daily table")
        self.assertGreater(self.tpl.index('id="pricingBody"'), ref_pos)
        self.assertGreater(self.tpl.index('id="benchWrap"'), ref_pos)
        self.assertGreater(self.tpl.index('id="cphWrap"'), ref_pos)

    def test_section_header_renamed(self):
        """'Monthly Cost' mislabeled weekly/5h/extra gauges (P2-9)."""
        self.assertIn("<h2>Cost &amp; Limits</h2>", self.tpl)
        self.assertNotIn("<h2>Monthly Cost</h2>", self.tpl)

    def test_mode_chips_self_describe_windows(self):
        """Sections affected by the period toggle carry a window chip (P2-18)."""
        self.assertGreaterEqual(self.tpl.count("data-mode-chip"), 8)
        self.assertIn("'last 14 days'", self.tpl)

    def test_sessions_footnote_corrected(self):
        """Titles are AI summaries, not Claude-Desktop-only (P2-20)."""
        self.assertNotIn("Titled rows come from Claude Desktop", self.tpl)
        self.assertIn("Titles are AI-generated summaries", self.tpl)

    def test_toggles_are_labeled_and_pressed(self):
        """Scope and period toggles are visually distinct controls (P2-23)."""
        self.assertIn('id="scopeToggleLabel"', self.tpl)
        self.assertIn('id="modeToggleLabel"', self.tpl)
        self.assertIn("aria-pressed", self.tpl)

    def test_skip_unchanged_renders(self):
        """Live refresh must not reflash unchanged heatmap/sessions DOM (P2-24)."""
        self.assertGreaterEqual(self.tpl.count("_renderKey"), 4)

    def test_token_breakdown_cache_toggle(self):
        self.assertIn('id="tokCacheToggle"', self.tpl)
        self.assertIn("_tokIncludeCacheReads", self.tpl)

    def test_red_reserved_for_high_costs(self):
        """Ordinary costs render black; red only above the 1.5x-average
        threshold (client table rebuild mirrors the server render)."""
        self.assertIn("hotCost", self.tpl)
        self.assertNotIn("d.cost>0?'var(--red)'", self.tpl)
        self.assertNotIn("s.cost>0?'var(--red)'", self.tpl)
        self.assertNotIn("ms.cost > 0 ? 'var(--red)'", self.tpl)

    def test_install_command_button_and_clipboard_fallback(self):
        """Footer install-command button copies the curl-pipe-bash one-liner;
        with no clipboard (http/permission) it reveals the text so it can be
        copied by hand — a copy-only button must never silently no-op."""
        self.assertIn('id="installCmdBtn"', self.tpl)
        self.assertIn('data-install-cmd="{{ install_cmd|e }}"', self.tpl)
        # Scope the assertions to the install button's own script block.
        sidx = self.tpl.index("getElementById('installCmdBtn')")
        block = self.tpl[sidx:self.tpl.index("</script>", sidx)]
        self.assertIn("navigator.clipboard", block)
        self.assertIn("replaceChildren", block)  # the reveal fallback


class TableUxRegressionTest(unittest.TestCase):
    """Pins the table/animation regression fixes: sticky headers that survive
    scroll, sortable columns, state-preserving live refresh, K-notation
    number formatting, and the composed (shift-free) two-phase boot."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    # ── number formatting consistency ──────────────────────────────────────

    def test_fmt_num_k_notation(self):
        from app.dashboard import _fmt_num
        self.assertEqual(_fmt_num(492_000), "492K")
        self.assertEqual(_fmt_num(1_234), "1.2K")
        self.assertEqual(_fmt_num(26_259), "26.3K")
        self.assertEqual(_fmt_num(2_400_000), "2.4M")
        self.assertEqual(_fmt_num(1_080_000_000), "1.08B")
        self.assertEqual(_fmt_num(999), "999")
        self.assertEqual(_fmt_num(999_950), "1M")      # round-up promotion

    def test_fn_js_no_comma_thousands(self):
        """Client fN() must not fall through to toLocaleString for 1e3-1e6."""
        start = self.tpl.index("function fN(")
        end = self.tpl.index("function fT(")
        self.assertNotIn("toLocaleString", self.tpl[start:end])
        self.assertIn("'K'", self.tpl[start:end])

    # ── sticky table headers ───────────────────────────────────────────────

    def test_sticky_headers_survive_scroll(self):
        """border-collapse:collapse paints borders at the table layer, so a
        sticky thead's border scrolls away with the rows."""
        self.assertIn("border-collapse: separate", self.tpl)
        self.assertNotIn("border-collapse: collapse", self.tpl)

    def test_tooltip_th_does_not_break_sticky(self):
        """position:relative on th[data-info] overrode position:sticky and the
        $/hr header scrolled away while its siblings stuck."""
        import re
        m = re.search(r"th\[data-info\]\s*\{([^}]*)\}", self.tpl)
        self.assertIsNotNone(m)
        self.assertNotIn("position", m.group(1))

    def test_first_table_column_has_padding(self):
        self.assertNotIn("th:first-child { padding-left: 0; }", self.tpl)
        self.assertNotIn("td:first-child { padding-left: 0; }", self.tpl)

    def test_session_machine_border_on_cell_not_row(self):
        """tr borders don't paint under border-collapse:separate."""
        self.assertIn(".sess-row > td:first-child", self.tpl)
        self.assertNotIn("border-left:5px solid '+mc", self.tpl)

    # ── sortable columns ───────────────────────────────────────────────────

    def test_sortable_tables_wired(self):
        self.assertIn("function initSortableTables", self.tpl)
        self.assertIn("function applySort", self.tpl)
        self.assertIn("function sortVal", self.tpl)
        self.assertIn("aria-sort", self.tpl)
        self.assertIn("sort-ind", self.tpl)

    def test_sort_keeps_detail_rows_attached(self):
        """Detail rows (.mb-detail) must travel with their parent row group."""
        start = self.tpl.index("function applySort")
        end = self.tpl.index("function updateSortIndicators")
        self.assertIn("mb-detail", self.tpl[start:end])

    def test_sort_reapplied_after_rebuilds(self):
        """Every tbody rebuild restores the active sort."""
        self.assertGreaterEqual(self.tpl.count("afterTableRender("), 5)

    # ── state-preserving live refresh ──────────────────────────────────────

    def test_expanded_rows_keyed_by_identity(self):
        """Expansion state keys on session_id / model name (not row index) so
        re-renders restore open rows."""
        self.assertIn("_openSessions", self.tpl)
        self.assertIn("_openModels", self.tpl)
        self.assertIn("data-sid", self.tpl)
        self.assertIn("_openSessions.has(s.session_id)", self.tpl)
        self.assertIn("_openModels.has(m.model)", self.tpl)

    def test_model_and_machine_tables_skip_unchanged_renders(self):
        self.assertIn("modelBody._renderKey", self.tpl)
        self.assertIn("machineBody._renderKey", self.tpl)

    def test_tbl_wrap_scroll_preserved(self):
        self.assertGreaterEqual(self.tpl.count("scrollTop"), 6)

    # ── session detail models group ────────────────────────────────────────

    def test_session_models_render_vertically(self):
        """Models in the expanded session detail are mb-stat rows, not a
        ' · '-joined horizontal string."""
        self.assertNotIn("join(' · ') || dot", self.tpl)
        start = self.tpl.index("function renderSessions")
        end = self.tpl.index("function toggleSess")
        self.assertIn("modelRows", self.tpl[start:end])

    # ── composed load (no layout shifts) ───────────────────────────────────

    def test_two_phase_boot(self):
        """Data/DOM boot is synchronous (pre-paint); charts attach at
        DOMContentLoaded into pre-sized containers."""
        self.assertIn("function bootCharts()", self.tpl)
        self.assertIn("\nboot();", self.tpl)
        self.assertIn("document.addEventListener('DOMContentLoaded', bootCharts)", self.tpl)

    def test_oauth_gauge_space_reserved(self):
        """Personal scope reserves the OAuth gauges' height before the
        rate-limits fetch resolves (no late shift)."""
        self.assertIn("reserveLimitsSpace", self.tpl)
        self.assertIn("minHeight", self.tpl)


class ProviderUsageLimitsLayoutTest(unittest.TestCase):
    """Personal subscription gauges are visibly grouped by provider."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_claude_provider_group_wraps_existing_gauges(self):
        self.assertIn('class="usage-limit-provider" data-provider="claude"',
                      self.tpl)
        self.assertIn('<h3 class="usage-limit-provider-name">Claude</h3>',
                      self.tpl)
        self.assertIn('aria-label="Claude usage limits"', self.tpl)
        self.assertIn('class="usage-limit-provider-label">Usage limits</',
                      self.tpl)

    def test_provider_group_preserves_gauges_and_projection_cards(self):
        assignment = re.search(
            r"oauthPanel\.innerHTML\s*=\s*providerGroup\([^;]+;",
            self.tpl)
        self.assertIsNotNone(assignment, "OAuth gauges must render through the provider group")
        rendered = assignment.group(0)
        self.assertIn("gaugeHtml", rendered)
        self.assertIn("modelRowsHtml", rendered)
        self.assertIn("extraHtml", rendered)
        self.assertIn("buildGauge('Weekly Limit'", self.tpl)
        self.assertIn("buildGauge('5-Hour Window'", self.tpl)
        self.assertIn("projectedPct", self.tpl)
        self.assertIn("budgetStats", self.tpl)

    def test_provider_group_helper_is_generic_for_future_sources(self):
        self.assertIn("function providerGroup(providerKey, providerName, content)",
                      self.tpl)
        self.assertIn("data-provider=\"' + esc(providerKey) + '\"", self.tpl)

    def test_reported_provider_limits_render_without_fabricated_gauges(self):
        self.assertIn("wb.providers || {}", self.tpl)
        self.assertIn("function buildReportedProviderGroups", self.tpl)
        self.assertIn("'codex': 'Codex'", self.tpl)
        self.assertIn("'opencode-go': 'OpenCode Go'", self.tpl)
        self.assertIn("provider.windows || []", self.tpl)
        self.assertIn("if (!windows.length", self.tpl)

    def test_reported_provider_gauges_preserve_percent_and_projection(self):
        self.assertIn("window.pct", self.tpl)
        self.assertIn("window.window_seconds", self.tpl)
        self.assertIn("provider-limit-projection", self.tpl)
        self.assertIn("API-equivalent", self.tpl)


class ScopedBucketGaugesTest(unittest.TestCase):
    """B3/B4: per-model gauge rows render from oauth.buckets ('scoped:' keys)
    instead of hardcoded opus_pct/sonnet_pct fields, with deterministic colors
    and per-bucket reset times."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_no_hardcoded_per_model_pct_fields(self):
        """sonnet_pct/opus_pct are gone from the API; nothing may read them."""
        self.assertNotIn("opus_pct", self.tpl)
        self.assertNotIn("sonnet_pct", self.tpl)

    def test_scoped_rows_loop_over_oauth_buckets(self):
        """Rows come from (oauth.buckets || []) filtered to 'scoped:' keys —
        the || [] guard keeps older servers (no buckets field) error-free."""
        self.assertIn("oauth.buckets || []", self.tpl)
        self.assertIn("indexOf('scoped:')", self.tpl)

    def test_deterministic_color_map(self):
        self.assertIn("'scoped:opus': 'var(--black)'", self.tpl)
        self.assertIn("'scoped:sonnet': 'var(--blue)'", self.tpl)
        self.assertIn("'scoped:fable': 'var(--yellow)'", self.tpl)

    def test_no_red_in_scoped_palette(self):
        """var(--red) is reserved for over-pace overflow — never a bucket."""
        m = re.search(r"var SCOPED_COLORS = \{(.*?)\};", self.tpl, re.DOTALL)
        self.assertIsNotNone(m, "SCOPED_COLORS map missing")
        self.assertNotIn("--red", m.group(1))
        m2 = re.search(r"var SCOPED_FALLBACK = \[(.*?)\];", self.tpl)
        self.assertIsNotNone(m2, "SCOPED_FALLBACK palette missing")
        self.assertNotIn("--red", m2.group(1))

    def test_scoped_row_shows_reset_time(self):
        """NEW vs the old rows: per-bucket resets_at via fmtReset."""
        self.assertIn("fmtReset(sb.resets_at)", self.tpl)

    def test_scoped_label_is_escaped(self):
        """display_name is external data — must flow through esc()."""
        self.assertIn("esc(sb.label)", self.tpl)

    def test_yellow_pct_text_uses_readable_token(self):
        """Yellow fill needs the darkened --yellow-text token for pct text.
        Scoped rows now render through buildGauge (fillColor: sbColor), so
        the mapping happens inside buildGauge: textColorFor(color) where
        color is the fillColor override."""
        self.assertIn("fillColor: sbColor", self.tpl)
        self.assertIn("var color = opts.fillColor || barColor(pct, expectedPct);",
                      self.tpl)
        self.assertIn("textColorFor(color)", self.tpl)

    def test_pct_field_name_pinned(self):
        """The renderer reads sb.pct — the /api/rate-limits buckets[] entries
        emit 'pct', and renaming that field server-side would silently render
        0% gauges (same string-pin style as the oauth.buckets check above)."""
        self.assertIn("sb.pct", self.tpl)

    def test_fallback_palette_disjoint_from_fixed_map(self):
        """Unknown buckets must never collide with a known model's color:
        SCOPED_FALLBACK draws only tokens absent from SCOPED_COLORS, and its
        index advances only for unknown buckets (fallbackIdx, not sbi)."""
        m = re.search(r"var SCOPED_COLORS = \{(.*?)\};", self.tpl, re.DOTALL)
        self.assertIsNotNone(m, "SCOPED_COLORS map missing")
        m2 = re.search(r"var SCOPED_FALLBACK = \[(.*?)\];", self.tpl)
        self.assertIsNotNone(m2, "SCOPED_FALLBACK palette missing")
        fixed = set(re.findall(r"var\(--[\w-]+\)", m.group(1)))
        fallback = re.findall(r"var\(--[\w-]+\)", m2.group(1))
        self.assertTrue(fallback, "fallback palette must not be empty")
        self.assertFalse(
            fixed & set(fallback),
            f"fallback palette reuses fixed-map colors: {fixed & set(fallback)}")
        self.assertIn("fallbackIdx", self.tpl)
        self.assertNotIn("SCOPED_FALLBACK[sbi", self.tpl)


class TrendGaugeNoteTest(unittest.TestCase):
    """D3/C4: sub-window burn/ETA/pace verdict + reset annotation on the
    Weekly and 5-Hour gauges, built from oauth.trend[bucketKey] (D2)."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_reads_oauth_trend(self):
        self.assertIn("oauth.trend", self.tpl)

    def test_fmt_relative_helper_exists(self):
        """Tiny relative-time helper lives near fmtReset, seconds-epoch in."""
        self.assertIn("function fmtRelative(epochSeconds)", self.tpl)
        fmt_reset_pos = self.tpl.index("function fmtReset(")
        fmt_rel_pos = self.tpl.index("function fmtRelative(")
        self.assertGreater(fmt_rel_pos, fmt_reset_pos,
                            "fmtRelative should sit near/after fmtReset")

    def test_build_trend_note_reads_burn_and_eta_fields(self):
        self.assertIn("function buildTrendNote(", self.tpl)
        self.assertIn("burn_6h_pct_per_hr", self.tpl)
        self.assertIn("burn_1h_pct_per_hr", self.tpl)
        self.assertIn("eta_100_epoch", self.tpl)

    def test_burn_labels_distinguish_window(self):
        """Weekly reads the 6h burn, 5-Hour reads the 1h burn — each call
        site labels which sub-window the rate covers."""
        self.assertIn("'burn_6h_pct_per_hr', '(6h)'", self.tpl)
        self.assertIn("'burn_1h_pct_per_hr', '(1h)'", self.tpl)

    def test_pace_verdict_reuses_existing_over_pace_style(self):
        """'over' pace must reuse the SAME .proj-warn class the whole-window
        projection already uses — never a new color, never var(--red) as a
        fill (that token is reserved for the over-pace overflow bar)."""
        self.assertIn('<strong class="proj-warn">over pace</strong>', self.tpl)
        self.assertIn("'on pace'", self.tpl)  # new verdict state, wasn't in the old code

    def test_whole_window_projection_relabeled(self):
        """The existing whole-window linear projection is now explicitly
        labeled so it can't be confused with the new sub-window burn note."""
        self.assertIn("(window avg)", self.tpl)

    def test_reset_annotation_uses_current_window_boundaries(self):
        """C4: a reset only annotates the gauge when it falls inside THAT
        bucket's current window (7d for weekly, 5h for five_hour)."""
        self.assertIn("nowSeconds - windowSeconds", self.tpl)
        self.assertIn("buildTrendNote(trend.seven_day, 'burn_6h_pct_per_hr', '(6h)', 7 * 86400)", self.tpl)
        self.assertIn("buildTrendNote(trend.five_hour, 'burn_1h_pct_per_hr', '(1h)', 5 * 3600)", self.tpl)

    def test_trend_note_wired_into_both_gauges(self):
        self.assertIn("trendNote: weeklyTrendNote", self.tpl)
        self.assertIn("trendNote: fiveHourTrendNote", self.tpl)
        self.assertIn("opts.trendNote", self.tpl)

    def test_missing_trend_renders_nothing_extra(self):
        """buildTrendNote returns null on every guarded miss, so a payload
        with no oauth.trend renders the gauge exactly as before."""
        start = self.tpl.index("function buildTrendNote(")
        end = self.tpl.index("function buildGauge(")
        block = self.tpl[start:end]
        self.assertIn("if(!trendEntry) return null;", block)

    def test_past_reset_annotated_with_ago_phrasing(self):
        """Fix 3: resets[] events are always historical (at_epoch is a past
        reading's fetched_epoch) — the gauge must annotate them with past
        'reset <N> ago' phrasing, not fmtReset's future-tense
        'resets <day> <time>' wording (wrong for something that already
        happened)."""
        self.assertIn("function fmtAgo(epochSeconds)", self.tpl)
        fmt_rel_pos = self.tpl.index("function fmtRelative(")
        fmt_ago_pos = self.tpl.index("function fmtAgo(")
        self.assertGreater(fmt_ago_pos, fmt_rel_pos,
                            "fmtAgo should sit near/after fmtRelative")
        start = self.tpl.index("function buildTrendNote(")
        end = self.tpl.index("function buildGauge(")
        block = self.tpl[start:end]
        self.assertIn("fmtAgo(rv.at_epoch)", block)
        self.assertIn("'reset ' + agoTxt", block)
        # Negative pin: the old future-tense call on a past epoch is gone.
        self.assertNotIn("fmtReset(new Date(rv.at_epoch", block)


class FiveHourExpiredGaugeTest(unittest.TestCase):
    """D6: a stale (already-past) 5-hour resets_at must not pin an 'expected'
    marker at 100% against old utilization — bar coloring falls back to
    absolute-pct bands (buildGauge/barColor already handle a null expected)."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_expired_five_hour_reset_nulls_expected(self):
        self.assertIn("hResetMs <= Date.now()) hExpected = null;", self.tpl)

    def test_guard_sits_right_after_hexpected_is_computed(self):
        pos_h = self.tpl.index(
            "var hExpected = calcExpectedPct(oauth.five_hour_resets_at, FIVE_HR_MS);")
        pos_guard = self.tpl.index("hResetMs <= Date.now()")
        pos_daylines = self.tpl.index("var weeklyDayLines = [];")
        self.assertGreater(pos_guard, pos_h)
        self.assertGreater(pos_daylines, pos_guard,
                            "D6 guard should run before the rest of gauge math")


class BudgetLeftWindowFixTest(unittest.TestCase):
    """D5: the 'budget left' estimate must divide ONE consistent window's
    cost/active-time by that SAME window's pct — the old code mixed a
    rolling-7d cost with a limit-window pct and blew up right after a reset."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_uses_limit_window_inputs(self):
        self.assertIn("oauth.limit_window", self.tpl)
        self.assertIn("limitWindow.cost / wPct", self.tpl)
        self.assertIn("lwActiveMin / wPct", self.tpl)

    def test_old_mixed_window_division_removed(self):
        """NEGATIVE pin: the removed bug divided rolling-7d wb.week_cost /
        wb.week_active_s by the limit-window wPct — those exact expressions
        must never reappear."""
        self.assertNotIn("weekCost / wPct", self.tpl)
        self.assertNotIn("weekActiveMin / wPct", self.tpl)

    def test_suppressed_when_limit_window_absent(self):
        start = self.tpl.index("var limitWindow = oauth.limit_window || null;")
        end = self.tpl.index("var weeklyNote = null;")
        block = self.tpl[start:end]
        self.assertIn("if(limitWindow && wExpected >= 2 && !recentWeeklyReset", block)

    def test_suppressed_on_recent_reset(self):
        """A reset in the trailing 24h suppresses the estimate even if
        limit_window and wExpected both look fine."""
        self.assertIn("recentWeeklyReset", self.tpl)
        self.assertIn("nowSeconds - 86400", self.tpl)

    def test_gate_and_render_use_same_rounded_active_minutes(self):
        """Fix 7: the D5 gate used to check limitWindow.active_s > 0 (raw
        seconds) while the render checked lwActiveMin > 0 (rounded minutes,
        computed only inside the block) — with 0 < active_s < 30 the gate
        passed but Math.round(active_s/60) rounded down to 0, so the pushed
        stat had an empty value. Gate and render must use the SAME
        already-rounded lwActiveMin. (F2 hoisted both declarations above
        the stats rows, so the search window starts at the declaration.)"""
        start = self.tpl.index("var limitWindow = oauth.limit_window || null;")
        end = self.tpl.index("var weeklyNote = null;")
        block = self.tpl[start:end]
        # lwActiveMin is computed before the gate, not only inside the block.
        gate_pos = block.index("if(limitWindow && wExpected >= 2")
        lw_active_min_pos = block.index("var lwActiveMin =")
        self.assertLess(lw_active_min_pos, gate_pos,
                         "lwActiveMin must be computed before the gate")
        self.assertIn("(limitWindow.cost > 0 || lwActiveMin > 0))", block)
        # Negative pin: the gate no longer compares the raw seconds field.
        self.assertNotIn("limitWindow.active_s > 0)", block)


class WindowAnchoredSpendRowsTest(unittest.TestCase):
    """F2: the weekly subpanel's spend/active rows are anchored to the
    CURRENT limit window (oauth.limit_window), which resets when the window
    does — natural rollover or granted reset alike. The never-resetting
    rolling-7d pair survives only as the fallback when limit_window is
    absent, with its honest label."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_window_rows_primary_rolling_fallback(self):
        self.assertIn("'spent · this window'", self.tpl)
        self.assertIn("'active · this window'", self.tpl)
        # Fallback path (and only there) keeps the rolling labels.
        self.assertIn("'spent · rolling 7d'", self.tpl)
        self.assertIn("'active · rolling 7d'", self.tpl)
        # Primary rows read limit_window fields, not wb.week_cost.
        self.assertIn("fC(limitWindow.cost || 0)", self.tpl)

    def test_rows_are_conditional_on_limit_window(self):
        start = self.tpl.index("var limitWindow = oauth.limit_window || null;")
        end = self.tpl.index("'spent · rolling 7d'")
        block = self.tpl[start:end]
        self.assertIn("if(limitWindow) {", block)
        self.assertIn("'spent · this window'", block)

    def test_carry_over_note_gated_on_fallback_path(self):
        """The 'stats below carry over' apology only makes sense when the
        rolling-7d fallback is showing — window-anchored stats reset WITH
        the window."""
        self.assertIn(
            "if (!limitWindow && wPct < 1 && (weekCost > 0 || weekActiveMin > 0))",
            self.tpl)


class PaceChartTrendOverlayTest(unittest.TestCase):
    """D4: utilization overlay + even-drain reference + reset markers on the
    pace-chart modal, gated on oauth.trend.seven_day.series and byte-identical
    to the cost-only chart when that series is absent."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_gated_on_weekly_series(self):
        self.assertIn("var trend = wb.oauth && wb.oauth.trend;", self.tpl)
        self.assertIn(
            "var weeklySeries = trend && trend.seven_day && trend.seven_day.series;",
            self.tpl)
        self.assertIn("if(weeklySeries && weeklySeries.length) {", self.tpl)

    def test_series_epochs_scaled_to_milliseconds_for_chart(self):
        """Chart/Date math needs milliseconds; series epochs arrive in
        seconds per the /api/rate-limits contract, so the mapping must scale
        them up (matches the weekStartEpoch*1000 convention already used
        elsewhere in this same function)."""
        self.assertIn("var sampleEpochMs = weeklySeries[usi][0] * 1000;", self.tpl)

    def test_utilization_dataset_on_new_right_axis(self):
        self.assertIn("label: 'Utilization %'", self.tpl)
        self.assertIn("yAxisID: 'y2'", self.tpl)
        self.assertIn("scalesCfg.y2 = {", self.tpl)
        self.assertIn("min: 0,\n            max: 100,", self.tpl)

    def test_even_drain_reference_line_gated_on_limit_window(self):
        start = self.tpl.index("if(weeklySeries && weeklySeries.length) {")
        end = self.tpl.index("// Working hours background shading")
        block = self.tpl[start:end]
        self.assertIn("if(limitWindow) {", block)
        self.assertIn("label: 'Even drain'", block)
        self.assertIn("borderDash: [3, 3]", block)
        # (b) no tooltip for the reference line
        self.assertIn(
            "filter: function(item) { return item.dataset.label !== 'Even drain'; }",
            self.tpl)

    def test_even_drain_diagonal_clipped_to_chart_range(self):
        """Fix 1: evenEndIdx = evenStartIdx + 168 exceeds totalHours whenever
        the weekly reset is in the future (resets_at up to 7d out is the
        common case) — the old code just dropped that endpoint, leaving a
        single-point dataset that Chart.js never draws a line for. The fix
        clips both endpoints to the visible [0, totalHours] range instead of
        dropping them, so the dataset keeps exactly two points."""
        start = self.tpl.index("if(limitWindow) {",
                                self.tpl.index("Even-drain reference diagonal"))
        end = self.tpl.index("extraDatasets.push({", start)
        block = self.tpl[start:end]
        # The clipped-endpoint value expressions (distinctive, not just the
        # boundary check) — pins the actual clip math, not merely its guard.
        self.assertIn("(totalHours - evenStartIdx) / 168", block)
        self.assertIn("(0 - evenStartIdx) / 168", block)
        # Both endpoints must still be assignable when in-range (the old
        # unconditional-drop behavior must not survive as dead code).
        self.assertIn("evenData[totalHours] =", block)
        self.assertIn("evenData[0] =", block)
        self.assertIn("evenData[evenEndIdx] = 100", block)
        self.assertIn("evenData[evenStartIdx] = 0", block)
        # A comment must explain the clip (per the fix requirement).
        self.assertIn("Clip the diagonal to the visible", block)

    def test_reset_markers_extend_the_now_line_plugin_pattern(self):
        """(d): follow the NOW-line plugin's own afterDraw + getPixelForValue
        + setLineDash pattern rather than inventing a new drawing mechanism."""
        start = self.tpl.index("afterDraw: function(chart) {")
        end = self.tpl.index("ctx.fillText('NOW'")
        block = self.tpl[start:end]
        self.assertIn("resetMarkers", block)
        self.assertIn("xAxis.getPixelForValue(resetMarkers[rmk])", block)
        self.assertIn("ctx.fillText('reset', rx", block)

    def test_now_line_plugin_drawing_unchanged(self):
        """The NOW-line drawing commands themselves are untouched."""
        self.assertIn("ctx.strokeStyle = 'rgba(230, 51, 41, 0.8)';", self.tpl)
        self.assertIn("ctx.fillText('NOW', x, yAxis.top - 4);", self.tpl)

    def test_base_dataset_array_extended_not_replaced(self):
        """extraDatasets starts empty — a series-absent response concats []
        onto the original single-dataset array, reproducing it exactly."""
        self.assertIn("var extraDatasets = [];", self.tpl)
        self.assertIn("].concat(extraDatasets);", self.tpl)


class DashboardServerRenderTest(unittest.TestCase):
    """Server-rendered daily rows follow the same red-is-a-warning rule."""

    def test_cost_color_threshold(self):
        from app.dashboard import _fmt_cost  # noqa: F401 (import sanity)
        import app.dashboard as dash
        # replicate the inline logic deterministically via a tiny harness:
        daily = [{"cost": c} for c in (5.0, 5.0, 5.0, 50.0, 0.0)]
        nz = [d["cost"] for d in daily if d["cost"] > 0]
        hot = sum(nz) / len(nz) * 1.5
        self.assertGreater(50.0, hot)   # the outlier day goes red
        self.assertLess(5.0, hot)       # ordinary days stay black
        src = (TEMPLATE.parent.parent / "app" / "dashboard.py").read_text()
        self.assertIn("_hot_cost", src)
        self.assertNotIn("'var(--red)' if d[\"cost\"] > 0", src)


class SpendHistorySectionTest(unittest.TestCase):
    """New 'Spend History' section: two Chart.js cards (limit windows +
    monthly spend) fed by a single boot-time fetch of /api/spend-history,
    with no polling and no error banner on failure (secondary analytics)."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_section_markup_present_after_rate_limits(self):
        self.assertIn('id="spendHistorySection"', self.tpl)
        self.assertIn('id="spendHistoryWindowsCard"', self.tpl)
        self.assertIn('id="spendHistoryMonthlyCard"', self.tpl)
        self.assertGreater(
            self.tpl.index('id="spendHistorySection"'),
            self.tpl.index('id="rateLimitsSection"'),
            "Spend History must be inserted after Cost & Limits")

    def test_fetch_url_uses_scope_global(self):
        self.assertIn(
            "'/api/spend-history?scope=' + encodeURIComponent(TF_SCOPE)",
            self.tpl)

    def test_windows_card_gated_on_windows_key(self):
        self.assertIn("if(data.windows && data.windows.length){", self.tpl)
        self.assertIn("if(data.months && data.months.length){", self.tpl)

    def test_no_polling_single_fetch(self):
        """History moves slowly — exactly one fetch call, and it must not
        live inside (or near) any setInterval callback."""
        self.assertEqual(self.tpl.count("/api/spend-history?scope="), 1)
        for m in re.finditer(r"setInterval\(", self.tpl):
            window = self.tpl[m.start():m.start() + 400]
            self.assertNotIn("spend-history", window)

    def test_new_functions_are_declarations_not_arrows(self):
        for fn in (
            "function maybeRenderSpendHistory(){",
            "function renderSpendWindowsChart(windows){",
            "function renderSpendMonthlyChart(months){",
            "function spShortDate(epochSeconds){",
            "function spAlpha(hex, alpha){",
        ):
            self.assertIn(fn, self.tpl)

    def test_section_hidden_until_data_shows_something(self):
        """Default-hidden section only unhides once at least one card has
        data — this also covers the fetch-error case (stays hidden, no
        banner) since the section is never touched on the .catch() path."""
        self.assertIn(
            'id="spendHistorySection" style="border-bottom:var(--border);display:none"',
            self.tpl)
        self.assertIn("document.getElementById('spendHistorySection').style.display = 'block';", self.tpl)
        catch_pos = self.tpl.index(".catch(function(){ /* secondary analytics")
        self.assertNotIn("spendHistorySection", self.tpl[catch_pos:catch_pos + 120])


class BudgetProjectionGaugesTest(unittest.TestCase):
    """Per-gauge dollar-budget projection: shared budgetStats helper (spent /
    est. window budget / budget left) wired into the 5-hour gauge and the
    scoped per-model rows, plus the weekly est-budget cell; all budget
    extrapolation gated behind pct >= 5 (coarse-percent noise floor)."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_budget_stats_helper_present_and_gated(self):
        self.assertIn("function budgetStats(cost, pct) {", self.tpl)
        start = self.tpl.index("function budgetStats(cost, pct) {")
        block = self.tpl[start:start + 1400]
        # Missing/zero cost -> [] (older servers, empty windows).
        self.assertIn("if(typeof cost !== 'number' || cost <= 0) return stats;",
                      block)
        # Extrapolated cells only past the 5% noise gate.
        self.assertIn("pct >= 5", block)
        self.assertIn("'est. window budget'", block)
        self.assertIn("'% budget left'", block)

    def test_five_hour_gauge_gets_stats_and_projection(self):
        start = self.tpl.index("var fhWindow = oauth.five_hour_window || null;")
        end = self.tpl.index("gaugeHtml += '</div>'")
        block = self.tpl[start:end]
        self.assertIn("budgetStats(fhWindow ? fhWindow.cost : null, hPct)",
                      block)
        self.assertIn("fiveHourProjected = (hPct / hExpected) * 100", block)
        # Both are wired into the 5-Hour buildGauge call.
        call = self.tpl.index("buildGauge('5-Hour Window'")
        call_block = self.tpl[call:call + 300]
        self.assertIn("projectedPct: fiveHourProjected", call_block)
        self.assertIn("stats: fiveHourStats", call_block)

    def test_scoped_rows_use_build_gauge_with_identity_color(self):
        start = self.tpl.index("var scopedBuckets = ")
        end = self.tpl.index("// Extra usage block")
        block = self.tpl[start:end]
        self.assertIn("fillColor: sbColor", block)
        self.assertIn("budgetStats(sb.window_cost, sbPct)", block)
        self.assertIn("calcExpectedPct(sb.resets_at, WEEK_MS)", block)
        # D6 analogue: stale resets_at nulls the marker/projection.
        self.assertIn("if(!isNaN(sbResetMs) && sbResetMs <= Date.now()) sbExpected = null;",
                      block)
        # The hand-rolled bare-bar markup is gone from the scoped loop.
        self.assertNotIn("'<div class=\"rate-gauge-track\">'", block)

    def test_build_gauge_fill_color_override(self):
        self.assertIn("var color = opts.fillColor || barColor(pct, expectedPct);",
                      self.tpl)

    def test_weekly_est_budget_cell_gated(self):
        start = self.tpl.index("var limitWindow = oauth.limit_window || null;")
        block = self.tpl[start:start + 1600]
        self.assertIn("if(limitWindow.cost > 0 && wPct >= 5)", block)
        self.assertIn("(limitWindow.cost / wPct) * 100", block)


class MarkerLabelOverlapTest(unittest.TestCase):
    """The 'expected' marker label near 0%/100% overprints the gauge title or
    reset text; a measured post-render pass hides any label whose rect
    intersects neighbor text (the marker LINE always stays)."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_overlap_pass_runs_after_panel_render(self):
        # The pass body is hoisted (shared with the enterprise monthly card);
        # the oauth path CALLS it right after writing the panel HTML.
        render = self.tpl.index(
            "oauthPanel.innerHTML = providerGroup(")
        window = self.tpl[render:render + 400]
        self.assertIn("runMarkerOverlapPass();", window)
        # The pass body itself still measures + hides colliding labels.
        body_idx = self.tpl.index("function runMarkerOverlapPass() {")
        body = self.tpl[body_idx:body_idx + 2000]
        self.assertIn(".rate-gauge-marker-label", body)
        self.assertIn("getBoundingClientRect", body)
        self.assertIn("lab.style.display = 'none';", body)

    def test_overlap_pass_reruns_on_resize_without_leaking(self):
        """Collisions depend on viewport width, so the pass re-arms on
        resize — but exactly once globally (the render block runs per 60s
        poll; a listener per poll leaks), and each pass re-decides from
        scratch (display reset before measuring)."""
        self.assertIn("function runMarkerOverlapPass() {", self.tpl)
        # Arming lives in a dedicated once-guarded helper (shared by the oauth
        # gauges and the enterprise monthly card, same panel). Guard set once.
        self.assertEqual(self.tpl.count("window._tfMarkerResizeArmed = true"), 1)
        self.assertIn("function armMarkerResizeOnce() {", self.tpl)
        idx = self.tpl.index("function runMarkerOverlapPass() {")
        block = self.tpl[idx:idx + 3200]
        self.assertIn("lab.style.display = '';", block)
        # Debounced, not raw-per-resize-event.
        self.assertIn("setTimeout(runMarkerOverlapPass, 150)", block)

    def test_overlap_checks_all_neighbor_text(self):
        idx = self.tpl.index("var others = gauge.querySelectorAll(")
        sel = self.tpl[idx:idx + 220]
        for cls_ in (".rate-gauge-label", ".rate-gauge-reset",
                     ".rate-gauge-dayline-label", ".rate-gauge-chart-btn"):
            self.assertIn(cls_, sel)

    # ── EventSource live refresh, poll as fallback (Task 3) ───────────────
    def test_eventsource_stream_is_primary(self):
        """SSE drives live updates; endpoint is the same-origin stream."""
        self.assertIn("new EventSource('/api/stats/stream')", self.tpl)
        self.assertIn("function startStream()", self.tpl)

    def test_eventsource_capability_guard(self):
        """No EventSource support falls back to polling, never throws."""
        self.assertIn("typeof EventSource === 'undefined'", self.tpl)

    def test_stop_stream_closes_and_clears_retry_timer(self):
        """stopStream must release both the instance and the reconnect timer
        so repeated onerror can't stack timers or leak connections."""
        self.assertIn("function stopStream()", self.tpl)
        self.assertIn("const STREAM_RETRY_MS = 60000", self.tpl)
        idx = self.tpl.index("function stopStream()")
        block = self.tpl[idx:idx + 400]
        self.assertIn(".close()", block)
        self.assertIn("clearTimeout", block)

    def test_visibilitychange_wires_stream(self):
        """Tab hide/show must manage the stream, not just the poll."""
        self.assertIn("stopStream()", self.tpl)
        idx = self.tpl.index("visibilitychange")
        block = self.tpl[idx:idx + 400]
        self.assertIn("stopStream", block)
        self.assertIn("startStream", block)

    def test_polling_retained_as_fallback(self):
        """The 30s version poll must remain as the SSE fallback."""
        self.assertIn("function startPolling()", self.tpl)
        self.assertIn("const POLL_INTERVAL = 30000", self.tpl)

    def test_fetch_and_apply_guards_hidden_tab(self):
        """fetchAndApply must bail on a hidden tab like checkForUpdate does —
        the SSE onmessage handler calls it without a document.hidden guard
        (latent today: hidden → stopStream, but a race could still fire it)."""
        start = self.tpl.index("function fetchAndApply(")
        end = self.tpl.index("function checkForUpdate(")
        block = self.tpl[start:end]
        self.assertIn("document.hidden", block,
                      "fetchAndApply must guard against a hidden tab")


if __name__ == "__main__":
    unittest.main()
