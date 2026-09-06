// Browser regressions for quota refresh presentation. Loads the rendered
// dashboard in headless Chromium with every request answered from in-memory
// fixtures (page.route); no server, network, or production data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.TOKENFOLD_PLAYWRIGHT || 'playwright');

const ROOT = path.resolve(__dirname, '..', '..');
const ORIGIN = 'http://tokenfold.test';
const HOUR = 3600, DAY = 86400;
const HTML = {
  personal: fs.readFileSync(process.env.TOKENFOLD_HTML_PERSONAL, 'utf8'),
  enterprise: fs.readFileSync(process.env.TOKENFOLD_HTML_ENTERPRISE, 'utf8'),
};
const embedded = html => JSON.parse(html.split('id="tf-data">')[1].split('</script>')[0]);
const iso = epoch => new Date(epoch * 1000).toISOString();
const failures = [];
async function check(name, action) {
  try { await action(); console.log('ok - ' + name); }
  catch (error) { failures.push(name + ': ' + error.message); console.log('FAIL - ' + name + '\n  ' + error.message); }
}

// Mutable fixture world; every route reads it at request time.
function world(now) {
  const w = {
    now, oauthPresent: true, claudeAt: now - 120, claudeSource: null,
    claude: {weekly: 16, five: 82, fable: 31},
    weeklyReset: now + 3 * DAY, fiveReset: now + 2 * HOUR, fableReset: now + 3 * DAY,
    claudeCost: {limit: 163.15, active: 54240, five: 40.5, fable: 52.1}, trend: true,
    codexPresent: true, codexAt: now - 60, codexPct: 12, codexReset: now + 4 * DAY,
    codexCost: 28.9, codexMonth: 235.73, zenMonth: 2.5,
    weekCost: 171.4, weekActive: 54240, hourly: [{h: 100, c: 40}, {h: 150, c: 60}],
    monthlyBudget: null, extraUsage: null, snapshotStatus: 200, limitsStatus: 200, fullOverride: null,
  };
  w.oauth = enriched => {
    const oauth = {
      weekly_pct: w.claude.weekly, weekly_resets_at: iso(w.weeklyReset),
      five_hour_pct: w.claude.five, five_hour_resets_at: iso(w.fiveReset),
      buckets: [
        {key: 'five_hour', label: '5-Hour', pct: w.claude.five, resets_at: iso(w.fiveReset)},
        {key: 'seven_day', label: '7-Day', pct: w.claude.weekly, resets_at: iso(w.weeklyReset)},
        {key: 'scoped:fable', label: 'Fable', pct: w.claude.fable, resets_at: iso(w.fableReset)},
      ],
      extra_usage: w.extraUsage, updated_at: iso(w.claudeAt), updated_at_epoch: w.claudeAt,
    };
    if (w.claudeSource) oauth.source = w.claudeSource;
    if (!enriched) return oauth;
    // Baseline server shape: limit_window always present, five_hour_window
    // only while its reset is in the future, scoped window_cost likewise.
    oauth.limit_window = {start_epoch: w.weeklyReset - 7 * DAY, cost: w.claudeCost.limit, active_s: w.claudeCost.active};
    if (w.fiveReset > Date.now() / 1000) oauth.five_hour_window = {start_epoch: w.fiveReset - 5 * HOUR, cost: w.claudeCost.five};
    if (w.fableReset > Date.now() / 1000) oauth.buckets[2].window_cost = w.claudeCost.fable;
    if (w.trend) oauth.trend = {seven_day: {series: [[w.now - 6 * DAY, 2], [w.now - 3 * DAY, 9], [w.now - HOUR, w.claude.weekly]], resets: []}};
    return oauth;
  };
  w.providers = enriched => {
    const providers = {};
    if (w.codexPresent) {
      const win = {key: 'primary', label: '7-day limit', pct: w.codexPct, resets_at: iso(w.codexReset), window_seconds: 604800};
      if (enriched && w.codexReset > Date.now() / 1000) {
        win.window_cost = w.codexCost;
        win.window_end_epoch = Math.floor(w.codexAt / 60) * 60;
      }
      providers.codex = {updated_at_epoch: w.codexAt, plan: 'plus', windows: [win], ...(enriched ? {month_cost: w.codexMonth} : {})};
    }
    if (enriched && w.zenMonth > 0) providers['opencode-zen'] = {windows: [], month_cost: w.zenMonth};
    return providers;
  };
  w.snapshot = () => ({weekly_budget: {providers: w.providers(false), ...(w.oauthPresent ? {oauth: w.oauth(false)} : {})}});
  w.full = () => w.fullOverride ? w.fullOverride() : ({weekly_budget: {
    source: 'events', window: 'rolling_7d', week_cost: w.weekCost, week_active_s: w.weekActive,
    hourly_costs: w.hourly.slice(), updated_at_epoch: Date.now() / 1000,
    ...(w.monthlyBudget ? {monthly_budget: w.monthlyBudget} : {}),
    ...(w.oauthPresent ? {oauth: w.oauth(true)} : {}), providers: w.providers(true)}});
  return w;
}

async function open(browser, scope, w, viewport, clock = false) {
  const context = await browser.newContext({viewport: viewport || {width: 1440, height: 1000}});
  const page = await context.newPage();
  if (clock) await page.clock.install({time: new Date(w.now * 1000)});
  const hits = {snapshot: 0, limits: 0, errors: []};
  page.on('pageerror', error => hits.errors.push(error.message));
  await page.addInitScript(() => {
    window.__unavailable = false; window.__heights = [];
    window.EventSource = class { constructor() {} close() {} };
    new MutationObserver(() => {
      const panel = document.getElementById('oauthGaugesPanel');
      if (!panel) return;
      if (/unavailable/.test(panel.textContent)) window.__unavailable = true;
      window.__heights.push(panel.getBoundingClientRect().height);
    }).observe(document, {childList: true, subtree: true, characterData: true, attributes: true});
  });
  const data = embedded(HTML[scope]);
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.origin !== ORIGIN) return route.fulfill({status: 204, body: ''});
    const p = url.pathname;
    if (p === '/') return route.fulfill({contentType: 'text/html', body: HTML[scope]});
    if (p.startsWith('/static/')) return route.fulfill({path: path.join(ROOT, p)});
    if (p === '/api/rate-limit-snapshots') {
      hits.snapshot++;
      const status = w.snapshotFailFirst && hits.snapshot === 1 ? 500 : w.snapshotStatus;
      return status === 200 ? route.fulfill({json: w.snapshot()}) : route.fulfill({status, body: 'down'});
    }
    if (p === '/api/rate-limits') {
      hits.limits++;
      return w.limitsStatus === 200 ? route.fulfill({json: w.full()}) : route.fulfill({status: w.limitsStatus, body: 'down'});
    }
    if (p === '/api/stats/version') return route.fulfill({json: {version: data.version}});
    if (p === '/api/stats') return route.fulfill({json: data});
    if (p === '/api/spend-history') return route.fulfill({json: {months: []}});
    if (p.startsWith('/api/served-models')) return route.fulfill({json: {models: []}});
    return route.fulfill({status: 404, body: ''});
  });
  await page.goto(ORIGIN + '/?scope=' + scope, {waitUntil: 'domcontentloaded'});
  return {page, context, hits, close: () => context.close()};
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate, label) {
  for (let i = 0; i < 250; i++) { if (predicate()) return; await sleep(20); }
  throw new Error('timeout waiting for ' + label);
}
// Drive named feeds now and wait for their fixture requests to complete.
async function drive(page, hits, feeds) {
  const before = {snapshot: hits.snapshot, limits: hits.limits};
  await page.evaluate(names => names.forEach(name => tfRefresh.request(name, true)),
    feeds.map(f => f === 'snapshot' ? 'limitSnapshots' : 'limits'));
  for (const f of feeds) await until(() => hits[f] > before[f], f + ' request');
  await page.waitForTimeout(150);
}
async function heightsReset(page) { await page.evaluate(() => { window.__heights = []; }); }
async function minHeight(page) { return page.evaluate(() => Math.min(...window.__heights, Infinity)); }

async function snap(page) {
  return page.evaluate(() => {
    const panel = document.getElementById('oauthGaugesPanel');
    const provider = key => {
      const node = panel.querySelector('[data-provider="' + key + '"]');
      if (!node) return null;
      const gauges = Array.from(node.querySelectorAll('.rate-gauge'));
      return {
        height: node.getBoundingClientRect().height,
        status: (node.querySelector('.usage-limit-provider-status') || {textContent: ''}).textContent,
        stats: Array.from(node.querySelectorAll('.rate-gauge-stat')).map(s => s.textContent.trim()),
        labels: Array.from(node.querySelectorAll('.rate-gauge-stat-label')).map(s => s.textContent.trim()),
        pcts: Array.from(node.querySelectorAll('.rate-gauge-pct')).map(s => s.textContent),
        remaining: Array.from(node.querySelectorAll('.rate-gauge-remaining')).map(s => s.textContent),
        detailStates: Array.from(node.querySelectorAll('.rate-gauge-stats[data-detail-state]')).map(s => s.dataset.detailState),
        windowStates: gauges.map(g => g.dataset.windowState || ''),
        markers: node.querySelectorAll('.rate-gauge-marker').length,
        text: node.textContent,
      };
    };
    const header = document.getElementById('rateLimitsUpdated');
    return {
      height: panel.getBoundingClientRect().height, text: panel.textContent,
      header: header.textContent, headerStale: header.classList.contains('rate-limits-updated--stale'),
      claude: provider('claude'), codex: provider('codex'), zen: provider('opencode-zen'),
      active: document.activeElement && document.activeElement.id,
      banner: document.getElementById('refreshIndicator').textContent,
      scrollWidth: document.documentElement.scrollWidth, viewport: innerWidth,
    };
  });
}

async function personalScenarios(browser, viewport) {
  const tag = viewport.width + 'px';
  const now = Date.now() / 1000;
  const w = world(now);
  const {page, hits, close} = await open(browser, 'personal', w, viewport);
  try {
    await page.waitForFunction(() => document.querySelector('[data-provider="codex"] .rate-gauge-stat'));
    await page.waitForTimeout(150);
    const base = await snap(page);
    await page.evaluate(() => {
      window.__claude = document.querySelector('[data-provider="claude"]');
      window.__codex = document.querySelector('[data-provider="codex"]');
      window.__trigger = document.getElementById('oauthChartTrigger');
    });
    await check(tag + ' first paint is coherent and dated per provider', () => {
      assert.match(base.claude.status, /observed (just now|\d+m ago)/);
      assert.match(base.codex.status, /observed (just now|\d+m ago)/);
      assert(base.claude.stats.some(s => /~\$1,019\.69/.test(s)), base.claude.stats.join('|'));
      assert.deepEqual(base.claude.detailStates.filter(s => s !== 'current'), []);
      assert.equal(base.header, 'month to date'); assert.equal(base.headerStale, false);
      assert(base.codex.stats.some(s => /\$235\.73/.test(s)), 'month total');
    });

    // Same percentages, newer observation, detail delayed by a 503.
    w.claudeAt = now - 30; w.codexAt = now - 20; w.limitsStatus = 503;
    await heightsReset(page);
    await drive(page, hits, ['snapshot']);
    let pending = await snap(page);
    await check(tag + ' newer timestamp with same percentages keeps previous dated detail', async () => {
      assert.deepEqual(pending.claude.stats, base.claude.stats);
      assert.deepEqual(pending.codex.stats, base.codex.stats);
      assert.deepEqual(pending.claude.pcts, base.claude.pcts);
      assert(pending.claude.detailStates.includes('previous'), pending.claude.detailStates.join(','));
      // The changed sample requests its detail immediately; with the 503 in
      // place that attempt may already have failed, which reads as delayed.
      assert.match(pending.claude.status, /detail (pending|delayed)/);
      assert.match(pending.claude.status, /showing .* detail/);
      assert(await minHeight(page) >= base.height - 2, 'panel collapsed to ' + await minHeight(page) + ' from ' + base.height);
      assert(await page.evaluate(() => window.__claude === document.querySelector('[data-provider="claude"]')
        && window.__codex === document.querySelector('[data-provider="codex"]')), 'provider node identity');
    });
    await drive(page, hits, ['limits']);
    pending = await snap(page);
    await check(tag + ' delayed detail is reported without dropping values', () => {
      assert.match(pending.claude.status, /detail delayed/);
      assert.deepEqual(pending.claude.stats, base.claude.stats);
      assert.match(pending.banner, /LIMITS/);
    });
    w.limitsStatus = 200; w.claudeCost.limit = 165.14;
    await drive(page, hits, ['limits']);
    let current = await snap(page);
    await check(tag + ' matching detail replaces pending state', () => {
      assert.deepEqual(current.claude.detailStates.filter(s => s !== 'current'), []);
      assert.doesNotMatch(current.claude.status, /pending|delayed/);
      assert(current.claude.stats.some(s => /~\$1,032\.13/.test(s)), current.claude.stats.join('|'));
      assert.doesNotMatch(current.banner, /LIMITS/);
    });

    // Changed sample: new percentage must not be paired with the old spend.
    w.claude.weekly = 18; w.claudeAt = now - 10; w.limitsStatus = 503;
    await heightsReset(page);
    await drive(page, hits, ['snapshot']);
    pending = await snap(page);
    await check(tag + ' changed percentage shows immediately while old dollars stay dated', async () => {
      assert.equal(pending.claude.pcts[0], '18%');
      assert(pending.claude.stats.some(s => /~\$1,032\.13/.test(s)), 'previous estimate retained verbatim');
      assert(!pending.claude.stats.some(s => /~\$917\.44/.test(s)), 'new percentage recomputed against old spend');
      assert(pending.claude.detailStates.includes('previous'));
      assert(await minHeight(page) >= current.height - 2, 'collapse');
    });
    // Obsolete detail: an enriched response for an older sample cannot win.
    const staleAt = now - 30;
    w.fullOverride = () => { const saved = [w.claudeAt, w.claude.weekly]; w.claudeAt = staleAt; w.claude.weekly = 16;
      w.fullOverride = null; const body = w.full(); [w.claudeAt, w.claude.weekly] = saved; w.fullOverride = () => body; return body; };
    w.limitsStatus = 200;
    await drive(page, hits, ['limits']);
    pending = await snap(page);
    await check(tag + ' obsolete detail for an older sample is ignored', () => {
      assert.equal(pending.claude.pcts[0], '18%');
      assert(pending.claude.detailStates.includes('previous'));
      assert.match(pending.claude.status, /detail pending/);
    });
    w.fullOverride = null; w.claudeCost.limit = 170;
    await drive(page, hits, ['limits']);
    current = await snap(page);
    await check(tag + ' matching detail for the changed sample is applied', () => {
      assert(current.claude.stats.some(s => /~\$944\.44/.test(s)), current.claude.stats.join('|'));
      assert.deepEqual(current.claude.detailStates.filter(s => s !== 'current'), []);
    });

    // Focus and the open chart survive a changed refresh.
    await page.evaluate(() => document.getElementById('oauthChartTrigger').focus());
    w.claudeAt = now - 5; w.claude.weekly = 20; w.claudeCost.limit = 180;
    await drive(page, hits, ['snapshot', 'limits']);
    await check(tag + ' keyboard focus stays on the same chart trigger', async () => {
      assert(await page.evaluate(() => window.__trigger === document.activeElement && window.__trigger.isConnected));
    });
    await page.locator('#oauthChartTrigger').click();
    await page.waitForFunction(() => Chart.getChart(document.querySelector('#usageChartCanvas')));
    const chartBefore = await page.evaluate(() => {
      const c = Chart.getChart(document.querySelector('#usageChartCanvas'));
      c.options.scales.x.min = 30; c.options.scales.x.max = 90; c.update('none');
      const data = c.data.datasets[0].data.filter(v => v !== null);
      return {id: c.id, last: data[data.length - 1], datasets: c.data.datasets.length};
    });
    w.hourly.push({h: 160, c: 30}); w.claudeAt = now - 2;
    await drive(page, hits, ['snapshot', 'limits']);
    await check(tag + ' open pace chart updates in place with zoom retained', async () => {
      const after = await page.evaluate(() => {
        const c = Chart.getChart(document.querySelector('#usageChartCanvas'));
        const data = c.data.datasets[0].data.filter(v => v !== null);
        return {id: c.id, last: data[data.length - 1], min: c.options.scales.x.min, max: c.options.scales.x.max, datasets: c.data.datasets.length};
      });
      assert.equal(after.id, chartBefore.id);
      assert.equal(after.datasets, chartBefore.datasets);
      assert.deepEqual([after.min, after.max], [30, 90]);
      assert(after.last > chartBefore.last, 'chart data did not change');
    });
    await page.keyboard.press('Escape');

    // Removal: full omission removes providers, never resurrects estimates.
    w.codexPresent = false;
    await drive(page, hits, ['snapshot', 'limits']);
    let removed = await snap(page);
    await check(tag + ' removed provider disappears while independent month totals stay', () => {
      assert.equal(removed.codex, null);
      assert(removed.zen && /\$2\.50/.test(removed.zen.text), 'zen month total');
    });
    w.oauthPresent = false;
    await drive(page, hits, ['snapshot', 'limits']);
    removed = await snap(page);
    await check(tag + ' removed Claude clears its section and the header stays neutral', () => {
      assert.equal(removed.claude, null);
      assert.equal(removed.header, 'month to date'); assert.equal(removed.headerStale, false);
      assert.doesNotMatch(removed.text, /pending|delayed|observed/);
    });
    w.oauthPresent = true; w.codexPresent = true;
    await drive(page, hits, ['snapshot', 'limits']);
    await check(tag + ' providers return after removal', async () => {
      const back = await snap(page);
      assert(back.claude && back.codex);
      assert.deepEqual(back.claude.detailStates.filter(s => s !== 'current'), []);
    });

    // Stale and expired observations describe the past, not current quota.
    // An older sample can never replace a newer one in a running page
    // (monotonic accounting), so this is a fresh load against aged fixtures.
    w.claudeAt = now - 26.5 * HOUR; w.fiveReset = now - 20 * HOUR; w.codexAt = now - 2 * HOUR;
    w.claudeSource = 'meridian-oauth';
    await page.reload({waitUntil: 'domcontentloaded'});
    await page.waitForFunction(() => document.querySelector('[data-provider="codex"] .rate-gauge-stat'));
    await page.waitForTimeout(150);
    const stale = await snap(page);
    await check(tag + ' stale and expired gauges never claim current remaining or estimates', () => {
      assert.match(stale.claude.status, /⚠ .*observed 1d/);
      assert.match(stale.claude.status, /meridian-oauth/);
      assert.equal(stale.claude.windowStates[0], 'stale');
      assert.equal(stale.claude.windowStates[1], 'expired');
      assert.match(stale.claude.remaining[1], /window ended/);
      assert.doesNotMatch(stale.claude.remaining[0], /^\d+% remaining$/);
      assert.doesNotMatch(stale.claude.text, /est\. window budget|budget left|Projected by reset|under pace|over pace/i);
      assert(stale.claude.labels.some(l => /rolling 7d/.test(l)), 'measured rolling spend retained: ' + stale.claude.labels.join('|'));
      assert.equal(stale.claude.markers, 0);
      assert.equal(stale.codex.windowStates[0], 'stale');
      assert.match(stale.codex.status, /⚠ .*observed 2h/);
      assert.doesNotMatch(stale.codex.text, /est\. window capacity|budget left/i);
      assert(stale.codex.stats.some(s => /\$28\.90/.test(s)), 'measured provider spend retained');
      assert.equal(stale.header, 'month to date'); assert.equal(stale.headerStale, false);
    });
    w.codexAt = now + 120;
    await drive(page, hits, ['snapshot', 'limits']);
    await check(tag + ' tolerated clock skew is labeled and never fed to inference', async () => {
      const skew = await snap(page);
      assert.equal(skew.codex.windowStates[0], 'unverified');
      assert.match(skew.codex.status, /observed just now .*clock ahead/);
      assert.doesNotMatch(skew.codex.text, /est\. window capacity|budget left|\d+% remaining/i);
      assert(skew.codex.stats.some(s => /\$28\.90/.test(s)), 'measured spend still shown');
    });
    w.codexAt = now + 900;
    await drive(page, hits, ['snapshot', 'limits']);
    await check(tag + ' far-future observation time is not fresh', async () => {
      const future = await snap(page);
      assert.equal(future.codex.windowStates[0], 'invalid');
      assert.match(future.codex.status, /observation time/);
      assert.doesNotMatch(future.codex.text, /est\. window capacity|budget left/i);
    });
    await check(tag + ' keyed patching tolerates hostile key names', async () => {
      const result = await page.evaluate(() => {
        const host = document.createElement('div');
        TokenfoldDom.patchHTML(host, '<b data-key="__proto__">a</b><i data-key="constructor">b</i><u data-key="hasOwnProperty">c</u>');
        const first = host.firstChild;
        TokenfoldDom.patchHTML(host, '<u data-key="hasOwnProperty">c2</u><b data-key="__proto__">a2</b>');
        return {text: host.textContent, kept: first === host.lastChild, count: host.childNodes.length};
      });
      assert.deepEqual(result, {text: 'c2a2', kept: true, count: 2});
    });
    await check(tag + ' no horizontal overflow and no page errors', async () => {
      const last = await snap(page);
      assert.equal(last.scrollWidth, last.viewport);
      assert.deepEqual(hits.errors, []);
    });
  } finally { await close(); }
}

async function pendingObservationAges(browser) {
  const w = world(Date.now() / 1000);
  const {page, hits, close} = await open(browser, 'personal', w);
  try {
    await page.waitForFunction(() => document.querySelector('[data-provider="codex"] .rate-gauge-stat'));
    w.claudeAt = w.now - 10; w.codexAt = w.now - 5; w.limitsStatus = 503;
    await drive(page, hits, ['snapshot']);
    // A coherent previous sample is safe only while that observation/window
    // remains current. A delayed detail request can outlive that validity.
    await page.clock.setFixedTime(new Date((w.now + 2 * HOUR + 1) * 1000));
    await drive(page, hits, ['snapshot']);
    const aged = await snap(page);
    await check('aging pending detail suppresses Claude inference', () => {
      assert.doesNotMatch(aged.claude.text, /est\. window budget|budget left|under pace|over pace|Projected by reset/i);
    });
    await check('aging pending detail suppresses provider inference', () => {
      assert.doesNotMatch(aged.codex.text, /est\. window capacity|budget left|Projected by reset/i);
    });
    await check('aging pending detail retains honestly measured rolling spend', () => {
      assert(aged.claude.labels.some(label => /rolling 7d/.test(label)), aged.claude.labels.join('|'));
    });
    await check('expired window does not retain pending dollar estimates', async () => {
      const text = await page.locator('[data-provider="claude"] [data-key="five_hour"]').textContent();
      assert.doesNotMatch(text, /\$|budget left|% remaining/);
    });
  } finally { await close(); }
}

async function extraUsageHonesty(browser) {
  const w = world(Date.now() / 1000);
  w.extraUsage = {enabled: true};
  const {page, hits, close} = await open(browser, 'personal', w);
  try {
    await page.waitForSelector('[data-key="extra"]');
    await check('unreported extra usage is not fabricated as zero or remaining quota', async () => {
      const text = await page.locator('[data-key="extra"]').textContent();
      assert.doesNotMatch(text, /\$0\.00|\b0%|100% remaining/);
    });
    w.extraUsage = {enabled: true, pct: 10, monthly_limit_cents: 10000, used_cents: 1000};
    w.claudeAt = w.now - 90;
    await drive(page, hits, ['snapshot', 'limits']);
    await page.clock.setFixedTime(new Date((w.now + 2 * HOUR) * 1000));
    await drive(page, hits, ['snapshot']);
    await check('stale extra usage retains readings without current remaining claims', async () => {
      const text = await page.locator('[data-key="extra"]').textContent();
      assert.match(text, /\$10\.00/);
      assert.doesNotMatch(text, /\d+% remaining/);
    });
  } finally { await close(); }
}

async function earlySnapshotFailure(browser) {
  const w = world(Date.now() / 1000);
  w.snapshotFailFirst = true;
  const {page, hits, close} = await open(browser, 'personal', w);
  try {
    await page.waitForFunction(() => document.querySelector('[data-provider="codex"] .rate-gauge-stat'));
    await page.waitForTimeout(150);
    await check('failed head-started snapshot falls back without an error flash', async () => {
      assert(hits.snapshot >= 2, 'no fallback fetch');
      assert.equal(await page.evaluate(() => window.__unavailable), false);
      assert.doesNotMatch(await page.evaluate(() => document.getElementById('refreshIndicator').textContent), /LIMIT/);
      assert.deepEqual(hits.errors, []);
    });
  } finally { await close(); }
}

async function enterpriseInitialView(browser) {
  const now = Date.now() / 1000;
  const w = world(now);
  w.oauthPresent = false; w.limitsStatus = 503;
  w.monthlyBudget = {budget_usd: 500, mtd_cost: 120, month: 'Sep 2026', elapsed_fraction: 0.2,
    pace: 'on', business_days: 22, month_end_epoch: now + 20 * DAY};
  const {page, hits, close} = await open(browser, 'enterprise', w, {width: 390, height: 800});
  try {
    await page.waitForFunction(() => document.querySelector('[data-provider="codex"] .rate-gauge-pct'));
    await until(() => hits.limits >= 1, 'limits attempt');
    await page.waitForTimeout(150);
    const initial = await snap(page);
    await check('enterprise first paint has no false unset budget or zero pace cells', async () => {
      assert.equal(await page.locator('#mbSetBtn').count(), 0);
      assert.equal(await page.locator('#weekPacePanel .rate-gauge-stat').count(), 0);
      assert.doesNotMatch(initial.text, /\$0\.00|0m/);
      assert.equal(initial.claude, null);
      assert.match(initial.codex.status, /observed/);
    });
    w.limitsStatus = 200;
    await drive(page, hits, ['limits']);
    await check('enterprise detail renders budget and pace once known', async () => {
      assert.equal(await page.locator('#mbBudgetEdit').count(), 1);
      assert.match(await page.locator('#weekPacePanel').textContent(), /15h 4m/);
      assert.equal((await snap(page)).scrollWidth, 390);
    });
    await page.locator('#mbBudgetEdit').click();
    await page.locator('.mb-edit-input').fill('1200');
    await page.evaluate(() => { window.__input = document.querySelector('.mb-edit-input'); });
    await drive(page, hits, ['snapshot', 'limits']);
    await check('enterprise open budget edit survives both feeds', async () => {
      assert(await page.evaluate(() => window.__input.isConnected && window.__input.value === '1200' && document.activeElement === window.__input));
      assert.deepEqual(hits.errors, []);
    });
  } finally { await close(); }
}

// The bundled headless shell may be absent for an npx-cached Playwright;
// the installed Chrome channel is the fallback.
async function launch() {
  try { return await chromium.launch({headless: true}); }
  catch (error) { return chromium.launch({headless: true, channel: 'chrome'}); }
}

module.exports = {world, open, drive, snap, check, failures, launch};

if (require.main === module) (async () => {
  const browser = await launch();
  try {
    for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 800}]) {
      try { await personalScenarios(browser, viewport); }
      catch (error) { failures.push(viewport.width + 'px scenario aborted: ' + error.message); console.log(error.stack); }
    }
    for (const scenario of [pendingObservationAges, extraUsageHonesty]) {
      try { await scenario(browser); }
      catch (error) { failures.push(scenario.name + ' aborted: ' + error.message); }
    }
    try { await earlySnapshotFailure(browser); }
    catch (error) { failures.push('early snapshot scenario aborted: ' + error.message); }
    try { await enterpriseInitialView(browser); }
    catch (error) { failures.push('enterprise scenario aborted: ' + error.message); }
  } finally { await browser.close(); }
  if (failures.length) { console.error('\n' + failures.length + ' failing check(s):\n- ' + failures.join('\n- ')); process.exitCode = 1; }
  else console.log('\nquota refresh presentation: all browser checks passed');
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
