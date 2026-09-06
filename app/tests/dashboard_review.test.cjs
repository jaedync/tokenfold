// Independent-review transition regressions using the real rendered dashboard.
// Reuses isolated route fixtures, never production data or provider credentials.
const assert = require('node:assert/strict');
const {world, open, drive, snap, check, failures, launch} = require('./dashboard_dom.test.cjs');
const HOUR = 3600, DAY = 86400;
const inference = /est\. window (budget|capacity)|budget left|under pace|over pace|Projected by reset/i;

async function ready(page) {
  await page.waitForFunction(() => document.querySelector('[data-provider="codex"] .rate-gauge-stat'));
  await page.waitForTimeout(100);
}
function withTrendNotes(w) {
  const original = w.oauth;
  w.oauth = full => {
    const oauth = original(full);
    if (!oauth.trend) return oauth;
    return {...oauth, trend: {...oauth.trend, seven_day: {...oauth.trend.seven_day,
      burn_6h_pct_per_hr: 2.25, eta_100_epoch: w.now + HOUR, pace: 'over'}}};
  };
}
async function chartState(page) {
  return page.evaluate(() => {
    const chart = Chart.getChart(document.getElementById('usageChartCanvas'));
    return {id: chart.id, labels: chart.data.datasets.map(d => d.label),
      min: chart.options.scales.x.min, max: chart.options.scales.x.max,
      hasY2: !!chart.scales.y2};
  });
}
async function zoomChart(page) {
  await page.locator('#oauthChartTrigger').click();
  await page.waitForFunction(() => Chart.getChart(document.getElementById('usageChartCanvas')));
  await page.evaluate(() => {
    const chart = Chart.getChart(document.getElementById('usageChartCanvas'));
    chart.options.scales.x.min = 30; chart.options.scales.x.max = 90; chart.update('none');
  });
  return chartState(page);
}

async function sourceTransfer(browser, offset) {
  const w = world(Date.now() / 1000); w.claudeSource = 'client'; withTrendNotes(w);
  const legacyAt = w.claudeAt;
  const {page, hits, close} = await open(browser, 'personal', w);
  try {
    await ready(page); const beforeChart = await zoomChart(page);
    w.claudeSource = 'meridian-oauth'; w.claudeAt = legacyAt + offset;
    w.claude.weekly = 7; w.limitsStatus = 503;
    await drive(page, hits, ['snapshot']);
    await check('source transfer ' + offset + ': accepts owner without carrying old-account detail', async () => {
      const value = await snap(page);
      assert.match(value.claude.status, /meridian-oauth/);
      assert.equal(value.claude.pcts[0], '7%');
      assert.doesNotMatch(value.claude.text, /\$163\.15|1,019\.69|2\.25%\/hr/);
      assert.equal((await chartState(page)).id, beforeChart.id);
      assert.deepEqual((await chartState(page)).labels, ['Actual usage']);
    });
    w.claudeSource = 'client'; w.claudeAt = w.now - 1; w.claude.weekly = 82;
    w.limitsStatus = 200; await drive(page, hits, ['snapshot', 'limits']);
    await check('source transfer ' + offset + ': late legacy cannot revert owner', async () => {
      const value = await snap(page);
      assert.match(value.claude.status, /meridian-oauth/); assert.equal(value.claude.pcts[0], '7%');
      assert.doesNotMatch(value.claude.text, /\$163\.15|1,019\.69|2\.25%\/hr/);
    });
    await sourceRemovalAndReturn(page, hits, w, offset);
    await check('source transfer ' + offset + ': no page errors', () => assert.deepEqual(hits.errors, []));
  } finally { await close(); }
}
async function sourceRemovalAndReturn(page, hits, w, offset) {
  w.oauthPresent = false; await drive(page, hits, ['snapshot', 'limits']);
  await check('source transfer ' + offset + ': authoritative omission removes section', async () => {
    assert.equal((await snap(page)).claude, null);
  });
  w.oauthPresent = true; await drive(page, hits, ['snapshot', 'limits']);
  await check('source transfer ' + offset + ': omission does not forget owner', async () => {
    assert.equal((await snap(page)).claude, null);
  });
  w.claudeSource = 'meridian-oauth'; w.claudeAt = w.now - 20; w.claude.weekly = 7;
  w.claudeCost = {limit: 21, active: 3600, five: 3, fable: 2}; w.trend = false;
  await drive(page, hits, ['snapshot', 'limits']);
  await check('source transfer ' + offset + ': managed matching detail returns', async () => {
    const value = await snap(page);
    assert.match(value.claude.status, /meridian-oauth/); assert.match(value.claude.text, /\$21\.00/);
    assert.doesNotMatch(value.claude.status, /pending|delayed/);
  });
}

async function retainedProvenance(browser) {
  const w = world(Date.now() / 1000); withTrendNotes(w);
  const {page, hits, close} = await open(browser, 'personal', w);
  try {
    await ready(page);
    assert.match((await snap(page)).claude.text, /2\.25%\/hr/);
    w.claudeAt = w.now - 10; w.weeklyReset += DAY; w.limitsStatus = 503;
    await drive(page, hits, ['snapshot']);
    await check('reset while pending cannot reuse previous trend inference', async () => {
      assert.doesNotMatch((await snap(page)).claude.text, /2\.25%\/hr|limit ~/);
    });
    // Latest metadata remains fresh, but retained full detail is over an hour old.
    await page.clock.setFixedTime(new Date((w.now + 2 * HOUR) * 1000));
    w.claudeAt = w.now + 2 * HOUR - 10; w.codexAt = w.claudeAt;
    await drive(page, hits, ['snapshot']);
    await check('fresh metadata cannot revalidate stale retained detail or trends', async () => {
      const value = await snap(page);
      assert.doesNotMatch(value.claude.text, /2\.25%\/hr|limit ~|est\. window budget|budget left/);
      assert.doesNotMatch(value.codex.text, /est\. window capacity|budget left/);
      assert(value.claude.labels.some(label => /rolling 7d/.test(label)));
    });
  } finally { await close(); }
}

async function clockWithoutSuccessfulResponses(browser, offline) {
  const w = world(Date.now() / 1000);
  const {page, hits, close} = await open(browser, 'personal', w, undefined, true);
  try {
    await ready(page);
    if (offline) await page.evaluate(() => {
      Object.defineProperty(navigator, 'onLine', {configurable: true, get: () => false});
      window.dispatchEvent(new Event('offline'));
    });
    else {
      w.snapshotStatus = 503; w.limitsStatus = 503;
      await drive(page, hits, ['snapshot', 'limits']);
    }
    const before = {snapshot: hits.snapshot, limits: hits.limits};
    await page.clock.fastForward((2 * HOUR + 1) * 1000);
    await page.waitForTimeout(100);
    await check((offline ? 'offline' : 'both feeds failing') + ': presentation ages without successful responses', async () => {
      const value = await snap(page);
      assert.match(value.claude.status, /stale.*observed 2h/);
      assert.doesNotMatch(value.claude.text, inference);
      assert.doesNotMatch(value.claude.remaining.join('|'), /\d+% remaining/);
      if (offline) assert.deepEqual({snapshot: hits.snapshot, limits: hits.limits}, before);
    });
    if (offline) await hiddenClock(page, hits, before);
    await check('clock run has no page errors', () => assert.deepEqual(hits.errors, []));
  } finally { await close(); }
}
async function hiddenClock(page, hits, before) {
  await page.evaluate(() => {
    window.__hidden = true;
    Object.defineProperty(document, 'hidden', {configurable: true, get: () => window.__hidden});
    document.dispatchEvent(new Event('visibilitychange'));
  });
  const hiddenText = (await snap(page)).text;
  await page.clock.fastForward(HOUR * 1000);
  await check('hidden presentation clock does not repaint or fetch', async () => {
    assert.equal((await snap(page)).text, hiddenText);
    assert.deepEqual({snapshot: hits.snapshot, limits: hits.limits}, before);
  });
  await page.evaluate(() => { window.__hidden = false; document.dispatchEvent(new Event('visibilitychange')); });
  await check('visible resume ages immediately even while offline', async () => {
    assert.match((await snap(page)).claude.status, /stale.*observed 3h/);
  });
}

async function chartDatasetTransitions(browser) {
  const w = world(Date.now() / 1000); const original = w.oauth; let reference = true;
  w.oauth = full => {
    const oauth = original(full);
    if (!full || reference) return oauth;
    const {limit_window, ...withoutReference} = oauth; return withoutReference;
  };
  const {page, hits, close} = await open(browser, 'personal', w);
  try {
    await ready(page); const before = await zoomChart(page);
    for (const [trend, ref, expected] of [[false, false, 1], [true, false, 2], [true, true, 3]]) {
      w.trend = trend; reference = ref; await drive(page, hits, ['limits']);
      await check('chart datasets=' + expected + ': keeps instance and zoom with correct scales', async () => {
        const after = await chartState(page);
        assert.equal(after.id, before.id); assert.deepEqual([after.min, after.max], [30, 90]);
        assert.equal(after.labels.length, expected); assert.equal(after.hasY2, trend);
        assert.equal(after.labels.includes('Even drain'), ref);
      });
    }
    await check('dataset transitions have no page errors', () => assert.deepEqual(hits.errors, []));
  } finally { await close(); }
}

async function extraUsageBoundaries(browser) {
  const w = world(Date.now() / 1000); w.extraUsage = {enabled: true};
  const {page, hits, close} = await open(browser, 'personal', w);
  try {
    await ready(page);
    for (const extra of [{enabled: true, pct: null, used_cents: null, monthly_limit_cents: null},
      {enabled: true, pct: 0, used_cents: 0, monthly_limit_cents: 0}, {enabled: true, used_cents: 125}]) {
      w.extraUsage = extra; w.claudeAt += 1; await drive(page, hits, ['snapshot', 'limits']);
      await check('extra usage preserves null/zero/partial distinctions: ' + JSON.stringify(extra), async () => {
        const text = await page.locator('[data-key="extra"]').textContent();
        if (extra.pct === 0) assert.match(text, /limit \$0\.00.*used \$0\.00.*0%.*100% remaining/);
        else assert.doesNotMatch(text, /0%|% remaining|\$0\.00/);
        if (extra.used_cents === 125) assert.match(text, /\$1\.25/);
      });
    }
    w.claudeAt = w.now + 120; w.extraUsage = {enabled: true, pct: 10, used_cents: 125};
    await drive(page, hits, ['snapshot', 'limits']);
    await check('future extra usage cannot claim current remaining', async () => {
      assert.doesNotMatch(await page.locator('[data-key="extra"]').textContent(), /\d+% remaining/);
    });
    for (const extra of [{enabled: false}, null]) {
      w.extraUsage = extra; w.claudeAt += 1; await drive(page, hits, ['snapshot', 'limits']);
      await check('disabled/removed extra usage disappears', async () => assert.equal(await page.locator('[data-key="extra"]').count(), 0));
    }
  } finally { await close(); }
}

(async () => {
  const browser = await launch();
  try {
    for (const offset of [-60, 0, 60]) {
      try { await sourceTransfer(browser, offset); }
      catch (error) { failures.push('source transfer ' + offset + ' aborted: ' + error.stack); }
    }
    for (const action of [retainedProvenance, chartDatasetTransitions, extraUsageBoundaries]) {
      try { await action(browser); } catch (error) { failures.push(action.name + ': ' + error.stack); }
    }
    for (const offline of [true, false]) {
      try { await clockWithoutSuccessfulResponses(browser, offline); }
      catch (error) { failures.push('clock ' + offline + ': ' + error.stack); }
    }
  } finally { await browser.close(); }
  if (failures.length) { console.error(failures.length + ' failing checks:\n- ' + failures.join('\n- ')); process.exitCode = 1; }
  else console.log('independent-review browser transitions: all passed');
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
