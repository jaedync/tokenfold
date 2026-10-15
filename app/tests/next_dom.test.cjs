// Browser checks for /next. Loads the rendered shell in headless Chromium;
// page.route serves static files from the repo, so ?mock=1 reads the
// committed fixtures. No server, network or production data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.TOKENFOLD_PLAYWRIGHT || 'playwright');

const ROOT = path.resolve(__dirname, '..', '..');
const ORIGIN = 'http://tokenfold.test';
const HTML = fs.readFileSync(process.env.TOKENFOLD_HTML_NEXT, 'utf8');
const FIXTURES = path.join(ROOT, 'static', 'next', 'fixtures');
const failures = [];
async function check(name, action) {
  try { await action(); console.log('ok - ' + name); }
  catch (error) { failures.push(name + ': ' + error.message); console.log('FAIL - ' + name + '\n  ' + error.message); }
}

async function open(browser, {viewport, url, api404 = false, fixtureOverride = null}) {
  const context = await browser.newContext({viewport, timezoneId: 'America/Chicago'});
  const page = await context.newPage();
  const errors = [];
  const requests = [];
  page.on('request', r => requests.push(new URL(r.url()).pathname));
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => {
    // A real-mode 404 is the browser's own network log line; anything else is a bug.
    if (m.type() === 'error' && !(api404 && /status of 404/.test(m.text()))) errors.push(m.text());
  });
  await page.route('**/*', route => {
    const u = new URL(route.request().url());
    if (u.origin !== ORIGIN) return route.fulfill({status: 204, body: ''});
    if (u.pathname === '/next') return route.fulfill({contentType: 'text/html', body: HTML});
    // The shell serves static/next under /static/next-<version>/ (cache busting).
    const pathname = u.pathname.replace(/^\/static\/next-[0-9a-f]+\//, '/static/next/');
    const fixture = pathname.match(/^\/static\/next\/fixtures\/(.+)\.json$/);
    if (fixture && fixtureOverride) {
      const body = fixtureOverride(fixture[1], JSON.parse(fs.readFileSync(path.join(FIXTURES, fixture[1] + '.json'), 'utf8')));
      if (body) return route.fulfill({json: body});
    }
    if (pathname.startsWith('/static/')) return route.fulfill({path: path.join(ROOT, pathname)});
    if (u.pathname.startsWith('/api/') && api404) return route.fulfill({status: 404, json: {detail: 'Not Found'}});
    return route.fulfill({status: 404, body: ''});
  });
  await page.goto(ORIGIN + url);
  return {page, errors, context, requests};
}

const text = (page, sel) => page.locator(sel).first().innerText();

(async () => {
  const browser = await chromium.launch();
  try {
    await check('desktop mock: every Runway region renders from fixtures', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal#runway'});
      await page.waitForSelector('.p-exchange .ex-v');
      assert.match(await text(page, '.lb-wait'), /2/);
      assert.match(await text(page, '.p-hero .verdict-word'), /^TIGHT$/i);
      assert.match(await text(page, '.p-hero .hero-v'), /^92%$/);
      assert.equal(await page.locator('.p-chart canvas').count() > 0, true);
      assert.equal(await page.locator('.p-rail .rw').count(), 6);
      assert.match(await text(page, '.p-now .grp[data-state="waiting"] .grp-head'), /Waiting on you · 2/i);
      assert.equal(await page.locator('.p-now .row').count(), 8);
      assert.match(await text(page, '.p-now .row'), /laptop/);
      assert.match(await text(page, '.p-today .today-v'), /\$412/);
      assert.match(await text(page, '.p-exchange .ex-v'), /\$2,440/);
      assert.equal(await page.title(), '92% 7d · 2 waiting · Tokenfold');
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
      assert.equal(overflow, false, 'page scrolls sideways');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('phone mock: hero, Now, other windows, chart, Today, Exchange rate in that order', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 390, height: 844}, url: '/next?mock=1&scope=personal'});
      await page.waitForSelector('.p-exchange .ex-v');
      const tops = await page.evaluate(() => ['hero', 'now', 'rail', 'chart', 'today', 'exchange']
        .map(id => document.querySelector(`[data-panel="${id}"]`).getBoundingClientRect().top));
      assert.deepEqual([...tops].sort((a, b) => a - b), tops);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('scope switch: Work shows the billing meter against the budget', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1024, height: 768}, url: '/next?mock=1&scope=personal'});
      await page.waitForSelector('.p-hero .verdict-word');
      await page.click('[data-scope="enterprise"]');
      await page.waitForFunction(() => /Billing meter/i.test(document.querySelector('#hero-k').textContent));
      assert.match(await text(page, '.p-hero .verdict-word'), /over budget pace/i);
      assert.match(await page.evaluate(() => location.search), /scope=enterprise/);
      // No quota windows and no exchange rate in work: those panels hide, the hero says why.
      assert.match(await text(page, '.p-hero .short'), /billing meter is the limit/i);
      assert.equal(await page.locator('[data-panel="rail"]').isHidden(), true);
      assert.equal(await page.locator('[data-panel="exchange"]').isHidden(), true);
      // Now takes the rail's place beside the hero: no empty grid cell.
      const box = (id) => page.evaluate((id) => {
        const r = document.querySelector(`[data-panel="${id}"]`).getBoundingClientRect();
        return {top: r.top, left: r.left, right: r.right};
      }, id);
      const hero = await box('hero'), now = await box('now');
      assert.equal(now.top, hero.top, 'Now is not beside the hero');
      assert.ok(now.left > hero.right, 'Now is not right of the hero');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      // Back to personal: both panels return.
      await page.click('button[data-scope="personal"]');
      await page.waitForSelector('[data-panel="rail"]:not([hidden]) .rw');
      assert.equal(await page.locator('[data-panel="exchange"]').isHidden(), false);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('live bar: fits at 1440, 1024, 800 and 390 with no sideways scroll; the clock and data age stay in view above 700 px', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal'});
      await page.waitForSelector('.p-hero .verdict-word');
      for (const width of [1440, 1024, 800, 390]) {
        await page.setViewportSize({width, height: 900});
        const bar = await page.evaluate(() => {
          const el = document.getElementById('livebar');
          const shown = (sel) => {
            const r = el.querySelector(sel).getBoundingClientRect();
            return r.width > 0 && r.right <= el.getBoundingClientRect().right + 0.5;
          };
          return {over: el.scrollWidth - el.clientWidth, clock: shown('.lb-clock'), age: shown('.lb-age')};
        });
        assert.equal(bar.over, 0, `live bar scrolls at ${width}`);
        if (width > 700) assert.ok(bar.clock && bar.age, `clock or data age hidden at ${width}`);
      }
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('real mode without endpoints: each panel says unavailable, the page stays up', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?scope=personal', api404: true});
      await page.waitForFunction(() => document.querySelectorAll('.p-state').length >= 6);
      for (const id of ['hero', 'now', 'rail', 'chart', 'today', 'exchange']) {
        assert.match(await text(page, `[data-panel="${id}"] .p-state`), /unavailable/i, id);
      }
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('views: tabs switch to Ledger and back', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal'});
      await page.waitForSelector('.p-hero .verdict-word');
      await page.click('[data-view-link="ledger"]');
      await page.waitForSelector('[data-panel="ledger-stats"]');
      assert.equal(await page.getAttribute('[data-view-link="ledger"]', 'aria-current'), 'page');
      assert.equal(await page.locator('[data-panel="hero"]').count(), 0);
      await page.click('[data-view-link="runway"]');
      await page.waitForSelector('[data-panel="hero"] .verdict-word');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('a new waiting session is announced and flashes', async () => {
      let calls = 0;
      const override = (name, body) => {
        if (name !== 'now') return null;
        calls += 1;
        if (calls < 2) return body;
        const sessions = body.sessions.map(s => (s.id === 's-c388' ? {...s, state: 'waiting', waiting_reason: 'permission', state_since: body.generated_at} : s));
        return {...body, sessions, live: {...body.live, waiting: 3, working: 4}};
      };
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal', fixtureOverride: override});
      await page.waitForSelector('.p-now .row');
      await page.waitForFunction(() => /devbox · api-server is waiting on you: needs permission/.test(document.getElementById('announce').textContent), null, {timeout: 12000});
      assert.equal(await page.locator('.p-now .row.flash[data-id="s-c388"]').count(), 1);
      assert.match(await page.title(), /3 waiting/);
      assert.deepEqual(errors, []);
      await context.close();
    });

    // ---------------------------------------------------------- Runway polish
    // A too-early binding window: the server has no forecast yet.
    const tooEarly = (name, body) => {
      if (name === 'runway-trajectory') {
        const t = body.points.at(-1)[0];
        return {...body, points: [[t - 6 * 3600, 0], [t - 3 * 3600, 5], [t, 11]], start: t - 6 * 3600};
      }
      if (name !== 'runway') return null;
      const windows = body.windows.map(w => (w.id !== body.binding ? w : {
        ...w, pct: 11, verdict: 'too_early', tight_epoch: null,
        window_start: body.generated_at - 6 * 3600, resets_at: body.generated_at + 162 * 3600,
        pace: {...w.pace, pace_state: 'too_early', expected_pct: 3.6, projected_pct: null, reaches_100_epoch: null, elapsed_fraction: 0.036},
      }));
      return {...body, windows};
    };
    const midWindow = (name, body) => {
      if (name !== 'runway') return null;
      const windows = body.windows.map(w => (w.id !== body.binding ? w : {...w, pace: {...w.pace, elapsed_fraction: 0.5}}));
      return {...body, windows};
    };
    const noSessions = (name, body) => (name === 'now'
      ? {...body, sessions: [], live: {...body.live, working: 0, waiting: 0, subagents: 0}} : null);
    const both = (...fns) => (name, body) => fns.reduce((b, fn) => fn(name, b) || b, body);

    await check('too early: pace shows elapsed time and even pace, no over/under figure', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal', fixtureOverride: tooEarly});
      await page.waitForFunction(() => /too early/i.test(document.querySelector('.p-hero .verdict-word').textContent));
      const pace = await text(page, '.p-hero .facts > div:first-child');
      assert.match(pace, /6 h .*into the window/);
      assert.match(pace, /even pace 4%/);
      assert.doesNotMatch(pace, /over|under|pts/);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('burn: the meter rate is labelled as such; sessions split it when they exist', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal'});
      await page.waitForSelector('.p-hero .burn-list li');
      assert.match(await text(page, '.p-hero .burn-head'), /Meter rate, last hour\s*\+0\.80%\/h/i);
      assert.equal(await page.locator('.p-hero .burnbar').count(), 1);
      assert.equal(await page.locator('.p-hero .burn-list li').count(), 3);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('burn: without live sessions the hero says so plainly and shows no empty bar', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 390, height: 844}, url: '/next?mock=1&scope=personal', fixtureOverride: noSessions});
      await page.waitForSelector('.p-hero .burn-none');
      assert.equal(await text(page, '.p-hero .burn-none'), 'No live session data. Agents report live state through the notify hooks.');
      assert.equal(await page.locator('.p-hero .burnbar').count(), 0);
      assert.equal(await page.title(), '92% 7d · 0 waiting · Tokenfold');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('chart range: late or early in a window opens on the last 24 h; the pick holds', async () => {
      const pressed = (page) => page.getAttribute('.p-chart .zoom [aria-pressed="true"]', 'data-value');
      for (const [override, want] of [[null, 'day'], [tooEarly, 'day'], [midWindow, 'window']]) {
        const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal', fixtureOverride: override});
        await page.waitForSelector('.p-chart canvas');
        await page.waitForFunction(() => document.querySelector('.p-chart .readout').textContent.length > 0);
        assert.equal(await pressed(page), want, `default for ${override ? override.name : 'late window'}`);
        if (override === tooEarly) assert.match(await text(page, '.p-chart .chart-empty'), /limit \(100%\) is above/i);
        const other = want === 'day' ? 'window' : 'day';
        await page.click(`.p-chart .zoom [data-value="${other}"]`);
        await page.waitForTimeout(5500);  // one /api/now poll later
        assert.equal(await pressed(page), other);
        assert.deepEqual(errors, []);
        await context.close();
      }
    });

    await check('masthead mark: each column names its window and gives its value', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 390, height: 844}, url: '/next?mock=1&scope=personal'});
      await page.waitForFunction(() => document.getElementById('mk-bind-v').textContent !== '—');
      assert.equal(await text(page, '#mk-bind-k'), '7D');
      assert.match(await page.locator('#mk-bind-v').textContent(), /^92%?$/);
      assert.match(await page.locator('#mk-short-v').textContent(), /^35%?$/);
      assert.match(await page.getAttribute('#mark', 'title'), /Claude · 7-day, 92% used/);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('work scope: the month chart draws the billed line from the meter', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1024, height: 768}, url: '/next?mock=1&scope=personal'});
      await page.waitForSelector('.p-hero .verdict-word');
      await page.click('[data-scope="enterprise"]');
      await page.waitForFunction(() => /Billing meter/i.test(document.querySelector('#hero-k').textContent));
      await page.waitForSelector('.p-chart canvas');
      assert.equal(await page.locator('.p-chart .chart-empty').isHidden(), true);
      assert.equal(await page.locator('.p-chart .zoom').isHidden(), true);
      assert.match(await text(page, '.p-chart .legend'), /Billed/);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('cache busting: every module and asset loads from the versioned path', async () => {
      const {page, errors, context, requests} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal'});
      await page.waitForSelector('.p-exchange .ex-v');
      const version = await page.evaluate(() => document.body.dataset.version);
      const statics = requests.filter(p => p.startsWith('/static/next'));
      assert.ok(statics.length > 15, `only ${statics.length} static requests`);
      for (const p of statics) assert.ok(p.startsWith(`/static/next-${version}/`), p);
      assert.ok(statics.some(p => p.endsWith('/js/panels/hero.js')), 'nested module');
      assert.deepEqual(errors, []);
      await context.close();
    });
  } finally {
    await browser.close();
  }
  if (failures.length) {
    console.error(`\n${failures.length} failure(s):\n` + failures.join('\n'));
    process.exit(1);
  }
})().catch(error => { console.error(error); process.exit(1); });
