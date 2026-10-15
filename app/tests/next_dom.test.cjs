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
  return {page, errors, context};
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
      assert.match(await text(page, '.p-rail'), /no quota windows/i);
      assert.match(await page.evaluate(() => location.search), /scope=enterprise/);
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

    await check('views: tabs switch to a placeholder view and back', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal'});
      await page.waitForSelector('.p-hero .verdict-word');
      await page.click('[data-view-link="ledger"]');
      await page.waitForSelector('[data-panel="ledger-placeholder"]');
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
  } finally {
    await browser.close();
  }
  if (failures.length) {
    console.error(`\n${failures.length} failure(s):\n` + failures.join('\n'));
    process.exit(1);
  }
})().catch(error => { console.error(error); process.exit(1); });
