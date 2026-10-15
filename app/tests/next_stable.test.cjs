// Reflow checks for /next: data that changes on every poll must not reload,
// move or reset what the user looks at. Mock mode with a fixture override
// that changes the values on every request, and a fake clock that runs the
// polls. An SSE bump comes from a line this harness appends to main.js (mock
// mode opens no stream). page.route serves the repo; no server, network or
// production data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.TOKENFOLD_PLAYWRIGHT || 'playwright');

const ROOT = path.resolve(__dirname, '..', '..');
const ORIGIN = 'http://tokenfold.test';
const HTML = fs.readFileSync(process.env.TOKENFOLD_HTML_NEXT, 'utf8');
const CSP = process.env.TOKENFOLD_CSP_NEXT || '';
const FIXTURES = path.join(ROOT, 'static', 'next', 'fixtures');
const START = new Date('2026-10-14T12:00:00-05:00');
const failures = [];
async function check(name, action) {
  try { await action(); console.log('ok - ' + name); }
  catch (error) { failures.push(name + ': ' + error.message); console.log('FAIL - ' + name + '\n  ' + error.message); }
}

// Money and percent values grow a little on every request after the first;
// the shape (ids, states, list lengths) stays the same.
const MONEY = /usd|cost|pct|spend|total|value|burn|share|price/i;
const KEEP = /epoch|_at$|since|start|^id$|budget/i;
function drift(v, key, n) {
  if (Array.isArray(v)) return v.map((x) => drift(x, key, n));
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, drift(x, k, n)]));
  if (typeof v === 'number' && key && MONEY.test(key) && !KEEP.test(key)) {
    const out = v * (1 + 0.004 * n);
    return /pct/i.test(key) ? Math.min(100, out) : out;
  }
  return v;
}

// The shell is rendered for the personal scope; the work scope is one click away.
async function open(browser, {viewport, url, hold = null, scope = 'personal'}) {
  const context = await browser.newContext({viewport, timezoneId: 'America/Chicago'});
  const page = await context.newPage();
  const errors = [];
  const counts = {};
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.route('**/*', async route => {
    const u = new URL(route.request().url());
    if (u.origin !== ORIGIN) return route.fulfill({status: 204, body: ''});
    if (u.pathname === '/next') return route.fulfill({contentType: 'text/html', body: HTML, headers: CSP ? {'Content-Security-Policy': CSP} : {}});
    const pathname = u.pathname.replace(/^\/static\/next-[0-9a-f]+\//, '/static/next/');
    const fixture = pathname.match(/^\/static\/next\/fixtures\/(.+)\.json$/);
    if (fixture && fs.existsSync(path.join(FIXTURES, fixture[1] + '.json'))) {
      // A held fixture answers when the check calls page.__release().
      if (hold && hold.test(fixture[1]) && page.__holding) await new Promise((resolve) => page.__held.push(resolve));
      const n = (counts[fixture[1]] = (counts[fixture[1]] || 0) + 1);
      const body = JSON.parse(fs.readFileSync(path.join(FIXTURES, fixture[1] + '.json'), 'utf8'));
      return route.fulfill({json: n === 1 ? body : drift(body, null, n)});
    }
    if (pathname === '/static/next/js/main.js') {
      const text = fs.readFileSync(path.join(ROOT, pathname), 'utf8');
      return route.fulfill({contentType: 'text/javascript', body: `${text}\ndocument.addEventListener('tf:bump', () => feeds.bump());\n`});
    }
    if (pathname.startsWith('/static/')) return route.fulfill({path: path.join(ROOT, pathname)});
    return route.fulfill({status: 404, body: ''});
  });
  page.__held = [];
  page.__holding = true;
  page.__release = () => { page.__holding = false; for (const r of page.__held.splice(0)) r(); };
  await page.clock.install({time: START});
  await page.goto(ORIGIN + url);
  if (scope === 'enterprise') {
    await page.clock.runFor(200);
    await page.click('button[data-scope="enterprise"]');
  }
  return {page, errors, context, counts};
}

// Run the fake clock until every visible panel of the view has content.
async function settle(page) {
  for (let i = 0; i < 300; i += 1) {
    await page.clock.runFor(50);
    await page.waitForTimeout(5);
    const ok = await page.evaluate(() => {
      const ps = [...document.querySelectorAll('#view .panel')].filter((p) => !p.hidden);
      return ps.length > 0 && !ps.some((p) => /Loading/.test(p.innerText));
    });
    if (ok) { await page.clock.runFor(500); await page.waitForTimeout(50); return; }
  }
  throw new Error('the view did not finish loading');
}

// Watch layout shifts, removed nodes and any "Loading" text from now on.
const watch = (page) => page.evaluate(() => {
  const w = window.__watch = {cls: 0, removed: 0, loading: 0, nodes: [...document.querySelectorAll('#view *')]};
  new PerformanceObserver((l) => { for (const e of l.getEntries()) if (!e.hadRecentInput) w.cls += e.value; })
    .observe({type: 'layout-shift'});
  new MutationObserver((ms) => {
    for (const m of ms) for (const n of m.removedNodes) if (n.nodeType === 1) w.removed += 1;
    if ([...document.querySelectorAll('#view .panel:not([hidden])')].some((p) => /Loading/.test(p.innerText))) w.loading += 1;
  }).observe(document.getElementById('view'), {childList: true, subtree: true});
});

// Three poll cycles of the slowest polled feed (120 s), an SSE bump every 30 s.
async function polls(page, seconds = 360) {
  for (let t = 5; t <= seconds; t += 5) {
    if (t % 30 === 0) await page.evaluate(() => document.dispatchEvent(new Event('tf:bump')));
    await page.clock.runFor(5000);
    await page.waitForTimeout(15);
  }
  await page.clock.runFor(500);
  await page.waitForTimeout(100);
}

const watched = (page) => page.evaluate(() => {
  const w = window.__watch;
  return {cls: w.cls, removed: w.removed, loading: w.loading, lost: w.nodes.filter((n) => !n.isConnected).length, total: w.nodes.length};
});

function assertCalm(r, fetched) {
  assert.ok(fetched >= 3, `only ${fetched} responses changed the data`);
  assert.ok(r.cls < 0.01, `layout shift ${r.cls.toFixed(4)}`);
  assert.equal(r.loading, 0, 'a "Loading" line came back');
  assert.equal(r.lost, 0, `${r.lost} of ${r.total} nodes were replaced`);
}

const scrollTo = (page, y) => page.evaluate((y) => window.scrollTo(0, Math.min(y, document.documentElement.scrollHeight - innerHeight)), y);

(async () => {
  const browser = await chromium.launch();
  try {
    for (const [scope, width] of [['personal', 1440], ['personal', 390], ['enterprise', 1440]]) {
      await check(`runway ${scope} ${width}: polls change values in place; scroll, focus, selection and the chart readout stay`, async () => {
        const {page, errors, context, counts} = await open(browser, {viewport: {width, height: 900}, url: '/next?mock=1#runway', scope});
        await settle(page);
        await scrollTo(page, 300);
        const zoom = page.locator('.p-chart .seg button:visible').first();
        if (await zoom.count()) await zoom.focus();
        // A selection in text that does not change with the data.
        await page.evaluate(() => {
          const r = document.createRange(); r.selectNodeContents(document.querySelector('.p-today h2'));
          getSelection().removeAllRanges(); getSelection().addRange(r);
        });
        const plot = page.locator('.p-chart .u-over');
        await plot.scrollIntoViewIfNeeded();
        const box = await plot.boundingBox();
        await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.5);
        await page.clock.runFor(100);
        const before = await page.evaluate(() => ({y: scrollY, focus: document.activeElement, readout: document.querySelector('.p-chart .readout').textContent}));
        assert.doesNotMatch(before.readout, /^Latest/, 'the hover readout did not show');
        await page.evaluate(() => { window.__focus = document.activeElement; });
        await watch(page);
        const n0 = Object.values(counts).reduce((a, b) => a + b, 0);
        await polls(page);
        const fetched = Object.values(counts).reduce((a, b) => a + b, 0) - n0;
        assertCalm(await watched(page), fetched);
        const after = await page.evaluate(() => ({
          y: scrollY, focusKept: document.activeElement === window.__focus,
          sel: getSelection().toString(), readout: document.querySelector('.p-chart .readout').textContent,
          hero: document.querySelector('.p-hero .hero-v').textContent,
        }));
        assert.equal(after.y, before.y, 'the window scrolled');
        assert.ok(after.focusKept, 'the focus moved');
        assert.match(after.sel, /Today/i, 'the text selection is gone');
        assert.doesNotMatch(after.readout, /^Latest/, 'the hover readout went back to the latest value');
        assert.notEqual(after.hero, '—');
        assert.deepEqual(errors, []);
        await context.close();
      });
    }

    for (const width of [1440, 390]) {
      await check(`ledger ${width}: polls keep the drawer open and scrolled, the filter, the focus and the inner scroll`, async () => {
        const {page, errors, context, counts} = await open(browser, {viewport: {width, height: 900}, url: '/next?mock=1#ledger'});
        await settle(page);
        // A model filter from the legend, then a session in the drawer.
        await page.locator('.lo-legend .lg-item').first().click();
        await page.clock.runFor(200);
        await page.locator('.lt-btn').first().click();
        await page.clock.runFor(500);
        await page.waitForSelector('.drawer.is-open');
        await page.evaluate(() => {
          const b = document.querySelector('.dr-body');
          b.scrollTop = Math.min(80, b.scrollHeight - b.clientHeight);
          // The timeline scrolls sideways on a phone; keep it off its right end.
          const t = document.querySelector('.tl-scroll');
          if (t.scrollWidth > t.clientWidth + 40) t.scrollLeft = 20;
          document.querySelector('.dr-nav button[aria-label="Next session"]:not([disabled]), .dr-close').focus();
          window.__focus = document.activeElement;
        });
        const before = await page.evaluate(() => ({
          drawer: document.querySelector('.dr-title').textContent, body: document.querySelector('.dr-body').scrollTop,
          tl: document.querySelector('.tl-scroll').scrollLeft, chips: document.querySelectorAll('.lc-chip').length,
        }));
        assert.equal(before.chips, 1, 'the filter chip is missing');
        await watch(page);
        const n0 = Object.values(counts).reduce((a, b) => a + b, 0);
        await polls(page);
        const fetched = Object.values(counts).reduce((a, b) => a + b, 0) - n0;
        assertCalm(await watched(page), fetched);
        const after = await page.evaluate(() => ({
          open: Boolean(document.querySelector('.drawer.is-open')), drawer: document.querySelector('.dr-title').textContent,
          body: document.querySelector('.dr-body').scrollTop, tl: document.querySelector('.tl-scroll').scrollLeft,
          chips: document.querySelectorAll('.lc-chip').length, focusKept: document.activeElement === window.__focus,
        }));
        assert.ok(after.open, 'the drawer closed');
        assert.equal(after.drawer, before.drawer, 'the drawer shows another session');
        assert.equal(after.body, before.body, 'the drawer scrolled');
        assert.equal(after.tl, before.tl, 'the timeline scrolled sideways');
        assert.equal(after.chips, 1, 'the filter is gone');
        assert.ok(after.focusKept, 'the focus moved');
        assert.deepEqual(errors, []);
        await context.close();
      });
    }
    for (const [scope, width] of [['personal', 1440], ['personal', 390], ['enterprise', 1440]]) {
      await check(`history ${scope} ${width}: polls keep the picked month, the focus and the scroll`, async () => {
        const {page, errors, context, counts} = await open(browser, {viewport: {width, height: 900}, url: '/next?mock=1#history', scope});
        await settle(page);
        // Pick the oldest month bar (not the default pick) with the keyboard focus on it.
        const bar = page.locator('.mo-col').first();
        await bar.scrollIntoViewIfNeeded();
        await bar.click();
        await page.clock.runFor(100);
        await page.evaluate(() => { window.__focus = document.activeElement; });
        const before = await page.evaluate(() => ({
          y: scrollY, picked: document.querySelector('.mo-col[aria-pressed="true"]').getAttribute('data-key'),
          read: document.querySelector('.mo-read .kicker').textContent,
        }));
        await watch(page);
        const n0 = Object.values(counts).reduce((a, b) => a + b, 0);
        await polls(page);
        const fetched = Object.values(counts).reduce((a, b) => a + b, 0) - n0;
        assertCalm(await watched(page), fetched);
        const after = await page.evaluate(() => ({
          y: scrollY, picked: document.querySelector('.mo-col[aria-pressed="true"]').getAttribute('data-key'),
          read: document.querySelector('.mo-read .kicker').textContent, focusKept: document.activeElement === window.__focus,
        }));
        assert.equal(after.picked, before.picked, 'the picked month changed');
        assert.equal(after.read, before.read, 'the readout shows another month');
        assert.equal(after.y, before.y, 'the window scrolled');
        assert.ok(after.focusKept, 'the focus moved');
        assert.deepEqual(errors, []);
        await context.close();
      });
    }

    for (const width of [1440, 390]) {
      await check(`lab ${width}: polls keep open details, the table scroll and the focus`, async () => {
        const {page, errors, context, counts} = await open(browser, {viewport: {width, height: 900}, url: '/next?mock=1#lab'});
        await settle(page);
        await page.evaluate(() => {
          for (const d of document.querySelectorAll('#view details')) d.open = true;
          for (const t of document.querySelectorAll('#view .tbl-scroll')) t.scrollLeft = Math.min(30, t.scrollWidth - t.clientWidth);
          document.querySelector('#view .seg button').focus();
          window.__focus = document.activeElement;
        });
        await scrollTo(page, 400);
        // Let the opened details lay out before the watch starts.
        await page.clock.runFor(500);
        await page.waitForTimeout(100);
        const state = () => page.evaluate(() => ({
          y: scrollY, open: [...document.querySelectorAll('#view details')].map((d) => d.open),
          left: [...document.querySelectorAll('#view .tbl-scroll')].map((t) => t.scrollLeft),
        }));
        const before = await state();
        assert.ok(before.open.length >= 2, 'the fixture shows fewer than two details');
        await watch(page);
        const n0 = Object.values(counts).reduce((a, b) => a + b, 0);
        await polls(page);
        const fetched = Object.values(counts).reduce((a, b) => a + b, 0) - n0;
        assertCalm(await watched(page), fetched);
        assert.deepEqual(await state(), before, 'details closed or a table scrolled');
        assert.ok(await page.evaluate(() => document.activeElement === window.__focus), 'the focus moved');
        assert.deepEqual(errors, []);
        await context.close();
      });
    }
    await check('runway 1440: a scope switch keeps the old content, marked busy, until the work data is in', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1#runway', hold: /-work$/});
      await settle(page);
      const shot = () => page.evaluate(() => ({
        hero: document.querySelector('.p-hero .hero-v').textContent,
        text: [...document.querySelectorAll('#view .panel')].map((p) => p.innerText).join('|'),
        busy: document.getElementById('view').getAttribute('aria-busy'),
      }));
      const before = await shot();
      await watch(page);
      await page.click('button[data-scope="enterprise"]');
      await page.clock.runFor(1500);
      await page.waitForTimeout(50);
      const held = await shot();
      assert.equal(held.busy, 'true', 'the view is not marked busy');
      assert.equal(held.text, before.text, 'the old content changed before the work data was in');
      page.__release();
      await page.clock.runFor(500);
      await page.waitForTimeout(100);
      const after = await shot();
      assert.equal(after.busy, null, 'the view stays busy');
      assert.notEqual(after.hero, before.hero, 'the work data did not replace the old content');
      assert.equal((await watched(page)).loading, 0, 'a "Loading" line showed');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('ledger 1440: a view switch shows the prefetched data at once', async () => {
      const {page, errors, context, counts} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1#runway'});
      await settle(page);
      await page.clock.runFor(5000);
      await page.waitForTimeout(100);
      assert.ok(counts.ledger >= 1, 'the ledger was not prefetched');
      const n = counts.ledger;
      await page.click('[data-view-link="ledger"]');
      await page.clock.runFor(50);
      await page.waitForTimeout(30);
      const text = await page.evaluate(() => document.getElementById('view').innerText);
      assert.doesNotMatch(text, /Loading/, 'the ledger says "Loading"');
      assert.match(text, /\$/, 'the ledger shows no amounts');
      assert.equal(counts.ledger, n, 'the switch read the fresh ledger again');
      assert.deepEqual(errors, []);
      await context.close();
    });
  } finally {
    await browser.close();
  }
  if (failures.length) {
    console.log('\n' + failures.length + ' failure(s)');
    process.exit(1);
  }
})().catch(error => { console.error(error); process.exit(1); });
