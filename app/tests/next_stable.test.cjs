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
// TOKENFOLD_ONLY=<regex> runs only the checks whose name matches.
const ONLY = process.env.TOKENFOLD_ONLY ? new RegExp(process.env.TOKENFOLD_ONLY) : null;
async function check(name, action) {
  if (ONLY && !ONLY.test(name)) return;
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
// boot: the scope the page opens in (the rendered shell is personal).
// holdStatic: static paths that wait for page.__release() like held fixtures.
// mutate(name, body, page): change a fixture body before it is served.
// noMoveBefore: run as a browser without Element.moveBefore.
async function open(browser, {viewport, url, hold = null, scope = 'personal', boot = 'personal', holdStatic = null, waitUntil = 'load',
  mutate = null, noMoveBefore = false}) {
  const context = await browser.newContext({viewport, timezoneId: 'America/Chicago'});
  const page = await context.newPage();
  if (noMoveBefore) await page.addInitScript(() => { delete Element.prototype.moveBefore; });
  const errors = [];
  const counts = {};
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.route('**/*', async route => {
    const u = new URL(route.request().url());
    if (u.origin !== ORIGIN) return route.fulfill({status: 204, body: ''});
    const html = boot === 'personal' ? HTML : HTML.replace('data-scope="personal"', `data-scope="${boot}"`);
    if (u.pathname === '/next') return route.fulfill({contentType: 'text/html', body: html, headers: CSP ? {'Content-Security-Policy': CSP} : {}});
    const pathname = u.pathname.replace(/^\/static\/next-[0-9a-f]+\//, '/static/next/');
    const fixture = pathname.match(/^\/static\/next\/fixtures\/(.+)\.json$/);
    if (fixture && fs.existsSync(path.join(FIXTURES, fixture[1] + '.json'))) {
      // A held fixture answers when the check calls page.__release().
      if (page.__hold && page.__hold.test(fixture[1])) await new Promise((resolve) => page.__held.push({name: fixture[1], resolve}));
      const n = (counts[fixture[1]] = (counts[fixture[1]] || 0) + 1);
      const body = JSON.parse(fs.readFileSync(path.join(FIXTURES, fixture[1] + '.json'), 'utf8'));
      const out = n === 1 ? body : drift(body, null, n);
      return route.fulfill({json: mutate ? mutate(fixture[1], out, page) : out});
    }
    if (pathname === '/static/next/js/main.js') {
      const text = fs.readFileSync(path.join(ROOT, pathname), 'utf8');
      return route.fulfill({contentType: 'text/javascript', body: `${text}\ndocument.addEventListener('tf:bump', () => feeds.bump());\n`});
    }
    if (holdStatic && holdStatic.test(pathname) && page.__holding) await new Promise((resolve) => page.__held.push({name: pathname, resolve}));
    if (pathname.startsWith('/static/')) return route.fulfill({path: path.join(ROOT, pathname)});
    return route.fulfill({status: 404, body: ''});
  });
  page.__held = [];
  page.__holding = true;
  page.__hold = hold;
  page.__release = () => { page.__holding = false; page.__hold = null; for (const r of page.__held.splice(0)) r.resolve(); };
  // Keep holding only the fixtures that match `re`; answer the others.
  page.__holdOnly = (re) => {
    page.__hold = re;
    const keep = page.__held.filter((r) => re.test(r.name));
    for (const r of page.__held.filter((x) => !keep.includes(x))) r.resolve();
    page.__held = keep;
  };
  await page.clock.install({time: START});
  await page.goto(ORIGIN + url, {waitUntil});
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

// Watch layout shifts, removed nodes and any "Loading" text from now on, in
// the view, the masthead and the live bar. removedWhat: the first removed
// elements, for the failure message.
const WATCHED = ['#view', '.masthead', '#livebar'];
const watch = (page) => page.evaluate((roots) => {
  const w = window.__watch = {cls: 0, removed: 0, removedWhat: [], loading: 0,
    nodes: [...document.querySelectorAll(roots.map((r) => `${r} *`).join(', '))]};
  new PerformanceObserver((l) => { for (const e of l.getEntries()) if (!e.hadRecentInput) w.cls += e.value; })
    .observe({type: 'layout-shift'});
  const mo = new MutationObserver((ms) => {
    for (const m of ms) {
      for (const n of m.removedNodes) {
        if (n.nodeType !== 1) continue;
        w.removed += 1;
        if (w.removedWhat.length < 50) w.removedWhat.push(`${m.target.className || m.target.localName} > ${n.localName}.${n.className || ''}`);
      }
    }
    if ([...document.querySelectorAll('#view .panel:not([hidden])')].some((p) => /Loading/.test(p.innerText))) w.loading += 1;
  });
  for (const r of roots) mo.observe(document.querySelector(r), {childList: true, subtree: true});
}, WATCHED);

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
  return {cls: w.cls, removed: w.removed, removedWhat: w.removedWhat, loading: w.loading,
    lost: w.nodes.filter((n) => !n.isConnected).length, total: w.nodes.length};
});

function assertCalm(r, fetched) {
  assert.ok(fetched >= 3, `only ${fetched} responses changed the data`);
  assert.ok(r.cls < 0.01, `layout shift ${r.cls.toFixed(4)}`);
  assert.equal(r.loading, 0, 'a "Loading" line came back');
  assert.equal(r.lost, 0, `${r.lost} of ${r.total} nodes were replaced`);
  // Data of the same shape adds and removes no node, also none made after the watch began.
  assert.equal(r.removed, 0, `${r.removed} nodes were removed: ${r.removedWhat.join(', ')}`);
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
    await check('ledger 1440: "Show all N runs" works after the detail redraw and stays open over polls', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1#ledger'});
      await settle(page);
      // The first of the biggest sessions with more runs than the tree shows.
      let found = false;
      for (let i = 0; i < 8 && !found; i += 1) {
        await page.locator('.lt-btn').nth(i).click();
        await page.clock.runFor(500);
        await page.waitForSelector('.drawer.is-open .served-word');
        found = await page.locator('.drawer .tree-more').count() > 0;
        if (!found) { await page.keyboard.press('Escape'); await page.clock.runFor(500); }
      }
      assert.ok(found, 'no session of the list has a folded run tree');
      const label = await page.locator('.drawer .tree-more').innerText();
      const all = Number(label.match(/\d+/)[0]);
      await page.locator('.drawer .tree-more').click();
      await page.clock.runFor(100);
      const rows = () => page.locator('.drawer .tree li').count();
      assert.equal(await rows(), all + 1, 'the click did not show every run');
      assert.equal(await page.locator('.drawer .tree-more').count(), 0, 'the button stays');
      await polls(page, 150);
      assert.equal(await rows(), all + 1, 'a poll folded the run tree again');
      assert.equal(await page.locator('.drawer .tree-more').count(), 0, 'the button came back');
      assert.deepEqual(errors, []);
      await context.close();
    });

    // The top project of the fixture drops to the last row when its sessions
    // cost a ten-thousandth, and comes back. The focused row button keeps the focus.
    const COST = /cost_usd$/;
    const cheap = (v) => (Array.isArray(v) ? v.map(cheap) : v && typeof v === 'object'
      ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, COST.test(k) && typeof x === 'number' ? x / 1e4 : k === 'orch_models' ? Object.fromEntries(Object.entries(x).map(([m, c]) => [m, c / 1e4])) : cheap(x)]))
      : v);
    const sinkTop = (name, body, page) => (name !== 'ledger' || !page.__sink ? body
      : {...body, sessions: body.sessions.map((x) => (x.project === 'agent-tools' ? cheap(x) : x))});
    for (const noMoveBefore of [false, true]) {
      await check(`ledger 1440${noMoveBefore ? ' without moveBefore' : ''}: a pivot row that moves keeps the focus`, async () => {
        const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1#ledger', mutate: sinkTop, noMoveBefore});
        await settle(page);
        const keys = () => page.evaluate(() => [...document.querySelectorAll('.p-ledger-pivot tbody tr')].map((r) => r.dataset.key));
        const first = (await keys())[0];
        assert.equal(first, 'project:agent-tools', 'the fixture has another top project');
        await page.evaluate(() => {
          const b = document.querySelector('.p-ledger-pivot tbody tr button.pv-name');
          b.focus();
          window.__focus = b;
        });
        for (const sink of [true, false]) {
          page.__sink = sink;
          await page.evaluate(() => document.dispatchEvent(new Event('tf:bump')));
          await page.clock.runFor(11000);
          await page.waitForTimeout(100);
          const order = await keys();
          assert.equal(sink ? order[order.length - 1] : order[0], 'project:agent-tools', `the row did not move: ${order.join(', ')}`);
          assert.ok(await page.evaluate(() => document.activeElement === window.__focus), 'the focus left the moved row');
        }
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
      await page.evaluate(() => { window.__frame = [...document.querySelectorAll('#view .panel, #view .panel-head, #view h2')]; });
      await page.click('button[data-scope="enterprise"]');
      await page.clock.runFor(1500);
      await page.waitForTimeout(50);
      const held = await shot();
      assert.equal(held.busy, 'true', 'the view is not marked busy');
      assert.equal(held.text, before.text, 'the old content changed before the work data was in');
      page.__release();
      // The trajectory refetch for the work window is a fake-clock timer: run
      // the clock in small steps until the switch ends, as settle() does.
      for (let i = 0; i < 300 && (await shot()).busy; i += 1) {
        await page.clock.runFor(50);
        await page.waitForTimeout(5);
      }
      const after = await shot();
      assert.equal(after.busy, null, 'the view stays busy');
      assert.notEqual(after.hero, before.hero, 'the work data did not replace the old content');
      const r = await watched(page);
      assert.equal(r.loading, 0, 'a "Loading" line showed');
      // The work data has other rows (fewer windows, models and machines), so
      // rows go; the frame of every panel stays. The fixtures need 26 removed
      // rows and parts; far more means a panel drew again from scratch.
      assert.ok(r.removed > 0 && r.removed <= 40, `${r.removed} nodes were removed: ${r.removedWhat.join(', ')}`);
      assert.deepEqual(r.removedWhat.filter((x) => /(^|> )(section|h2)\b|\.(panel|panel-head|grid)\b/.test(x)), [], 'a panel frame was removed');
      assert.ok(await page.evaluate(() => window.__frame.every((n) => n.isConnected)), 'a panel or its title was replaced');
      assert.deepEqual(errors, []);
      await context.close();
    });

    // Today of the personal fixture names a machine the work fixture does not
    // have ("devbox"). Leave the view during the switch, let the switch end
    // while today of the work scope still waits, come back: the personal
    // today must never show while the view is not busy. Again with the tab
    // hidden, where the idle prefetch does not run.
    for (const hidden of [false, true]) {
      await check(`runway 1440${hidden ? ' hidden tab' : ''}: a view switch during a scope switch never shows old scope data`, async () => {
        const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1#runway', hold: /-work$/});
        await settle(page);
        assert.match(await page.locator('.p-today').innerText(), /devbox/, 'the personal fixture changed');
        await page.evaluate(() => {
          const w = window.__leak = {seen: 0, texts: []};
          const look = () => {
            const work = document.querySelector('button[data-scope="enterprise"]').getAttribute('aria-pressed') === 'true';
            const busy = document.getElementById('view').getAttribute('aria-busy') === 'true';
            const today = document.querySelector('#view .p-today');
            if (work && !busy && today && /devbox/.test(today.innerText)) { w.seen += 1; w.texts.push(today.innerText.slice(0, 80)); }
          };
          new MutationObserver(look).observe(document.getElementById('view'), {childList: true, subtree: true, characterData: true, attributes: true});
        });
        await page.click('button[data-scope="enterprise"]');
        await page.clock.runFor(200);
        await page.waitForTimeout(30);
        await page.click('[data-view-link="ledger"]');
        await page.clock.runFor(200);
        await page.waitForTimeout(30);
        // Every work read answers but today's.
        page.__holdOnly(/^today-work$/);
        for (let i = 0; i < 100; i += 1) {
          await page.clock.runFor(50);
          await page.waitForTimeout(5);
          if (await page.evaluate(() => !document.getElementById('view').hasAttribute('aria-busy'))) break;
        }
        assert.equal(await page.evaluate(() => document.getElementById('view').getAttribute('aria-busy')), null, 'the switch did not end');
        if (hidden) {
          await page.evaluate(() => {
            Object.defineProperty(document, 'hidden', {configurable: true, get: () => true});
            document.dispatchEvent(new Event('visibilitychange'));
          });
        }
        await page.clock.runFor(5000);
        await page.waitForTimeout(30);
        await page.click('[data-view-link="runway"]');
        await page.clock.runFor(200);
        await page.waitForTimeout(30);
        assert.doesNotMatch(await page.locator('.p-today').innerText(), /devbox/, 'the personal today shows in the work scope');
        if (hidden) {
          await page.evaluate(() => {
            delete document.hidden;
            document.dispatchEvent(new Event('visibilitychange'));
          });
        }
        page.__release();
        await settle(page);
        const text = await page.locator('.p-today').innerText();
        assert.doesNotMatch(text, /devbox/, 'the personal today shows in the work scope');
        assert.match(text, /laptop/, 'the work today did not show');
        const leak = await page.evaluate(() => window.__leak);
        assert.equal(leak.seen, 0, `old scope data showed: ${leak.texts.join(' / ')}`);
        assert.deepEqual(errors, []);
        await context.close();
      });
    }

    const EARLY = {
      runway: {personal: ['now', 'runway', 'today', 'limit-rate'], enterprise: ['now-work', 'runway-work', 'today-work']},
      ledger: {personal: ['now', 'runway', 'ledger'], enterprise: ['now-work', 'runway-work', 'ledger']},
      history: {personal: ['now', 'runway', 'window-history', 'spend-history'],
        enterprise: ['now-work', 'runway-work', 'spend-history-work', 'stats-work', 'enterprise-budget']},
      lab: {personal: ['now', 'runway', 'stats', 'served-timeline'], enterprise: ['now-work', 'runway-work', 'stats-work']},
    };
    for (const view of Object.keys(EARLY)) {
      for (const boot of ['personal', 'enterprise']) {
        await check(`${view} ${boot}: the first reads start before main.js loads, and none runs twice`, async () => {
          const {page, errors, context, counts} = await open(browser, {
            viewport: {width: 1440, height: 900}, url: `/next?mock=1#${view}`, boot, holdStatic: /\/js\/main\.js$/, waitUntil: 'commit'});
          for (let i = 0; i < 100 && EARLY[view][boot].some((k) => !counts[k]); i += 1) await page.waitForTimeout(20);
          assert.deepEqual(EARLY[view][boot].filter((k) => !counts[k]), [], 'early reads missing while main.js waits');
          page.__release();
          await page.waitForLoadState('load');
          await settle(page);
          // The ledger view reads the ledger fixture twice by design (30 days and the period).
          const twice = Object.entries(counts).filter(([k, n]) => n > (view === 'ledger' && k === 'ledger' ? 2 : 1));
          assert.deepEqual(twice, [], 'a boot read ran twice');
          assert.deepEqual(errors, []);
          await context.close();
        });
      }
    }

    // Live mode (no mock): the API answers 404, but each early read must match
    // its feed's URL, query included, so the feed takes it.
    for (const view of Object.keys(EARLY)) {
      for (const boot of ['personal', 'enterprise']) {
        await check(`${view} ${boot} live: every early read is taken by its feed`, async () => {
          const {page, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: `/next#${view}`, boot});
          await page.clock.runFor(1000);
          await page.waitForTimeout(100);
          const left = await page.evaluate(async () => (await import(`${document.body.dataset.static}/js/api.js`)).pendingEarly());
          assert.deepEqual(left, [], 'an early read went unused');
          await context.close();
        });
      }
    }

    // The browser's own restoration runs while the page is one screen of
    // "Loading" lines. The position must come back once the data is in.
    for (const [view, width] of [['ledger', 1440], ['runway', 390], ['history', 1440]]) {
      await check(`${view} ${width}: a reload puts the scroll position back after the data loads`, async () => {
        const {page, errors, context} = await open(browser, {viewport: {width, height: 900}, url: `/next?mock=1#${view}`});
        await settle(page);
        const y = await page.evaluate(() => Math.min(700, document.documentElement.scrollHeight - innerHeight));
        assert.ok(y > 200, `the ${view} page is too short to test (${y})`);
        await scrollTo(page, y);
        await page.clock.runFor(300);
        page.__hold = /./;
        await page.reload();
        await page.clock.runFor(300);
        await page.waitForTimeout(30);
        assert.equal(await page.evaluate(() => scrollY), 0, 'the page was not loading');
        page.__release();
        await settle(page);
        assert.equal(await page.evaluate(() => Math.round(scrollY)), y, `the scroll position did not come back (${await page.evaluate(() => [scrollY, document.documentElement.scrollHeight, JSON.stringify(history.state)].join(' '))})`);
        assert.deepEqual(errors, []);
        await context.close();
      });
    }

    // While a view loads, each panel keeps a reserved height: a default on the
    // first visit, the height it had at the last load after that.
    for (const view of ['runway', 'ledger', 'history', 'lab']) {
      for (const width of [1440, 390]) {
        await check(`${view} ${width}: the panels keep their place while the data loads`, async () => {
          const {page, errors, context} = await open(browser, {viewport: {width, height: 900}, url: `/next?mock=1#${view}`, hold: /./});
          await page.evaluate(() => {
            window.__shift = 0;
            new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__shift += e.value; }).observe({type: 'layout-shift'});
          });
          await page.clock.runFor(300);
          await page.waitForTimeout(30);
          const loading = await page.evaluate(() => [...document.querySelectorAll('#view .panel:not([hidden])')]
            .filter((p) => p.offsetHeight && !p.querySelector('[data-loading]')).map((p) => p.dataset.panel));
          // The ledger bar keeps a marker too; its period buttons need no data.
          assert.deepEqual(loading, [], 'panels with no loading state while the data loads');
          page.__release();
          await settle(page);
          const first = await page.evaluate(() => window.__shift);
          assert.ok(first < 0.15, `first load: layout shift ${first.toFixed(3)} with the default heights`);
          await page.clock.runFor(2000);
          page.__hold = /./;
          await page.reload();
          await page.evaluate(() => {
            window.__shift = 0;
            new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__shift += e.value; }).observe({type: 'layout-shift'});
          });
          await page.clock.runFor(300);
          await page.waitForTimeout(30);
          page.__release();
          await settle(page);
          const again = await page.evaluate(() => window.__shift);
          assert.ok(again < 0.02, `second load: layout shift ${again.toFixed(3)} with the recorded heights`);
          assert.deepEqual(errors, []);
          await context.close();
        });
      }
    }

    await check('ledger 1440: a wheel input during the load cancels the scroll restore', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1#ledger'});
      await settle(page);
      await scrollTo(page, 700);
      await page.clock.runFor(300);
      page.__hold = /./;
      await page.reload();
      await page.clock.runFor(300);
      await page.mouse.move(700, 500);
      await page.mouse.wheel(0, -100);
      page.__release();
      await settle(page);
      assert.equal(await page.evaluate(() => scrollY), 0, 'the restore moved the page after the user scrolled');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('runway 1440: a view tab opens at the top; back and forward put each position back', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1#runway'});
      await settle(page);
      await page.clock.runFor(5000);
      await scrollTo(page, 400);
      await page.clock.runFor(300);
      // element.click(): a Playwright click scrolls the masthead link into view first.
      await page.evaluate(() => document.querySelector('[data-view-link="ledger"]').click());
      await page.clock.runFor(100);
      assert.equal(await page.evaluate(() => location.hash), '#ledger');
      assert.equal(await page.evaluate(() => scrollY), 0, 'the ledger did not open at the top');
      await settle(page);
      await scrollTo(page, 600);
      await page.clock.runFor(300);
      await page.goBack();
      await page.clock.runFor(200);
      await page.waitForTimeout(30);
      assert.equal(await page.evaluate(() => location.hash), '#runway');
      assert.equal(await page.evaluate(() => Math.round(scrollY)), 400, 'back did not put the runway position back');
      await page.goForward();
      await page.clock.runFor(200);
      await page.waitForTimeout(30);
      assert.equal(await page.evaluate(() => Math.round(scrollY)), 600, 'forward did not put the ledger position back');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('runway 1440: Back from a tab to the page as opened shows the default view at its position', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1'});
      await settle(page);
      await page.clock.runFor(5000);
      await scrollTo(page, 350);
      await page.waitForTimeout(30);
      await page.evaluate(() => document.querySelector('[data-view-link="ledger"]').click());
      await page.clock.runFor(100);
      assert.equal(await page.evaluate(() => document.querySelector('#view > div').dataset.view), 'ledger');
      await page.goBack();
      await page.clock.runFor(200);
      await page.waitForTimeout(30);
      assert.equal(await page.evaluate(() => document.querySelector('#view > div').dataset.view), 'runway', 'Back did not show the default view');
      assert.equal(await page.evaluate(() => Math.round(scrollY)), 350, 'Back did not put the position back');
      assert.deepEqual(errors, []);
      await context.close();
    });

    for (const width of [1440, 390]) {
      await check(`ledger ${width}: the "loading this period" note adds no height`, async () => {
        const {page, errors, context} = await open(browser, {viewport: {width, height: 900}, url: '/next?mock=1#ledger'});
        await settle(page);
        // Mock mode always has the exact period, so the note is set by hand.
        const heights = await page.evaluate(() => {
          const panel = document.querySelector('.p-ledger-stats');
          const before = panel.offsetHeight;
          document.querySelector('.ls-busy').textContent = 'Loading this period';
          return [before, panel.offsetHeight];
        });
        assert.equal(heights[1], heights[0], 'the note changed the summary height');
        assert.deepEqual(errors, []);
        await context.close();
      });
    }

    await check('runway 1440: the footer starts below the fold before the panels mount', async () => {
      const {page, context} = await open(browser, {
        viewport: {width: 1440, height: 900}, url: '/next?mock=1#runway', holdStatic: /\/js\/main\.js$/, waitUntil: 'commit'});
      await page.waitForSelector('footer.foot', {state: 'attached'});
      await page.waitForTimeout(100);
      const top = await page.evaluate(() => document.querySelector('footer.foot').getBoundingClientRect().top);
      assert.ok(top >= 900, `the footer is at ${Math.round(top)} px before the panels mount`);
      page.__release();
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
