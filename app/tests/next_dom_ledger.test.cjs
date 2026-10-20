// Browser checks for the /next Ledger view. Loads the rendered shell in
// headless Chromium; page.route serves static files from the repo, so
// ?mock=1 reads the committed fixtures (ledger.json and
// fixtures/ledger-sessions/<id>.json). No server, network or production data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.TOKENFOLD_PLAYWRIGHT || 'playwright');

const ROOT = path.resolve(__dirname, '..', '..');
const ORIGIN = 'http://tokenfold.test';
const HTML = fs.readFileSync(process.env.TOKENFOLD_HTML_NEXT, 'utf8');
// The page's real Content-Security-Policy, so a check fails on a violation
// (Chromium logs each one as a console error).
const CSP = process.env.TOKENFOLD_CSP_NEXT || '';
const failures = [];
async function check(name, action) {
  try { await action(); console.log('ok - ' + name); }
  catch (error) { failures.push(name + ': ' + error.message); console.log('FAIL - ' + name + '\n  ' + error.message); }
}

async function open(browser, {viewport, url, api404 = false, ledger = null}) {
  const context = await browser.newContext({viewport, timezoneId: 'America/Chicago'});
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => {
    if (m.type() === 'error' && !(api404 && /status of 404/.test(m.text()))) errors.push(m.text());
  });
  await page.route('**/*', route => {
    const u = new URL(route.request().url());
    if (u.origin !== ORIGIN) return route.fulfill({status: 204, body: ''});
    if (u.pathname === '/next') return route.fulfill({contentType: 'text/html', body: HTML, headers: CSP ? {'Content-Security-Policy': CSP} : {}});
    // The shell serves static/next under /static/next-<version>/ (cache busting).
    const pathname = u.pathname.replace(/^\/static\/next-[0-9a-f]+\//, '/static/next/');
    if (ledger && pathname.endsWith('/fixtures/ledger.json')) {
      return route.fulfill({json: ledger(JSON.parse(fs.readFileSync(path.join(ROOT, pathname), 'utf8')))});
    }
    if (pathname.startsWith('/static/')) return route.fulfill({path: path.join(ROOT, pathname)});
    if (u.pathname.startsWith('/api/') && api404) return route.fulfill({status: 404, json: {detail: 'Not Found'}});
    return route.fulfill({status: 404, body: ''});
  });
  await page.goto(ORIGIN + url);
  return {page, errors, context};
}

const MOCK = '/next?mock=1&scope=personal#ledger';
const text = (page, sel) => page.locator(sel).first().innerText();
const stat = (page, i) => page.locator('.p-ledger-stats .ls .ls-v').nth(i).innerText();
const ready = page => page.waitForSelector('.p-ledger-pivot tbody tr .pv-name');
const noOverflow = page => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);

(async () => {
  const browser = await chromium.launch();
  try {
    await check('work ledger: a day the billing meter billed more than the sessions is named', async () => {
      const gaps = (l) => ({...l, meter_gaps: [{day: '2026-10-14', billed: 400, recorded: 172}, {day: '2026-09-01', billed: 90, recorded: 0}]});
      for (const width of [1440, 390]) {
        const {page, errors, context} = await open(browser, {viewport: {width, height: 900}, url: MOCK, ledger: gaps});
        await ready(page);
        // Sep 1 is outside the 7-day period, so only Oct 14 counts.
        assert.equal(await text(page, '.ls-gap'), 'The billing meter shows $228 more than Tokenfold recorded on Oct 14.');
        assert.ok(await noOverflow(page), `the page scrolls sideways at ${width}`);
        if (width === 1440) {
          // The meter covers every machine, so a filtered view says nothing.
          await page.locator('.p-ledger-pivot tbody tr .pv-name').first().click();
          await page.waitForSelector('.ls-gap', {state: 'hidden'});
        }
        assert.deepEqual(errors, []);
        await context.close();
      }
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: MOCK});
      await ready(page);
      assert.equal(await page.locator('.ls-gap').isHidden(), true, 'no meter gaps, no line');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('desktop mock: every Ledger region renders from fixtures', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: MOCK});
      await ready(page);
      for (const id of ['ledger-stats', 'ledger-ctl', 'ledger-overview', 'ledger-split', 'ledger-timeline', 'ledger-pivot', 'ledger-top']) {
        assert.equal(await page.locator(`[data-panel="${id}"]`).count(), 1, id);
      }
      assert.match(await text(page, '.ls-period'), /Last 7 days/i);
      assert.match(await stat(page, 0), /\$1,464/);
      assert.match(await text(page, '.lc-quiet'), /\d+ quiet sessions hidden/i);
      assert.ok(await page.locator('.lo-wrap svg rect').count() > 30, 'daily bars drawn');
      assert.ok(await page.locator('.lo-legend .lg-prov').count() >= 3, 'legend groups by provider');
      assert.match(await text(page, '.p-ledger-split .ws-pct'), /^\d+%$/);
      // Every view stylesheet is linked: the split bar must keep the ledger rule, not another view's.
      assert.equal(await page.locator('.p-ledger-split .ws-bar').evaluate((e) => getComputedStyle(e).display), 'flex', 'the split bar lost its ledger style');
      assert.ok(await page.locator('.tl-lane').count() >= 3, 'machine swimlanes');
      assert.ok(await page.locator('.tl-sess').count() > 10, 'session bars');
      assert.equal(await page.locator('.lt-item').count(), 8);
      // The shell links the view stylesheet (no late restyle) from the versioned static path, no ?v=.
      const css = await page.locator('#ledger-css').evaluate((l) => l.href);
      assert.match(new URL(css).pathname + new URL(css).search, /^\/static\/next-[0-9a-f]+\/css\/ledger\.css$/);
      // The cost note appears once in the view.
      const body = await page.evaluate(() => document.body.innerText);
      assert.equal((body.match(/not billed/g) || []).length, 1, 'one "not billed" note');
      assert.equal(await noOverflow(page), true, 'page scrolls sideways');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('quiet toggle: showing quiet sessions adds them to the count', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: MOCK});
      await ready(page);
      const before = Number((await stat(page, 1)).replace(/,/g, ''));
      const cost = await stat(page, 0);
      const hidden = Number((await text(page, '.lc-quiet')).match(/(\d+)/)[1]);
      await page.click('.lc-quiet');
      await page.waitForFunction(() => /^Hide \d+ quiet/.test(document.querySelector('.lc-quiet').textContent));
      assert.equal(Number((await stat(page, 1)).replace(/,/g, '')), before + hidden);
      assert.equal(await stat(page, 0), cost, 'dollars always include quiet sessions');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('filters: a pivot row adds a chip that narrows the stats; × removes it', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: MOCK});
      await ready(page);
      const all = await stat(page, 0);
      const row = page.locator('.p-ledger-pivot tbody tr').first();
      const name = await row.locator('.pv-name').innerText();
      const cost = await row.locator('td.c-cost').innerText();
      await row.locator('.pv-name').click();
      await page.waitForSelector('.lc-chip');
      assert.match(await text(page, '.lc-chip .lc-val'), new RegExp(name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
      assert.equal((await stat(page, 0)).replace(/^≈/, ''), cost);
      assert.equal(await row.locator('.pv-name').getAttribute('aria-pressed'), 'true');
      assert.match(await text(page, '.ls-period'), /filtered/i);
      await page.click('.lc-chip button');
      await page.waitForFunction(() => !document.querySelector('.lc-chip'));
      assert.equal(await stat(page, 0), all);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('period: Today, 30D and a drag on the chart set the period', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: MOCK});
      await ready(page);
      await page.click('.p-ledger-ctl [data-value="30d"]');
      await page.waitForFunction(() => /Last 30 days/.test(document.querySelector('.ls-period').textContent));
      const month = await stat(page, 0);
      await page.click('.p-ledger-ctl [data-value="today"]');
      await page.waitForFunction(() => /Today so far/.test(document.querySelector('.ls-period').textContent));
      assert.notEqual(await stat(page, 0), month);
      const box = await page.locator('.lo-hit').boundingBox();
      const y = box.y + box.height / 2;
      await page.mouse.move(box.x + box.width * 0.2, y);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width * 0.35, y, {steps: 6});
      await page.mouse.up();
      await page.waitForFunction(() => /–/.test(document.querySelector('.ls-period').textContent));
      assert.equal(await page.getAttribute('.p-ledger-ctl [data-value="custom"]', 'aria-pressed'), 'true');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('breakdown: headers sort and every dimension renders', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: MOCK});
      await ready(page);
      const nums = () => page.$$eval('.p-ledger-pivot tbody tr', rows => rows.map(r => Number(r.children[3].textContent.replace(/,/g, ''))));
      await page.click('.p-ledger-pivot [data-sort="sessions"]');
      assert.equal(await page.getAttribute('.p-ledger-pivot th:has([data-sort="sessions"])', 'aria-sort'), 'descending');
      let v = await nums();
      assert.deepEqual(v, [...v].sort((a, b) => b - a));
      await page.click('.p-ledger-pivot [data-sort="sessions"]');
      assert.equal(await page.getAttribute('.p-ledger-pivot th:has([data-sort="sessions"])', 'aria-sort'), 'ascending');
      v = await nums();
      assert.deepEqual(v, [...v].sort((a, b) => a - b));
      for (const dim of ['model', 'machine', 'tool', 'day']) {
        await page.click(`.p-ledger-pivot .seg [data-value="${dim}"]`);
        assert.ok(await page.locator('.p-ledger-pivot tbody tr .pv-name').count() > 0, dim);
      }
      assert.match(await text(page, '.p-ledger-pivot thead'), /Top model/i);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('biggest sessions: an untitled session takes the server label', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: MOCK});
      await ready(page);
      const names = await page.locator('.lt-title').allInnerTexts();
      assert.ok(names.length > 0);
      assert.ok(!names.some((n) => /Untitled/.test(n)), names.join(' | '));
      assert.ok(names.some((n) => /^Subagents: /.test(n)), names.join(' | '));
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('timeline: one session is a tab stop; arrow keys move between sessions', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: MOCK});
      await ready(page);
      assert.equal(await page.locator('.tl-sess[tabindex="0"]').count(), 1, 'one tab stop');
      const first = page.locator('.tl-sess[tabindex="0"]');
      const id0 = await first.getAttribute('data-sid');
      await first.focus();
      await page.keyboard.press('ArrowRight');
      const id1 = await page.evaluate(() => document.activeElement.dataset.sid);
      assert.ok(id1 && id1 !== id0, 'ArrowRight moves focus to another session');
      assert.equal(await page.locator('.tl-sess[tabindex="0"]').count(), 1, 'still one tab stop');
      assert.equal(await page.locator('.tl-sess[tabindex="0"]').getAttribute('data-sid'), id1);
      await page.keyboard.press('ArrowLeft');
      assert.equal(await page.evaluate(() => document.activeElement.dataset.sid), id0);
      await page.keyboard.press('Enter');
      await page.waitForSelector('.drawer.is-open');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('drawer: opens with detail, traps focus, walks prev/next, Esc returns focus', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: MOCK});
      await ready(page);
      const opener = page.locator('.lt-btn').first();
      const title = await opener.locator('.lt-title').innerText();
      await opener.click();
      await page.waitForSelector('.drawer.is-open');
      await page.waitForSelector('.drawer .served-word');
      assert.equal(await text(page, '.dr-title'), title.replace(/\s*Live$/i, ''));
      // The fan-out chart draws on the next animation frame (ledger-fanout.js).
      await page.waitForSelector('.drawer .gantt svg', {timeout: 5000});
      assert.ok(await page.locator('.drawer .gantt svg').count() === 1, 'fan-out chart');
      assert.ok(await page.locator('.drawer .tree li').count() >= 2, 'run tree');
      assert.ok(await page.locator('.drawer .mix-bar i').count() >= 1, 'model mix');
      assert.ok(await page.locator('.drawer .tok-row').count() >= 4, 'tokens');
      assert.equal(await page.evaluate(() => document.querySelector('.drawer').contains(document.activeElement)), true);
      for (let i = 0; i < 25; i += 1) await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => document.querySelector('.drawer').contains(document.activeElement)), true, 'Tab stays in the drawer');
      await page.keyboard.press('Shift+Tab');
      assert.equal(await page.evaluate(() => document.querySelector('.drawer').contains(document.activeElement)), true, 'Shift+Tab stays in the drawer');
      const next = page.locator('.dr-nav [aria-label="Next session"]');
      const prev = page.locator('.dr-nav [aria-label="Previous session"]');
      const go = (await next.isDisabled()) ? prev : next;
      await go.click();
      await page.waitForFunction(t => document.querySelector('.dr-title').textContent !== t, title);
      await page.keyboard.press('Escape');
      await page.waitForSelector('.drawer', {state: 'hidden'});
      assert.equal(await page.evaluate(() => document.activeElement && document.activeElement.classList.contains('lt-btn')), true, 'focus returns to the list');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('phone mock: summary, controls, chart, split, list, timeline, breakdown in that order', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 390, height: 844}, url: MOCK});
      await ready(page);
      const tops = await page.evaluate(() => ['stats', 'ctl', 'overview', 'split', 'top', 'timeline', 'pivot']
        .map(id => document.querySelector(`[data-panel="ledger-${id}"]`).getBoundingClientRect().top));
      assert.deepEqual([...tops].sort((a, b) => a - b), tops);
      assert.equal(await noOverflow(page), true, 'page scrolls sideways');
      await page.locator('.lt-btn').first().click();
      await page.waitForSelector('.drawer .served-word');
      assert.equal(await noOverflow(page), true, 'drawer scrolls sideways');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('real mode without endpoints: the Ledger says unavailable, the page stays up', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?scope=personal#ledger', api404: true});
      await page.waitForFunction(() => /unavailable/i.test((document.querySelector('[data-panel="ledger-stats"]') || {}).textContent || ''));
      assert.match(await text(page, '[data-panel="ledger-overview"]'), /unavailable/i);
      assert.equal(await page.locator('[data-view-link="runway"]').count(), 1);
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
