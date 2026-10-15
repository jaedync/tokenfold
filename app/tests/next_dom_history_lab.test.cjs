// Browser checks for the History and Lab views of /next. Same harness as
// next_dom.test.cjs: page.route serves static files and fixtures from the
// repo, so nothing touches a server, the network or production data.
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

async function open(browser, {viewport, url, api404 = false, fixtureOverride = null, delays = {}}) {
  const context = await browser.newContext({viewport, timezoneId: 'America/Chicago'});
  const page = await context.newPage();
  const errors = [];
  const fixtures = [];
  const apis = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => {
    if (m.type() === 'error' && !(api404 && /status of 404/.test(m.text()))) errors.push(m.text());
  });
  await page.route('**/*', async route => {
    const u = new URL(route.request().url());
    if (u.origin !== ORIGIN) return route.fulfill({status: 204, body: ''});
    if (u.pathname === '/next') return route.fulfill({contentType: 'text/html', body: HTML});
    // The shell serves static/next under /static/next-<version>/ (cache busting).
    const pathname = u.pathname.replace(/^\/static\/next-[0-9a-f]+\//, '/static/next/');
    const fixture = pathname.match(/^\/static\/next\/fixtures\/(.+)\.json$/);
    if (fixture) fixtures.push(fixture[1]);
    if (fixture && delays[fixture[1]]) await new Promise(r => setTimeout(r, delays[fixture[1]]));
    if (fixture && fixtureOverride) {
      const body = fixtureOverride(fixture[1], JSON.parse(fs.readFileSync(path.join(FIXTURES, fixture[1] + '.json'), 'utf8')));
      if (body) return route.fulfill({json: body});
    }
    if (pathname.startsWith('/static/')) return route.fulfill({path: path.join(ROOT, pathname)});
    if (u.pathname.startsWith('/api/')) apis.push(u.pathname + u.search);
    if (u.pathname.startsWith('/api/') && api404) return route.fulfill({status: 404, json: {detail: 'Not Found'}});
    return route.fulfill({status: 404, body: ''});
  });
  await page.goto(ORIGIN + url);
  return {page, errors, context, fixtures, apis};
}

const text = (page, sel) => page.locator(sel).first().innerText();
const sideways = page => page.evaluate(() => document.documentElement.scrollWidth > innerWidth);

(async () => {
  const browser = await chromium.launch();
  try {
    await check('history desktop mock: window cards, strip, months and value render', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal#history'});
      await page.waitForSelector('.p-history-windows .wc');
      await page.waitForSelector('.p-history-value .vchart');
      assert.equal(await page.locator('.wc').count(), 11);
      assert.equal(await page.locator('.wc[data-hit]').count(), 2);
      assert.equal(await page.locator('.ws-bar').count(), 11);
      assert.match(await text(page, '.wc .wc-range'), /Oct 2 to Oct 9/, 'newest window first');
      assert.equal(await page.locator('.p-history-months .mo-col').count(), 12);
      assert.match(await text(page, '.p-history-months'), /API-equivalent at list price, not billed/);
      assert.equal(await page.locator('[data-panel="history-meter"]').isHidden(), true);
      assert.equal(await page.locator('.vprice').count(), 0, 'no price line without a saved price');
      assert.equal(await sideways(page), false, 'page scrolls sideways');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('history: the 5-hour tab loads its own feed and caps the cards', async () => {
      const {page, errors, context, fixtures} = await open(browser, {viewport: {width: 1024, height: 768}, url: '/next?mock=1&scope=personal#history'});
      await page.waitForSelector('.wc');
      assert.equal(fixtures.includes('window-history-5h'), false, '5h feed asked before the tab');
      await page.getByRole('button', {name: '5-hour', exact: false}).click();
      await page.waitForFunction(() => document.querySelectorAll('.ws-bar').length === 26);
      assert.equal(await page.locator('.wc').count(), 12);
      await page.getByRole('button', {name: /Show all/}).click();
      await page.waitForFunction(() => document.querySelectorAll('.wc').length === 26);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('history value: a saved plan price adds the ratio and stays after reload', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal#history'});
      await page.waitForSelector('.p-history-value .vchart');
      await page.fill('#v-price', '200');
      await page.click('.v-form button[type="submit"]');
      await page.waitForSelector('.vprice', {state: 'attached'});
      assert.match(await text(page, '.v-sub'), /≈\d+\.\d× your \$200 plan price/);
      assert.equal(await page.locator('.v-ratio li').count(), 6);
      await page.reload();
      await page.waitForSelector('.vprice', {state: 'attached'});
      assert.equal(await page.inputValue('#v-price'), '200');
      await page.click('.v-form button[type="button"]');
      await page.waitForFunction(() => !document.querySelector('.vprice'));
      assert.equal(await page.evaluate(() => localStorage.getItem('tf_plan_price_usd')), null);
      await page.fill('#v-price', '-5');
      await page.click('.v-form button[type="submit"]');
      assert.match(await text(page, '.v-msg'), /from 1 to 100,000/);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('history: a slow window history says it is building, with a long timeout', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal#history', delays: {'window-history': 2500}});
      await page.waitForSelector('.p-history-windows .p-state');
      assert.match(await text(page, '.p-history-windows .p-state'), /Loading the window history/);
      assert.match(await text(page, '.p-history-windows .p-state'), /up to a minute/);
      await page.waitForSelector('.p-history-windows .wc');
      const timeouts = await page.evaluate(async () => {
        const api = await import(document.body.dataset.static + '/js/api.js');
        return [api.timeoutFor('windowHistory'), api.timeoutFor('windowHistory5h'), api.timeoutFor('now')];
      });
      assert.ok(timeouts[0] >= 60000 && timeouts[1] >= 60000, `window history timeouts ${timeouts}`);
      assert.equal(timeouts[2], 20000);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('history work scope: billing meter shows and says it is the limit, limit windows and value hide', async () => {
      const {page, errors, context, fixtures} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal#history'});
      await page.waitForSelector('.wc');
      fixtures.length = 0;
      await page.click('[data-scope="enterprise"]');
      await page.waitForSelector('.p-history-meter .mt-big');
      assert.match(await text(page, '.p-history-meter .mt-big'), /\$992/);
      assert.match(await text(page, '.p-history-meter'), /of \$2,000 budget/);
      assert.match(await text(page, '.p-history-meter .hl-lead'), /this meter is the limit/);
      assert.equal(await page.locator('[data-panel="history-windows"]').isHidden(), true);
      assert.equal(await page.locator('[data-panel="history-value"]').isHidden(), true);
      await page.waitForSelector('.p-history-months .mo-col');
      assert.equal(fixtures.some(n => n.startsWith('window-history')), false, 'work scope asked for limit windows');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('history: server model names render as text, not markup', async () => {
      const evil = '<img src=x onerror="window.__xss=1">';
      const override = (name, body) => {
        if (name !== 'spend-history') return null;
        const months = body.months.map(m => ({...m, by_model: {...m.by_model, [`Evil / ${evil}`]: 900}}));
        return {...body, months};
      };
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal#history', fixtureOverride: override});
      await page.waitForSelector('.p-history-months .mo-col');
      assert.equal(await page.locator('.p-history-months img').count(), 0);
      assert.match(await text(page, '.p-history-months'), /<img src=x/);
      assert.equal(await page.evaluate(() => window.__xss), undefined);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('lab desktop mock: served check, energy, price table and benchmarks render', async () => {
      const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?mock=1&scope=personal#lab'});
      await page.waitForSelector('.p-lab-served .sv-row');
      await page.waitForSelector('.p-lab-bench .hr-tbl');
      assert.equal(await page.locator('.sv-row').count(), 2);
      assert.match(await text(page, '.p-lab-energy .en-v'), /^≈\s*290/);
      assert.ok(await page.locator('.pr-tbl tbody tr').count() >= 5);
      assert.ok(await page.locator('.bn-list > li').count() >= 1);
      assert.ok(await page.locator('.hr-tbl tbody tr').count() >= 1);
      for (const id of ['lab-served', 'lab-energy', 'lab-prices', 'lab-bench']) {
        assert.ok((await text(page, `[data-panel="${id}"] .panel-sub`)).length > 20, `${id} has no lead sentence`);
      }
      assert.equal(await sideways(page), false, 'page scrolls sideways');
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('lab work scope: the served check hides and never asks', async () => {
      const {page, errors, context, fixtures} = await open(browser, {viewport: {width: 1024, height: 768}, url: '/next?mock=1&scope=personal#history'});
      await page.waitForSelector('.wc');
      await page.click('[data-scope="enterprise"]');
      await page.waitForSelector('.p-history-meter .mt-big');
      await page.click('[data-view-link="lab"]');
      await page.waitForSelector('.p-lab-prices .pr-tbl');
      assert.equal(await page.locator('[data-panel="lab-served"]').isHidden(), true);
      assert.equal(fixtures.includes('served-timeline'), false);
      await page.click('button[data-scope="personal"]');
      await page.waitForSelector('.p-lab-served .sv-row');
      assert.equal(await page.locator('[data-panel="lab-served"]').isHidden(), false);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('phone mock: History and Lab do not scroll sideways', async () => {
      for (const view of ['history', 'lab']) {
        const {page, errors, context} = await open(browser, {viewport: {width: 390, height: 844}, url: `/next?mock=1&scope=personal#${view}`});
        await page.waitForSelector(view === 'history' ? '.p-history-value .vchart' : '.p-lab-bench .hr-tbl');
        assert.equal(await sideways(page), false, `${view} scrolls sideways`);
        assert.deepEqual(errors, []);
        await context.close();
      }
    });

    await check('real mode work scope: History never asks for window history', async () => {
      const {page, errors, context, apis} = await open(browser, {viewport: {width: 1440, height: 900}, url: '/next?scope=personal#runway', api404: true});
      await page.waitForSelector('#view .panel');
      await page.click('[data-scope="enterprise"]');
      apis.length = 0;
      await page.click('[data-view-link="history"]');
      await page.waitForFunction(() => [...document.querySelectorAll('#view .p-state')].some(e => /unavailable/i.test(e.textContent)));
      await page.waitForTimeout(300);
      assert.equal(apis.some(a => a.startsWith('/api/window-history')), false, apis.join(' '));
      assert.equal(await page.locator('[data-panel="history-windows"]').isHidden(), true);
      assert.deepEqual(errors, []);
      await context.close();
    });

    await check('real mode without endpoints: History and Lab panels say unavailable', async () => {
      for (const view of ['history', 'lab']) {
        const {page, errors, context} = await open(browser, {viewport: {width: 1440, height: 900}, url: `/next?scope=personal#${view}`, api404: true});
        await page.waitForFunction(() => [...document.querySelectorAll('#view .p-state')].filter(e => /unavailable/i.test(e.textContent)).length >= 3);
        assert.deepEqual(errors, []);
        await context.close();
      }
    });
  } finally {
    await browser.close();
  }
  if (failures.length) {
    console.error(`\n${failures.length} failure(s):\n` + failures.join('\n'));
    process.exit(1);
  }
})().catch(error => { console.error(error); process.exit(1); });
