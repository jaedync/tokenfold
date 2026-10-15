// Unit checks for static/next/js/feeds.js in headless Chromium: a scope
// switch keeps the old content until the new data is in, a remount does not
// refetch fresh data, and prefetch() reads a feed once. A fake api answers
// each request when the check says so; no server, network or production data.
const path = require('node:path');
const {chromium} = require(process.env.TOKENFOLD_PLAYWRIGHT || 'playwright');

const ROOT = path.resolve(__dirname, '..', '..');
const ORIGIN = 'http://tokenfold.test';
const CSP = "default-src 'self'; style-src 'self'; img-src 'self' data:";
const failures = [];
async function check(name, page, fn) {
  try {
    await page.evaluate(fn);
    console.log('ok - ' + name);
  } catch (error) {
    failures.push(name + ': ' + error.message);
    console.log('FAIL - ' + name + '\n  ' + error.message);
  }
}

(async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    await page.route('**/*', route => {
      const u = new URL(route.request().url());
      if (u.pathname === '/') return route.fulfill({contentType: 'text/html', body: '<!doctype html><body></body>', headers: {'Content-Security-Policy': CSP}});
      if (u.pathname.startsWith('/js/')) return route.fulfill({path: path.join(ROOT, 'static', 'next', u.pathname)});
      return route.fulfill({status: 404, body: ''});
    });
    await page.goto(ORIGIN + '/');
    await page.evaluate(async () => {
      const {createFeeds} = await import('/js/feeds.js');
      const {createStore} = await import('/js/store.js');
      window.must = (ok, msg) => { if (!ok) throw new Error(msg); };
      window.tick = (ms = 5) => new Promise((r) => setTimeout(r, ms));
      // A fake api: every request waits until answer(name, data) or fail(name).
      window.setup = (opts = {}) => {
        const calls = [];
        const api = {
          getJSON(name, params, {scope, signal, meta}) {
            return new Promise((resolve, reject) => {
              const call = {name, params, scope, resolve, reject, done: false, meta};
              calls.push(call);
              signal.addEventListener('abort', () => { call.done = true; reject(new DOMException('aborted', 'AbortError')); });
            });
          },
        };
        const store = createStore({scope: 'personal'});
        const feeds = createFeeds({store, api, mock: false, ...opts});
        const open = (name) => calls.filter((c) => c.name === name && !c.done);
        // stale: the server served a body built before the newest data (X-Tokenfold-Stale).
        const answer = async (name, data, {stale = false} = {}) => {
          const c = open(name)[0];
          must(c, `no open request for ${name}`);
          if (stale) { must(c.meta, 'getJSON got no meta object'); c.meta.stale = true; }
          c.done = true; c.resolve(data);
          await tick();
        };
        const fail = async (name) => {
          const c = open(name)[0];
          must(c, `no open request for ${name}`);
          c.done = true; c.reject(Object.assign(new Error('boom'), {status: 500}));
          await tick();
        };
        return {store, feeds, calls, open, answer, fail};
      };
    });

    await check('a scope switch keeps the old data until the new data is in', page, async () => {
      const t = setup();
      t.feeds.want('a', {endpoint: 'a'});
      await tick();
      await t.answer('a', {v: 'personal'});
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick();
      const s = t.store.get();
      must(s.switching === true, 'switching is not set');
      must(s.a.status === 'ok' && s.a.data.v === 'personal', `the old data went away: ${JSON.stringify(s.a)}`);
      must(t.open('a')[0].scope === 'enterprise', 'no request for the new scope');
      await t.answer('a', {v: 'work'});
      must(!t.store.get().switching, 'switching did not end');
      must(t.store.get().a.data.v === 'work', 'the new data is not in');
    });

    await check('a switch ends only when every feed in use has data of the new scope', page, async () => {
      const t = setup();
      t.feeds.want('a', {endpoint: 'a'});
      t.feeds.want('b', {endpoint: 'b'});
      await tick();
      await t.answer('a', 1); await t.answer('b', 1);
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick();
      await t.answer('a', 2);
      must(t.store.get().switching === true, 'the switch ended before b answered');
      await t.answer('b', 2);
      must(!t.store.get().switching, 'switching did not end');
    });

    await check('a dependent feed refetches before the switch ends', page, async () => {
      const t = setup();
      t.feeds.want('a', {endpoint: 'a'});
      t.feeds.want('b', {endpoint: 'b', params: (s) => (s.a && s.a.data ? {w: s.a.data.w} : null)});
      await tick();
      await t.answer('a', {w: 'week'});
      await tick();
      await t.answer('b', 'week data');
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick();
      // b waits for the new data of a: the old parameter is of the old scope.
      must(t.open('b').length === 0, 'b asked with the old parameter');
      must(t.store.get().b.data === 'week data', 'the old data of b did not stay during the switch');
      await t.answer('a', {w: 'month'});
      await tick();
      must(t.store.get().switching === true, 'the switch ended while b had the old parameter');
      must(t.open('b')[0] && t.open('b')[0].params.w === 'month', 'b did not ask again with the new parameter');
      await t.answer('b', 'month data');
      must(!t.store.get().switching, 'switching did not end');
    });

    await check('a failed read during a switch drops the old scope data', page, async () => {
      const t = setup();
      t.feeds.want('a', {endpoint: 'a'});
      await tick();
      await t.answer('a', {v: 'personal'});
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick();
      await t.fail('a');
      const a = t.store.get().a;
      must(a.status === 'error' && a.data === null, `old scope data stays: ${JSON.stringify(a)}`);
      must(!t.store.get().switching, 'switching did not end');
    });

    await check('a switch drops the data of feeds not in use and of feeds with no read in the new scope', page, async () => {
      const t = setup();
      const release = t.feeds.want('idle', {endpoint: 'idle'});
      t.feeds.want('mine', {endpoint: 'mine', params: (s) => (s.scope === 'personal' ? {} : null)});
      await tick();
      await t.answer('idle', 1); await t.answer('mine', 1);
      release();
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick();
      const s = t.store.get();
      must(s.idle.status === 'loading' && s.idle.data === null, 'an unused feed kept old scope data');
      must(s.mine.status === 'loading' && s.mine.data === null, 'a feed with no read kept old scope data');
      must(!s.switching, 'nothing to wait for, but switching is set');
    });

    await check('a feed released during a switch never shows old scope data when it comes back', page, async () => {
      const t = setup();
      const releaseA = t.feeds.want('a', {endpoint: 'a'});
      t.feeds.want('b', {endpoint: 'b'});
      await tick();
      await t.answer('a', {v: 'personal'}); await t.answer('b', {v: 'personal'});
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick();
      // A view switch before a answers: a loses its user.
      releaseA();
      let s = t.store.get();
      must(!(s.a && s.a.data), `a released feed kept old scope data: ${JSON.stringify(s.a)}`);
      await t.answer('b', {v: 'work'});
      must(!t.store.get().switching, 'switching did not end');
      t.feeds.want('a', {endpoint: 'a'});
      s = t.store.get();
      must(s.a.status === 'loading' && s.a.data === null, `old scope data came back: ${JSON.stringify(s.a)}`);
      await tick();
      await t.answer('a', {v: 'work'});
      must(t.store.get().a.data.v === 'work', 'the new data is not in');
    });

    await check('the end of a switch clears every feed of an old generation, with or without users', page, async () => {
      const t = setup({switchHoldMs: 50});
      // A prefetched feed, no users: its read is in flight when the scope changes.
      t.feeds.want('a', {endpoint: 'a'});
      t.feeds.prefetch('p', {endpoint: 'p'});
      await tick();
      await t.answer('a', 1); await t.answer('p', {v: 'personal'});
      const releaseB = t.feeds.want('b', {endpoint: 'b'});
      await tick();
      await t.answer('b', {v: 'personal'});
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick();
      releaseB();
      await tick(80);
      const s = t.store.get();
      must(!s.switching, 'still switching');
      for (const k of ['a', 'b', 'p']) must(!(s[k] && s[k].data), `${k} kept old scope data: ${JSON.stringify(s[k])}`);
    });

    await check('a slow switch stops holding the old content after switchHoldMs', page, async () => {
      const t = setup({switchHoldMs: 50});
      t.feeds.want('a', {endpoint: 'a'});
      await tick();
      await t.answer('a', 1);
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick(80);
      const s = t.store.get();
      must(!s.switching, 'still switching');
      must(s.a.status === 'loading' && s.a.data === null, 'old scope data shows after the hold');
    });

    await check('a remount does not refetch fresh data at once', page, async () => {
      const t = setup();
      let release = t.feeds.want('a', {endpoint: 'a', intervalMs: 60000});
      await tick();
      await t.answer('a', 1);
      release();
      release = t.feeds.want('a', {endpoint: 'a', intervalMs: 60000});
      await tick(20);
      must(t.open('a').length === 0, 'fresh data was fetched again');
      release();
    });

    await check('a remount refetches data older than its interval', page, async () => {
      const t = setup();
      let release = t.feeds.want('a', {endpoint: 'a', intervalMs: 30});
      await tick();
      await t.answer('a', 1);
      release();
      await tick(40);
      release = t.feeds.want('a', {endpoint: 'a', intervalMs: 30});
      await tick();
      must(t.open('a').length === 1, 'old data was not fetched again');
      release();
    });

    await check('prefetch reads a feed once and a later want() reuses the read', page, async () => {
      const t = setup();
      t.feeds.prefetch('p', {endpoint: 'p', intervalMs: 20});
      await tick();
      must(t.open('p').length === 1, 'prefetch did not read');
      const release = t.feeds.want('p', {endpoint: 'p', intervalMs: 20});
      await tick();
      must(t.calls.filter((c) => c.name === 'p').length === 1, 'want() started a second read');
      await t.answer('p', 1);
      must(t.store.get().p.data === 1, 'the prefetched data is not in the store');
      release();
      t.feeds.prefetch('q', {endpoint: 'q', intervalMs: 20});
      await tick();
      await t.answer('q', 1);
      await tick(60);
      must(t.calls.filter((c) => c.name === 'q').length === 1, 'a prefetched feed polls');
      t.feeds.prefetch('q', {endpoint: 'q', intervalMs: 20});
      await tick();
      must(t.calls.filter((c) => c.name === 'q').length === 2, 'a second prefetch of old data did not read');
    });

    await check('a stale answer is read again once, soon, not at the next poll', page, async () => {
      const t = setup({staleRetryMs: 30});
      t.feeds.want('a', {endpoint: 'a', intervalMs: 60000, bump: true});
      await tick();
      await t.answer('a', 1, {stale: true});
      must(t.store.get().a.data === 1, 'the stale body is not shown');
      await tick(50);
      must(t.open('a').length === 1, 'no second read after a stale answer');
      await t.answer('a', 2, {stale: true});
      await tick(50);
      must(t.open('a').length === 0, 'a second stale answer read again (once per bump)');
      must(t.store.get().a.data === 2, 'the second body is not shown');
    });

    await check('a bump allows one more stale read again; a fresh answer needs none', page, async () => {
      const t = setup({staleRetryMs: 30, bumpGapMs: 0});
      t.feeds.want('a', {endpoint: 'a', intervalMs: 60000, bump: true});
      await tick();
      await t.answer('a', 1, {stale: true});
      await tick(50);
      await t.answer('a', 2, {stale: true});
      t.feeds.bump();
      await tick(10);
      await t.answer('a', 3, {stale: true});
      await tick(50);
      must(t.open('a').length === 1, 'no read again after the stale answer to a bump');
      await t.answer('a', 4);
      await tick(50);
      must(t.open('a').length === 0, 'a fresh answer was read again');
    });

    await check('a bump never moves a sooner read later', page, async () => {
      // A bump soon after a read waits out the gap; a stale read-again or a
      // poll due before that must still happen on time.
      const t = setup({staleRetryMs: 40, bumpGapMs: 400});
      t.feeds.want('a', {endpoint: 'a', intervalMs: 60000, bump: true});
      t.feeds.want('p', {endpoint: 'p', intervalMs: 60, bump: true});
      await tick();
      await t.answer('a', 1, {stale: true});
      await t.answer('p', 1);
      t.feeds.bump();
      await tick(100);
      must(t.open('a').length === 1, 'the bump moved the stale read-again later');
      must(t.open('p').length === 1, 'the bump moved the poll later');
    });

    await check('a feed that waits for another feed reads after a view switch during a scope switch', page, async () => {
      // The ledger period feed needs the 30-day data. A scope switch starts its
      // read, a view switch aborts it and drops the data; back on the view, the
      // period feed has no params until the 30-day data is in again.
      const t = setup();
      const dep = {endpoint: 'p', intervalMs: 60000, params: (s) => (s.l && s.l.data ? {from: 1} : null)};
      const want = () => [t.feeds.want('l', {endpoint: 'l', intervalMs: 60000}), t.feeds.want('p', dep)];
      let rel = want();
      await tick();
      await t.answer('l', 1);
      await tick();
      await t.answer('p', 1);
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick();
      must(t.open('l').length === 1, 'the scope switch did not read again');
      rel.forEach((r) => r());
      await tick();
      rel = want();
      await tick();
      await t.answer('l', 2);
      await tick();
      must(t.open('p').length === 1, 'the period feed never reads again');
      await t.answer('p', 2);
      must(t.store.get().p.data === 2, 'the period data is not in');
      rel.forEach((r) => r());
    });

    await check('a read aborted by a view switch does not make old data count as fresh', page, async () => {
      // The period changes (new params), the view goes before the answer; back
      // on the view, the data in the store is still for the old params.
      const t = setup();
      t.store.set('from', 1);
      const spec = {endpoint: 'p', intervalMs: 60000, params: (s) => ({from: s.from})};
      let rel = t.feeds.want('p', spec);
      await tick();
      await t.answer('p', 'one');
      t.store.set('from', 2);
      await tick();
      must(t.open('p').length === 1, 'no read for the new params');
      rel();
      await tick();
      rel = t.feeds.want('p', spec);
      await tick();
      must(t.open('p').length === 1, 'the data for the old params counted as fresh');
      must(t.open('p')[0].params.from === 2, 'the read is not for the new params');
      rel();
    });

    await check('a scope switch never reads with params from the old scope data', page, async () => {
      // The Runway trajectory reads the binding window of the runway data. The
      // personal window is unknown in the work scope (404), so the read must
      // wait for the work runway data; the switch waits for both.
      const t = setup();
      const dep = {endpoint: 'traj', params: (s) => (s.run && s.run.data ? {window: s.run.data} : null)};
      t.feeds.want('run', {endpoint: 'run'});
      t.feeds.want('traj', dep);
      await tick();
      await t.answer('run', 'personal-week');
      await tick();
      await t.answer('traj', 'personal-points');
      t.store.set('scope', 'enterprise');
      t.feeds.reset();
      await tick();
      must(t.open('traj').length === 0, `a read with old params: ${JSON.stringify(t.open('traj').map((c) => c.params))}`);
      must(t.store.get().traj.data === 'personal-points', 'the old data did not stay during the switch');
      await t.answer('run', 'work-week');
      await tick();
      must(t.store.get().switching === true, 'the switch ended before the dependent feed answered');
      const c = t.open('traj');
      must(c.length === 1 && c[0].params.window === 'work-week' && c[0].scope === 'enterprise', `wrong read: ${JSON.stringify(c.map((x) => [x.scope, x.params]))}`);
      await t.answer('traj', 'work-points');
      must(!t.store.get().switching, 'the switch did not end');
      must(t.store.get().traj.data === 'work-points', 'the new data is not in');
    });

    if (errors.length) failures.push('page errors: ' + errors.join('; '));
  } finally {
    await browser.close();
  }
  if (failures.length) {
    console.log('\n' + failures.length + ' failure(s)');
    process.exit(1);
  }
})().catch(error => { console.error(error); process.exit(1); });
