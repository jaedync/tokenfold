// Unit checks for patch() in static/next/js/dom.js, in headless Chromium.
// page.route serves dom.js from the repo on a blank page with the shell's
// CSP; no server, network or production data.
const assert = require('node:assert/strict');
const path = require('node:path');
const {chromium} = require(process.env.TOKENFOLD_PLAYWRIGHT || 'playwright');

const ROOT = path.resolve(__dirname, '..', '..');
const ORIGIN = 'http://tokenfold.test';
const CSP = "default-src 'self'; style-src 'self'; img-src 'self' data:";
const failures = [];
async function check(name, page, fn, arg) {
  try {
    await page.evaluate(fn, arg);
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
      if (u.pathname === '/') return route.fulfill({contentType: 'text/html', body: '<!doctype html><body><div id="root"></div></body>', headers: {'Content-Security-Policy': CSP}});
      if (u.pathname === '/dom.js') return route.fulfill({path: path.join(ROOT, 'static', 'next', 'js', 'dom.js')});
      if (u.pathname.startsWith('/js/')) return route.fulfill({path: path.join(ROOT, 'static', 'next', u.pathname)});
      return route.fulfill({status: 404, body: ''});
    });
    await page.goto(ORIGIN + '/');
    await page.evaluate(async () => {
      window.dom = await import('/dom.js');
      window.fresh = () => { const r = document.getElementById('root'); r.replaceChildren(); return r; };
      window.must = (ok, msg) => { if (!ok) throw new Error(msg); };
    });

    await check('same shape: every node stays, only the text changes', page, () => {
      const {h, patch} = window.dom;
      const root = window.fresh();
      patch(root, h('ul', null, h('li', null, 'a'), h('li', null, 'b')), 'tail');
      const ul = root.firstChild, li = ul.children[1], tail = root.lastChild;
      patch(root, h('ul', null, h('li', null, 'a'), h('li', null, 'c')), 'tail 2');
      window.must(root.firstChild === ul && ul.children[1] === li && root.lastChild === tail, 'a node was replaced');
      window.must(li.textContent === 'c' && tail.nodeValue === 'tail 2', 'text did not change');
    });

    await check('keyed children keep their node across a reorder, an insert and a removal', page, () => {
      const {h, patch} = window.dom;
      const root = window.fresh();
      const list = (keys) => keys.map((k) => h('li', {'data-key': k}, `row ${k}`));
      patch(root, list(['a', 'b', 'c']));
      const [a, b, c] = root.children;
      patch(root, list(['c', 'x', 'a']));
      window.must(root.children.length === 3, 'wrong count');
      window.must(root.children[0] === c && root.children[2] === a, 'a keyed node was replaced');
      window.must(!b.isConnected, 'a removed key stayed');
      window.must(root.children[1].textContent === 'row x', 'the new key is missing');
    });

    await check('attributes and CSSOM styles update; removed attributes go', page, () => {
      const {h, patch} = window.dom;
      const root = window.fresh();
      patch(root, h('span', {class: 'a', title: 't', style: {width: '10%'}}));
      const s = root.firstChild;
      patch(root, h('span', {class: 'b', style: {width: '20%'}}));
      window.must(root.firstChild === s, 'replaced');
      window.must(s.className === 'b' && !s.hasAttribute('title') && s.style.width === '20%', `got ${s.outerHTML}`);
    });

    await check('a kept node runs the handler of the newest render', page, () => {
      const {h, patch} = window.dom;
      const root = window.fresh();
      const hits = [];
      patch(root, h('button', {onclick: () => hits.push(1)}, 'go'));
      const b = root.firstChild;
      patch(root, h('button', {onclick: () => hits.push(2)}, 'go'));
      b.click();
      window.must(root.firstChild === b, 'replaced');
      window.must(JSON.stringify(hits) === '[2]', `handlers ran: ${hits}`);
      patch(root, h('button', null, 'go'));
      b.click();
      window.must(JSON.stringify(hits) === '[2]', 'a removed handler still ran');
    });

    await check('focus, an open <details> and a text selection survive a patch', page, () => {
      const {h, patch} = window.dom;
      const root = window.fresh();
      const view = (n) => [h('details', null, h('summary', null, 'More'), h('p', null, 'inside')),
        h('p', {class: 'sel'}, 'select me'), h('button', null, `n ${n}`)];
      patch(root, view(1));
      root.querySelector('details').open = true;
      const btn = root.querySelector('button');
      btn.focus();
      const sel = getSelection(), r = document.createRange();
      r.selectNodeContents(root.querySelector('.sel'));
      sel.removeAllRanges(); sel.addRange(r);
      patch(root, view(2));
      window.must(document.activeElement === btn, 'focus moved');
      window.must(root.querySelector('details').open, 'details closed');
      window.must(getSelection().toString() === 'select me', `selection is "${getSelection()}"`);
      window.must(btn.textContent === 'n 2', 'text did not change');
    });

    await check('a node that already lives in the parent stays itself', page, () => {
      const {h, patch} = window.dom;
      const root = window.fresh();
      const keep = h('canvas');
      patch(root, h('b', null, 'x'), keep);
      patch(root, keep, h('b', null, 'y'));
      window.must(root.firstChild === keep && root.lastChild.textContent === 'y', root.innerHTML);
    });

    await check('a different tag replaces the node; SVG children match in their namespace', page, () => {
      const {h, patch} = window.dom;
      const NS = 'http://www.w3.org/2000/svg';
      const svg = (w) => { const s = document.createElementNS(NS, 'svg'); const r = document.createElementNS(NS, 'rect'); r.setAttribute('width', w); s.append(r); return s; };
      const root = window.fresh();
      patch(root, h('span', null, 'a'), svg(5));
      const rect = root.querySelector('rect');
      patch(root, h('b', null, 'a'), svg(9));
      window.must(root.firstChild.localName === 'b', 'tag did not change');
      window.must(root.querySelector('rect') === rect && rect.getAttribute('width') === '9', 'rect replaced or not updated');
    });

    await check('an input keeps the value the user is typing', page, () => {
      const {h, patch} = window.dom;
      const root = window.fresh();
      patch(root, h('input', {value: '1'}));
      const input = root.firstChild;
      input.focus(); input.value = '42';
      patch(root, h('input', {value: '1'}));
      window.must(input.value === '42', `value ${input.value}`);
    });

    await check('an SVG node from ledger-kit svg() runs the handler of the newest render', page, async () => {
      // One module graph: svg() and patch() must share dom.js's handler table.
      const {patch} = await import('/js/dom.js');
      const {svg} = await import('/js/panels/ledger-kit.js');
      const root = window.fresh();
      const hits = [];
      const draw = (n) => svg('svg', {width: 10, height: 10}, svg('rect', {width: 10, height: 10, onclick: () => hits.push(n)}));
      patch(root, draw(1));
      const rect = root.querySelector('rect');
      patch(root, draw(2));
      window.must(root.querySelector('rect') === rect, 'the rect was replaced');
      rect.dispatchEvent(new MouseEvent('click', {bubbles: true}));
      window.must(JSON.stringify(hits) === '[2]', `handlers ran: ${hits}`);
    });

    // Keyed rows that move keep focus, selection and inner scroll, with the
    // native moveBefore() and without it (older browsers), and the patch
    // moves only the rows outside the longest run that stays in order.
    for (const native of [true, false]) {
      const mode = native ? 'moveBefore' : 'insertBefore';
      await check(`${mode}: a focused button in a keyed row that moves to the end keeps the focus`, page, (native) => {
        const {h, patch} = window.dom;
        const saved = Element.prototype.moveBefore;
        if (!native) delete Element.prototype.moveBefore;
        try {
          const root = window.fresh();
          const list = (keys) => keys.map((k) => h('li', {'data-key': k}, h('button', null, `row ${k}`)));
          patch(root, list(['a', 'b', 'c', 'd', 'e']));
          const a = root.children[0], btn = a.querySelector('button');
          btn.focus();
          let moved = 0;
          const mo = new MutationObserver((ms) => { for (const m of ms) moved += m.removedNodes.length; });
          mo.observe(root, {childList: true});
          patch(root, list(['b', 'c', 'd', 'e', 'a']));
          mo.takeRecords().forEach((m) => { moved += m.removedNodes.length; });
          mo.disconnect();
          window.must(root.lastElementChild === a, 'the row did not move to the end');
          window.must(document.activeElement === btn, `the focus moved to ${document.activeElement && document.activeElement.outerHTML}`);
          window.must(moved <= 1, `${moved} rows moved for one row out of place`);
        } finally {
          if (saved && !native) Element.prototype.moveBefore = saved;
        }
      }, native);

      await check(`${mode}: a reversed list keeps the focus, the selection and the inner scroll`, page, (native) => {
        const {h, patch} = window.dom;
        const saved = Element.prototype.moveBefore;
        if (!native) delete Element.prototype.moveBefore;
        try {
          const root = window.fresh();
          const keys = ['a', 'b', 'c', 'd', 'e', 'f'];
          const list = (ks) => ks.map((k) => h('li', {'data-key': k},
            h('button', null, `go ${k}`), h('p', {class: 'sel'}, `text ${k}`),
            h('div', {class: 'box', style: {height: '20px', overflow: 'auto'}}, h('div', {style: {height: '200px'}}, 'tall'))));
          patch(root, list(keys));
          const [, b, , d, e] = root.children;
          const btn = b.querySelector('button');
          btn.focus();
          const sel = getSelection(), r = document.createRange();
          r.selectNodeContents(d.querySelector('.sel'));
          sel.removeAllRanges(); sel.addRange(r);
          const box = e.querySelector('.box');
          box.scrollTop = 50;
          patch(root, list(keys.slice().reverse()));
          window.must(root.children[4] === b && root.children[2] === d && root.children[1] === e, 'a keyed row was replaced');
          window.must(document.activeElement === btn, 'the focus moved');
          window.must(getSelection().toString() === 'text d', `selection is "${getSelection()}"`);
          window.must(box.scrollTop === 50, `inner scroll is ${box.scrollTop}`);
        } finally {
          if (saved && !native) Element.prototype.moveBefore = saved;
        }
      }, native);
    }

    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
  if (failures.length) {
    console.log('\n' + failures.length + ' failure(s)');
    process.exit(1);
  }
})().catch(error => { console.error(error); process.exit(1); });
