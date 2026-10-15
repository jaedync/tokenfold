// Tiny DOM helpers. Text goes through textContent and attributes through
// setAttribute, never innerHTML: machine, project and session names are
// untrusted strings from many hosts.

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') {
      for (const [prop, val] of Object.entries(v)) el.style.setProperty(prop, val);
    } else if (k.startsWith('on') && typeof v === 'function') listen(el, k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, String(v));
  }
  append(el, children);
  return el;
}

// Handlers given to h() go through one dispatcher per element and type, so
// patch() can hand a kept node the handler of the node that replaces it (a
// closure over the new data, not the old).
const HANDLERS = new WeakMap();
function dispatch(ev) {
  const fn = (HANDLERS.get(ev.currentTarget) || {})[ev.type];
  return fn ? fn.call(ev.currentTarget, ev) : undefined;
}
export function listen(el, type, fn) {
  let map = HANDLERS.get(el);
  if (!map) { map = {}; HANDLERS.set(el, map); }
  if (!(type in map)) el.addEventListener(type, dispatch);
  map[type] = fn;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

// Replace every child of el.
export function fill(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

// Make el's children look like `children` (built fresh with h()), but keep
// every existing node that can stay: a poll then changes only the text and
// attributes that moved, and hover, focus, selection, scroll and <details>
// state survive. Children with data-key match by key (a sorted list keeps
// each row's node); others match by tag in order. A node from `children` that
// is already a child of el is kept as is. Keep no reference to a node built
// for patch(): when an old node matches, the new one is dropped.
export function patch(el, ...children) {
  const next = [];
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    next.push(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  morphChildren(el, next);
  return el;
}

const keyOf = (n) => (n.nodeType === 1 ? n.getAttribute('data-key') : null);
const sameKind = (a, b) => a.nodeType === b.nodeType && a.nodeName === b.nodeName
  && (a.nodeType !== 1 || (a.namespaceURI === b.namespaceURI && keyOf(a) === keyOf(b)));

function morphChildren(parent, next) {
  const old = Array.from(parent.childNodes);
  const byKey = new Map();
  for (const o of old) { const k = keyOf(o); if (k !== null && !byKey.has(k)) byKey.set(k, o); }
  // Nodes passed in that already live here stay themselves, never a match for another.
  const staying = new Set(next.filter((n) => n.parentNode === parent));
  const used = new Set();
  let cursor = 0;
  const out = next.map((n) => {
    if (staying.has(n) && !used.has(n)) { used.add(n); return n; }
    const k = keyOf(n);
    let match = null;
    if (k !== null) {
      const o = byKey.get(k);
      if (o && !used.has(o) && sameKind(o, n)) match = o;
    } else {
      for (let i = cursor; i < old.length; i += 1) {
        const o = old[i];
        if (used.has(o) || staying.has(o) || keyOf(o) !== null || !sameKind(o, n)) continue;
        match = o; cursor = i + 1;
        break;
      }
    }
    if (!match) return n;
    used.add(match);
    morphNode(match, n);
    return match;
  });
  for (const o of old) if (!used.has(o)) o.remove();
  place(parent, out);
}

// Put `out` in order under parent with as few moves as possible: the kept
// nodes on the longest run that is already in order stay, the others move.
// A move with insertBefore takes the node out of the document for a moment,
// which drops the focus and inner scroll inside it, so moveBefore (which
// keeps them) goes first. Either move collapses a selection inside the node
// (the DOM updates live ranges as for a removal), so it is put back.
function place(parent, out) {
  const index = new Map();
  Array.from(parent.childNodes).forEach((n, i) => index.set(n, i));
  const kept = out.filter((n) => index.has(n));
  const stay = new Set(longestRun(kept.map((n) => index.get(n))).map((i) => kept[i]));
  const moves = [];
  let ref = null;
  for (let i = out.length - 1; i >= 0; i -= 1) {
    const node = out[i];
    if (!stay.has(node)) moves.push([node, ref]);
    ref = node;
  }
  if (!moves.length) return;
  const native = typeof parent.moveBefore === 'function' && parent.isConnected;
  const saved = saveState(moves.filter(([n]) => index.has(n)).map(([n]) => n), native);
  for (const [node, before] of moves) {
    if (native && index.has(node)) {
      try { parent.moveBefore(node, before); continue; } catch { /* not movable here: insert */ }
    }
    parent.insertBefore(node, before);
  }
  if (saved) restoreState(saved);
}

// Indexes (into seq) of one longest strictly increasing subsequence.
function longestRun(seq) {
  const tails = [], prev = new Array(seq.length);
  for (let i = 0; i < seq.length; i += 1) {
    let lo = 0, hi = tails.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (seq[tails[mid]] < seq[i]) lo = mid + 1; else hi = mid; }
    prev[i] = lo ? tails[lo - 1] : -1;
    tails[lo] = i;
  }
  const run = [];
  for (let i = tails.length ? tails[tails.length - 1] : -1; i >= 0; i = prev[i]) run.push(i);
  return run.reverse();
}

// The focus, the selection and inner scroll positions inside nodes that move.
function saveState(moving, native) {
  if (!moving.length) return null;
  const inside = (n) => n && moving.some((m) => m === n || m.contains(n));
  const act = document.activeElement;
  const sel = typeof getSelection === 'function' ? getSelection() : null;
  const range = sel && sel.rangeCount && (inside(sel.anchorNode) || inside(sel.focusNode))
    ? [sel.anchorNode, sel.anchorOffset, sel.focusNode, sel.focusOffset] : null;
  const scrolled = [];
  for (const m of native ? [] : moving) {
    if (m.nodeType !== 1) continue;
    for (const el of [m, ...m.querySelectorAll('*')]) {
      if (el.scrollTop || el.scrollLeft) scrolled.push([el, el.scrollTop, el.scrollLeft]);
    }
  }
  return { focus: inside(act) ? act : null, range, scrolled };
}

function restoreState({ focus, range, scrolled }) {
  for (const [el, top, left] of scrolled) {
    if (el.scrollTop !== top) el.scrollTop = top;
    if (el.scrollLeft !== left) el.scrollLeft = left;
  }
  if (focus && focus.isConnected && document.activeElement !== focus) focus.focus({ preventScroll: true });
  if (range && range[0].isConnected && range[2].isConnected) {
    try { getSelection().setBaseAndExtent(...range); } catch { /* the offsets no longer fit: drop the selection */ }
  }
}

function morphNode(o, n) {
  if (o.nodeType !== 1) {
    if (o.nodeValue !== n.nodeValue) o.nodeValue = n.nodeValue;
    return;
  }
  for (const { name, value } of Array.from(n.attributes)) {
    if (name === 'style' || (name === 'open' && o.localName === 'details')) continue;
    if (o.getAttribute(name) !== value) o.setAttribute(name, value);
  }
  for (const { name } of Array.from(o.attributes)) {
    // <details open> belongs to the user, not to the data.
    if (name === 'style' || (name === 'open' && o.localName === 'details')) continue;
    if (!n.hasAttribute(name)) o.removeAttribute(name);
  }
  // CSSOM, not the style attribute: the page's CSP has no 'unsafe-inline'.
  if (o.style && o.style.cssText !== n.style.cssText) o.style.cssText = n.style.cssText;
  const fresh = HANDLERS.get(n);
  const kept = HANDLERS.get(o);
  if (fresh || kept) {
    for (const type of Object.keys(kept || {})) if (!fresh || !(type in fresh)) kept[type] = null;
    for (const [type, fn] of Object.entries(fresh || {})) listen(o, type, fn);
  }
  if ('value' in n && (o.localName === 'input' || o.localName === 'select' || o.localName === 'textarea')
    && o !== document.activeElement && o.value !== n.value) o.value = n.value;
  morphChildren(o, Array.from(n.childNodes));
}

// Writes only on change, so a 5 s poll does not restyle text that stayed the same.
export function setText(el, text) {
  const s = text === null || text === undefined ? '' : String(text);
  if (el.textContent !== s) el.textContent = s;
}

export function setAttr(el, name, value) {
  if (value === null || value === undefined || value === false) {
    if (el.hasAttribute(name)) el.removeAttribute(name);
    return;
  }
  const s = value === true ? '' : String(value);
  if (el.getAttribute(name) !== s) el.setAttribute(name, s);
}

// The "≈" mark for every estimate.
export const approx = () => h('span', { class: 'approx', title: 'Estimate' }, '≈');

// Segmented button group. aria-pressed carries the state for screen readers.
export function segmented(label, options, current, onPick) {
  const group = h('div', { class: 'seg', role: 'group', 'aria-label': label });
  const buttons = options.map(([value, text]) => h('button', {
    type: 'button', 'data-value': value, 'aria-pressed': String(value === current),
    onclick: () => { select(value); onPick(value); },
  }, text));
  function select(value) {
    for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.value === value));
  }
  group.append(...buttons);
  return { el: group, select };
}

let ids = 0;
export const uid = (prefix) => `${prefix}-${++ids}`;

// Standard panel heading: red square, display title, optional sub line and extras.
export function panelHead(title, ...extra) {
  const id = uid('pt');
  const titleEl = h('h2', { class: 'panel-title', id }, title);
  return { el: h('div', { class: 'panel-head' }, titleEl, ...extra), id, titleEl };
}
