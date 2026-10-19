// Sessions timeline: one swimlane per machine, one bar per session (colour =
// orchestrator model, height = cost on a log scale), thin lines under a bar
// for its subagent runs. Quiet sessions stay out unless the control bar
// shows them. Click or Enter opens the session drawer. The bars are one tab
// stop (roving tabindex): arrow keys, Home and End move between sessions.
import { h, patch, setText, panelHead } from '../dom.js';
import * as f from '../format.js';
import { derive, matched, dayStart, sessionName, projectLabel, DAY_S } from './ledger-model.js';
import { colorsFor, svg, tooltip, tipRow, money, plural, openSession, keepFocus, dayLabel, ledgerWait, ledgerStatus } from './ledger-kit.js';

const H = 3600, ROW = 30, AX = 30, PAD = 6, MAX_THREADS = 4, MAX_ROWS = 8;
const INK = '#161616', MUTED = '#5d5950', PAPER_2 = '#e8e1d2', RED = '#d2342a', HAIR = 'rgba(22,22,22,.14)';
const dateFmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });
const hourFmt = new Intl.DateTimeFormat('en-US', { hour: 'numeric' });

function geometry(d, avail, phone) {
  const from = d.range.from;
  const to = d.ui.pkey === 'today' ? Math.max(d.range.to + 50 * 60, from + H) : Math.max(from + H, d.range.to);
  const spanD = (to - from) / DAY_S;
  const W = phone ? Math.max(avail, Math.round(spanD * (spanD > 8 ? 70 : 150))) : avail;
  const gut = d.range.open ? 44 : 8;
  return { from, to, spanD, W, x: (t) => ((t - from) / (to - from)) * (W - gut - 2) + 2 };
}

// A burst of parallel sessions (a test loop, say) would need dozens of rows.
// After MAX_ROWS a session shares the row that frees up first; those are
// drawn first, so a full-size bar stays on top of them.
function packLane(list, g, labelOf) {
  const rows = [];
  const place = list.map((m) => {
    const x0 = g.x(Math.max(m.s.start, g.from)), x1 = Math.max(x0 + 3, g.x(Math.min(m.s.end, g.to)));
    let r = rows.findIndex((end) => end + 5 <= x0);
    const over = r < 0 && rows.length >= MAX_ROWS;
    if (over) r = rows.indexOf(Math.min(...rows));
    else if (r < 0) { rows.push(0); r = rows.length - 1; }
    rows[r] = Math.max(rows[r], x1 + (labelOf(m, x1 - x0) ? 42 : 0));
    return { ...m, r, x0, x1, over };
  });
  return { rows, place: [...place.filter((p) => p.over), ...place.filter((p) => !p.over)] };
}

function sessionG(p, top, ctx) {
  const { s } = p;
  const col = ctx.c.color(s.model);
  const hgt = Math.round(5 + 11 * Math.min(1, Math.log10(1 + p.total) / Math.log10(1 + ctx.costMax)));
  const base = top + 15, w = Math.max(3, p.x1 - p.x0);
  const g = svg('g', { class: `tl-sess${p.ok ? '' : ' is-dim'}${ctx.sel === s.id ? ' is-sel' : ''}`, 'data-sid': s.id, 'data-key': s.id,
    tabindex: p.ok && ctx.tabStop === s.id ? 0 : -1, role: 'button', 'aria-hidden': p.ok ? null : 'true',
    'aria-label': `${sessionName(s)}, ${s.machine}, ${money(p.total)}, ${plural(s.runs.length, 'subagent')}` });
  g.append(svg('rect', { class: 's-bar', x: p.x0, y: base - hgt, width: w, height: hgt, fill: col.fill, stroke: w > 5 ? INK : null, 'stroke-width': w > 5 ? 1 : null }));
  if (s.live) g.append(svg('rect', { class: 'live-cap', x: p.x1 - 2, y: base - hgt - 3, width: 4, height: hgt + 6, fill: RED }));
  if (ctx.threads) {
    // Parallel runs stack (up to 4 lines), so a fan-out reads as a comb.
    const lanes = [];
    for (const r of s.runs) {
      const a = ctx.g.x(Math.max(r.start, ctx.g.from)), b = Math.max(a + 1.5, ctx.g.x(Math.min(r.end, ctx.g.to)));
      let k = lanes.findIndex((e) => e + 1 <= a);
      if (k < 0) { if (lanes.length < MAX_THREADS) { lanes.push(0); k = lanes.length - 1; } else k = lanes.indexOf(Math.min(...lanes)); }
      lanes[k] = b;
      g.append(svg('rect', { x: a, y: base + 2.5 + k * 3, width: b - a, height: 2, fill: ctx.c.color(r.model).line }));
    }
  }
  if (ctx.labelOf(p, w)) {
    g.append(svg('text', { x: p.x1 + 5, y: base - 1, class: 'tl-cost', text: p.total >= 100 ? f.usd(p.total) : `$${Math.round(p.total)}` }));
  }
  return g;
}

function axis(g, d, H_) {
  const out = [];
  const { from, to, spanD } = g;
  const pxDay = g.W / spanD;
  const every = pxDay >= 70 ? 1 : pxDay >= 34 ? 2 : pxDay >= 18 ? 4 : 7;
  const nowX = d.range.open ? g.x(d.now) : -1e9;
  let i = 0;
  for (let t = dayStart(from); t < to; t = dayStart(t, 1), i += 1) {
    const nx0 = g.x(Math.max(from, t)), nx1 = g.x(Math.min(to, t + 7 * H));
    if (nx1 > nx0) out.push(svg('rect', { x: nx0, y: AX, width: nx1 - nx0, height: H_ - AX, fill: PAPER_2, opacity: 0.8 }));
    if (t > from) out.push(svg('line', { x1: g.x(t), x2: g.x(t), y1: AX - 8, y2: H_, stroke: INK, 'stroke-width': 1.5, opacity: 0.5 }));
    if (spanD > 1.2 && i % every === 0 && t + 3 * H < to) {
      const label = (pxDay < 80 ? dateFmt.format(new Date(t * 1000)) : dayLabel(t)).toUpperCase();
      const lx = Math.max(g.x(t), 2) + 6;
      if (lx + label.length * 7.6 < nowX - 20 || lx > nowX + 20) out.push(svg('text', { x: lx, y: AX - 10, class: 'tl-day', text: label }));
    }
  }
  if (spanD <= 1.2) {
    const step = g.W < 600 ? 3 : 2;
    for (let hr = 0; hr <= 24; hr += step) {
      const t = dayStart(from) + hr * H;
      if (t < from || t > to - 20 * 60) continue;
      out.push(svg('line', { x1: g.x(t), x2: g.x(t), y1: AX - 6, y2: H_, stroke: HAIR }));
      out.push(svg('text', { x: g.x(t) + 4, y: AX - 10, class: 'tl-hour', text: hourFmt.format(new Date(t * 1000)) }));
    }
  }
  return out;
}

export default {
  id: 'ledger-timeline',
  label: 'Sessions',
  cls: 'p-wide',
  mount(el, ctx) {
    const { store } = ctx;
    const head = panelHead('Sessions');
    const sub = h('span', { class: 'panel-sub' });
    head.el.append(sub);
    el.setAttribute('aria-labelledby', head.id);
    const lanes = h('div', { class: 'tl-lanes' });
    const scroll = h('div', { class: 'tl-scroll' });
    const frame = h('div', { class: 'tl' }, lanes, scroll);
    const empty = h('div');
    const foot = h('div', { class: 'tl-foot' },
      h('div', { class: 'tl-key' },
        h('span', null, h('i', { class: 'k-bar' }), h('i', { class: 'k-bar thin' }), 'Session: colour is the orchestrator model, height is cost'),
        h('span', null, h('i', { class: 'k-sub' }, h('i'), h('i')), 'Its subagent runs, by model'),
        h('span', null, h('i', { class: 'k-now' }), 'Now')),
      h('span', null, 'Click a session to see what it bought. Shaded bands are 00:00 to 07:00.'));
    el.append(head.el, empty, frame, foot);
    const tip = tooltip();

    let seen = null, seenStatus = null, lastW = 0, lastSel = null, byId = new Map(), shownSpan = null;
    // Keyboard order of the matching sessions (lane by lane, by start) and
    // the one bar that takes Tab.
    let order = [], tabStop = null;
    function render(force) {
      const s = store.get();
      const d = derive(s, ctx.now());
      const avail = scroll.clientWidth || 800;
      if (!force && d === seen && avail === lastW && ledgerStatus(s) === seenStatus) return;
      seen = d; lastW = avail; seenStatus = ledgerStatus(s);
      if (!d.hasData) { patch(scroll); patch(lanes); patch(empty, ledgerWait(s)); frame.hidden = true; return; }
      const ok = new Map(matched(d.listed, d.ui.filters).map((m) => [m.s.id, m.cost]));
      frame.hidden = !d.listed.length;
      patch(empty, d.listed.length ? null : h('p', { class: 'p-state' }, 'No sessions in this period.'));
      if (!d.listed.length) { setText(sub, ''); return; }
      const phone = window.innerWidth <= 700;
      const g = geometry(d, avail, phone);
      const c = colorsFor(d);
      const costMax = Math.max(1, ...d.listed.map((x) => x.cost));
      const labelMin = Math.max(5, costMax * 0.04);
      const labelOf = (m, w) => w > 16 && m.total >= labelMin;
      const threads = (g.W - 46) / g.spanD >= 60;
      const machines = new Map();
      for (const x of d.listed) {
        const m = machines.get(x.machine) || { cost: 0, list: [] };
        m.cost += x.cost;
        m.list.push({ s: x, ok: ok.has(x.id), total: ok.has(x.id) ? ok.get(x.id) : x.cost });
        machines.set(x.machine, m);
      }
      byId = new Map(d.listed.map((x) => [x.id, x]));
      const lanesSorted = [...machines.entries()].sort((a, b) => b[1].cost - a[1].cost);
      for (const [, m] of lanesSorted) m.list.sort((a, b) => a.s.start - b.s.start);
      order = lanesSorted.flatMap(([, m]) => m.list.filter((p) => p.ok).map((p) => p.s.id));
      if (!order.includes(tabStop)) tabStop = order.includes(s.ledgerDrawer) ? s.ledgerDrawer : (order[0] || null);
      const body = [], laneEls = [h('div', { class: 'tl-axis-spacer' })];
      let y = AX, shown = 0, runs = 0;
      const cctx = { c, costMax, labelOf, threads, g, sel: s.ledgerDrawer, tabStop };
      for (const [name, m] of lanesSorted) {
        const list = m.list;
        const { rows, place } = packLane(list, g, labelOf);
        const laneH = Math.max(64, rows.length * ROW + PAD * 2);
        let laneCost = 0, laneN = 0;
        for (const p of place) {
          if (p.ok) { laneN += 1; shown += 1; laneCost += p.total; runs += p.s.runs.length; }
          body.push(sessionG(p, y + PAD + p.r * ROW, cctx));
        }
        body.push(svg('line', { x1: 0, x2: g.W, y1: y + laneH, y2: y + laneH, stroke: HAIR }));
        laneEls.push(h('div', { class: 'tl-lane', 'data-key': name, style: { height: `${laneH}px` } }, h('b', { title: name }, name),
          laneN ? [h('span', null, plural(laneN, 'session')), h('span', null, `≈${money(laneCost)}`)] : h('span', null, 'no match')));
        y += laneH;
      }
      const H_ = y;
      const nowEls = d.range.open ? [svg('line', { x1: g.x(d.now), x2: g.x(d.now), y1: AX - 6, y2: H_, stroke: RED, 'stroke-width': 3 }),
        svg('rect', { x: g.x(d.now) - 17, y: AX - 25, width: 34, height: 16, fill: RED }),
        svg('text', { x: g.x(d.now), y: AX - 13, 'text-anchor': 'middle', class: 'tl-now', text: 'NOW' })] : [];
      keepFocus(scroll, () => patch(scroll, svg('svg', { width: g.W, height: H_, viewBox: `0 0 ${g.W} ${H_}`, role: 'group', 'aria-label': 'Sessions by machine over time' },
        ...axis(g, d, H_), svg('line', { x1: 0, x2: g.W, y1: AX, y2: AX, stroke: INK, 'stroke-width': 2 }), ...nowEls, ...body)));
      patch(lanes, laneEls);
      const filtered = d.ui.filters.length > 0;
      setText(sub, `${plural(shown, 'session')} · ${plural(runs, 'subagent run')}${threads ? '' : ' · zoom in to see subagent lines'}${filtered ? ' · the rest are faded' : ''}`);
      // Start at the right end (now) for a new period only: a poll keeps
      // where the user scrolled to.
      const span = `${d.range.from}|${d.ui.pkey}|${avail}`;
      if (g.W > avail && span !== shownSpan) scroll.scrollLeft = g.W;
      shownSpan = span;
      lastSel = s.ledgerDrawer;
    }

    function markSel(id) {
      for (const n of scroll.querySelectorAll('.tl-sess.is-sel')) n.classList.remove('is-sel');
      if (id) { const n = scroll.querySelector(`.tl-sess[data-sid="${CSS.escape(id)}"]`); if (n) n.classList.add('is-sel'); }
    }

    scroll.addEventListener('pointermove', (ev) => {
      const gEl = ev.target.closest('.tl-sess');
      if (!gEl || ev.pointerType === 'touch' || gEl.classList.contains('is-dim')) { tip.hide(); return; }
      const x = byId.get(gEl.dataset.sid);
      if (!x) return;
      tip.show(ev, h('div', { class: 'tt-k' }, `${x.machine} · ${projectLabel(x.project)}`), h('b', null, sessionName(x)), x.live ? h('span', { class: 'live-tag' }, 'Live') : null,
        tipRow(`${dayLabel(x.start)}, ${f.clock(x.start)}`, f.shortDur(x.end - x.start)),
        tipRow(x.model, `≈${money(x.cost)}`, colorsFor(seen).color(x.model).fill),
        tipRow(plural(x.runs.length, 'subagent'), plural(x.prompts, 'prompt')));
    });
    scroll.addEventListener('pointerleave', () => tip.hide());
    scroll.addEventListener('click', (ev) => {
      const gEl = ev.target.closest('.tl-sess');
      if (gEl && !gEl.classList.contains('is-dim')) { tip.hide(); openSession(store, gEl.dataset.sid, gEl); }
    });
    const STEP = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
    function moveTo(id) {
      const next = scroll.querySelector(`.tl-sess[data-sid="${CSS.escape(id)}"]`);
      if (!next) return;
      for (const n of scroll.querySelectorAll('.tl-sess[tabindex="0"]')) n.setAttribute('tabindex', '-1');
      next.setAttribute('tabindex', '0');
      tabStop = id;
      tip.hide();
      next.focus();
    }
    // Focus that arrives another way (a click, the drawer handing it back)
    // moves the tab stop too.
    scroll.addEventListener('focusin', (ev) => {
      const gEl = ev.target.closest('.tl-sess');
      if (gEl && gEl.getAttribute('tabindex') !== '0' && order.includes(gEl.dataset.sid)) moveTo(gEl.dataset.sid);
    });
    scroll.addEventListener('keydown', (ev) => {
      const gEl = ev.target.closest('.tl-sess');
      if (!gEl) return;
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openSession(store, gEl.dataset.sid, gEl); return; }
      const i = order.indexOf(gEl.dataset.sid);
      let j = null;
      if (ev.key in STEP) j = Math.min(order.length - 1, Math.max(0, i + STEP[ev.key]));
      else if (ev.key === 'Home') j = 0;
      else if (ev.key === 'End') j = order.length - 1;
      if (j === null || i < 0) return;
      ev.preventDefault();
      if (j !== i) moveTo(order[j]);
    });
    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(() => render(false)) : null;
    if (ro) ro.observe(el);

    return {
      update(s) {
        render(false);
        if (s.ledgerDrawer !== lastSel) { lastSel = s.ledgerDrawer; markSel(lastSel); }
      },
      destroy() { if (ro) ro.disconnect(); tip.destroy(); },
    };
  },
};
