// "Where it went": daily cost by model over 30 days (daily rollup), as
// stacked bars or as shares. The shaded span under the black bracket is the
// period; drag across days to set it. Legend items filter by model.
import { h, fill, setText, segmented, panelHead } from '../dom.js';
import * as f from '../format.js';
import { feedLine } from '../panel.js';
import { derive, dayStart, filterSets, uiOf } from './ledger-model.js';
import { providerGroup, groupRank } from './ledger-colors.js';
import { setPeriod, toggleFilter, colorsFor, svg, tooltip, tipRow, swatch, dayLabel } from './ledger-kit.js';

const INK = '#161616', MUTED = '#5d5950', PAPER_2 = '#e8e1d2', GRID = 'rgba(22,22,22,.12)';
const dateFmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });

function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw || 1)), n = raw / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
}

const epochOfDay = (day) => { const [y, m, dd] = day.split('-').map(Number); return new Date(y, m - 1, dd).getTime() / 1000; };

// Days with their model dollars; a model filter keeps only those models.
function chartData(d) {
  const by = filterSets(d.ui.filters);
  return (d.daily30 || []).map((x) => {
    const models = {};
    for (const [m, v] of Object.entries(x.by_model || {})) if (v > 0 && (!by.model || by.model.has(m))) models[m] = v;
    return { t: epochOfDay(x.day), models, total: Object.values(models).reduce((a, b) => a + b, 0) };
  });
}

function axes(g, m) {
  for (let v = 0; v <= m.top + 1e-9; v += m.step) {
    g.append(svg('line', { x1: m.ml, x2: m.W - m.mr, y1: m.y(v), y2: m.y(v), stroke: v === 0 ? INK : GRID, 'stroke-width': v === 0 ? 3 : 1 }));
    g.append(svg('text', { x: m.ml - 6, y: m.y(v) + 4, 'text-anchor': 'end', class: 'ax', text: m.share ? `${Math.round(v * 100)}%` : f.usdShort(v) }));
  }
}

function bars(g, m, days, c) {
  days.forEach((day, i) => {
    let acc = 0;
    const x = m.ml + i * m.bw + Math.max(1, m.bw * 0.14), w = Math.max(2, m.bw * 0.72);
    for (const model of c.order) {
      const v = day.models[model];
      if (!v) continue;
      const y1 = m.y(acc + v), y0 = m.y(acc);
      acc += v;
      g.append(svg('rect', { x, y: y1, width: w, height: Math.max(0, y0 - y1 - 0.6), fill: c.color(model).fill }));
    }
    // Today still runs: hatch it so it does not read as a quiet day.
    if (i === days.length - 1 && day.total) g.append(svg('rect', { x, y: m.y(day.total), width: w, height: m.y(0) - m.y(day.total), fill: 'url(#lo-hatch)' }));
  });
}

// 100% stacked area: a migration reads as one band growing into another.
function shares(g, m, days, c) {
  const xs = (i) => m.ml + i * m.bw + m.bw / 2;
  const cum = days.map(() => 0);
  const boxes = [];
  for (const model of c.order) {
    const lo = cum.slice();
    const hi = cum.map((v, i) => v + (days[i].total ? (days[i].models[model] || 0) / days[i].total : 0));
    if (hi.every((v, i) => v - lo[i] < 1e-6)) continue;
    hi.forEach((v, i) => { cum[i] = v; });
    let p = `M${m.ml},${m.y(hi[0])}`;
    days.forEach((_, i) => { p += `L${xs(i)},${m.y(hi[i])}`; });
    p += `L${m.W - m.mr},${m.y(hi[days.length - 1])}L${m.W - m.mr},${m.y(lo[days.length - 1])}`;
    for (let i = days.length - 1; i >= 0; i -= 1) p += `L${xs(i)},${m.y(lo[i])}`;
    g.append(svg('path', { d: `${p}L${m.ml},${m.y(lo[0])}Z`, fill: c.color(model).fill, stroke: '#f3eee3', 'stroke-width': 1 }));
    // Name the band where it is thickest, when the name fits.
    let best = -1, bd = 0;
    for (let i = 2; i < days.length - 2; i += 1) {
      const th = Math.min(...[i - 1, i, i + 1].map((k) => (hi[k] - lo[k]) * m.ih));
      if (th > best) { best = th; bd = i; }
    }
    if (best < 18) continue;
    const fs = m.W < 560 ? 10 : 11.5, cx = xs(bd), cy = m.y((hi[bd] + lo[bd]) / 2) + 4, half = model.length * fs * 0.3;
    const box = [cx - half - 4, cy - fs, cx + half + 4, cy + 3];
    if (box[0] < m.ml || box[2] > m.W - m.mr || boxes.some((o) => box[0] < o[2] && box[2] > o[0] && box[1] < o[3] && box[3] > o[1])) continue;
    boxes.push(box);
    g.append(svg('text', { x: cx, y: cy, 'text-anchor': 'middle', class: 'band-label', fill: c.color(model).text, 'font-size': fs, text: model }));
  }
}

export default {
  id: 'ledger-overview',
  label: 'Where it went',
  mount(el, ctx) {
    const { store } = ctx;
    let mode = 'usd';
    const head = panelHead('Where it went');
    const sub = h('span', { class: 'panel-sub' });
    const modeSeg = segmented('Chart mode', [['usd', '$ / day'], ['share', 'Share']], mode, (v) => { mode = v; draw(true); });
    head.el.append(h('div', { class: 'lo-tools' }, sub, modeSeg.el));
    el.setAttribute('aria-labelledby', head.id);
    const state = h('div');
    const wrap = h('div', { class: 'lo-wrap' });
    const hint = h('p', { class: 'lo-hint' }, 'Drag across the days to set the period. Click one day to see only that day.');
    const note = h('p', { class: 'lo-note', hidden: true });
    const legend = h('div', { class: 'lo-legend' });
    el.append(head.el, state, wrap, hint, note, legend);
    const tip = tooltip();

    let geo = null, days = [], drag = null, last = null, lastW = 0, lastStatus = null;

    function bracket(lo, hi) {
      if (!geo) return;
      const { g, m } = geo;
      g.replaceChildren();
      const x0 = m.ml + lo * m.bw, x1 = m.ml + (hi + 1) * m.bw;
      g.append(svg('rect', { x: x0, y: m.mt - 10, width: x1 - x0, height: 5, fill: INK }));
      if (lo > 0) g.append(svg('rect', { x: x0 - 1, y: m.mt - 10, width: 2, height: m.ih + 10, fill: INK }));
      if (hi < days.length - 1) g.append(svg('rect', { x: x1 - 1, y: m.mt - 10, width: 2, height: m.ih + 10, fill: INK }));
      geo.shade.setAttribute('x', x0);
      geo.shade.setAttribute('width', x1 - x0);
    }

    function periodIdx(d) {
      const i0 = days.findIndex((x) => x.t >= dayStart(d.range.from) - 1);
      const lastT = dayStart(d.range.to - 1);
      let i1 = days.length - 1;
      while (i1 > 0 && days[i1].t > lastT + 1) i1 -= 1;
      return [Math.max(0, i0 < 0 ? 0 : i0), Math.max(0, i1)];
    }

    function draw(force) {
      const d = derive(store.get(), ctx.now());
      const W = Math.max(300, wrap.clientWidth || 600);
      // Without data, derive() stays the same object while the feed moves
      // from loading to error, so the feed status is part of the key.
      const status = store.get().ledger && store.get().ledger.status;
      if (!force && d === last && W === lastW && status === lastStatus) return;
      last = d; lastW = W; lastStatus = status;
      const line = d.daily30 ? null : feedLine(store.get().ledger, 'daily spend');
      fill(state, line || (d.daily30 && !d.daily30.length ? h('p', { class: 'p-state' }, 'No daily spend in these 30 days.') : null));
      if (!d.daily30) { wrap.replaceChildren(); legend.replaceChildren(); geo = null; return; }
      const c = colorsFor(d);
      days = chartData(d);
      const share = mode === 'share';
      const small = W < 560;
      const H = small ? 200 : 290, ml = small ? 40 : 50, mr = 6, mt = 16, mb = 26;
      const iw = W - ml - mr, ih = H - mt - mb, bw = iw / Math.max(1, days.length);
      const maxV = share ? 1 : Math.max(1, ...days.map((x) => x.total));
      const step = share ? 0.25 : niceStep(maxV / 4);
      const top = share ? 1 : Math.ceil(maxV / step) * step;
      const m = { W, H, ml, mr, mt, ih, bw, step, top, share, y: (v) => mt + ih - (v / top) * ih };
      const shade = svg('rect', { y: mt - 10, height: ih + 10, fill: PAPER_2 });
      const body = svg('g');
      axes(body, m);
      (share ? shares : bars)(body, m, days, c);
      const ticks = svg('g');
      days.forEach((x, i) => {
        const today = i === days.length - 1;
        // A Monday label must keep clear of the right-aligned "Today" (about 40 px wide).
        if (!(new Date(x.t * 1000).getDay() === 1 || today) || (!today && W - mr - (ml + i * bw + bw / 2) < 72)) return;
        ticks.append(svg('text', { x: today ? W - mr : ml + i * bw + bw / 2, y: H - 8, 'text-anchor': today ? 'end' : 'middle', class: 'ax', text: today ? 'Today' : dateFmt.format(new Date(x.t * 1000)) }));
      });
      const hover = svg('rect', { y: mt, width: bw, height: ih, fill: 'none', stroke: INK, 'stroke-width': 2, visibility: 'hidden', 'pointer-events': 'none' });
      const brk = svg('g');
      const hit = svg('rect', { x: ml, y: 0, width: iw, height: H, fill: 'transparent', class: 'lo-hit' });
      const defs = svg('defs', null, svg('pattern', { id: 'lo-hatch', width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' },
        svg('rect', { width: 2.2, height: 6, fill: 'rgba(243,238,227,.6)' })));
      const label = `${share ? 'Share of daily cost' : 'Daily cost'} by model for ${days.length} days. The selected period is marked with a black bracket.`;
      wrap.replaceChildren(svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': label }, defs, shade, body, brk, ticks, hover, hit));
      geo = { g: brk, m, shade, hover };
      const [lo, hi] = periodIdx(d);
      bracket(lo, hi);
      const byDims = Object.keys(filterSets(d.ui.filters)).filter((k) => k !== 'model');
      setText(sub, `${share ? 'Share of daily cost' : 'Daily cost'} by model · ${days.length} days${d.ui.filters.some((x) => x.dim === 'model') ? ' · filtered' : ''}`);
      note.hidden = !byDims.length;
      setText(note, byDims.length ? `This chart comes from the daily totals. It follows model filters only, not ${byDims.join(' or ')} filters.` : '');
      drawLegend(d, c);
    }

    function drawLegend(d, c) {
      const inPeriod = new Set(d.sessions.flatMap((s) => s.parts.filter((p) => p.cost > 0).map((p) => p.model)));
      const spend = {};
      for (const x of d.daily30 || []) for (const [m, v] of Object.entries(x.by_model || {})) spend[m] = (spend[m] || 0) + v;
      const models = c.order.filter((m) => spend[m] > 0);
      const groups = new Map();
      for (const m of models) {
        const g = providerGroup(m, d.providers);
        if (!groups.has(g)) groups.set(g, []);
        groups.get(g).push(m);
      }
      const { filters } = d.ui;
      fill(legend, [...groups.keys()].sort((a, b) => groupRank(a) - groupRank(b)).map((g) => h('div', { class: 'lg-group' },
        h('span', { class: 'lg-prov' }, g),
        groups.get(g).map((m) => h('button', {
          type: 'button', class: `lg-item${inPeriod.has(m) ? '' : ' is-empty'}`, 'data-model': m,
          'aria-pressed': String(filters.some((x) => x.dim === 'model' && x.val === m)),
          title: `Filter every panel to ${m}`, onclick: () => toggleFilter(store, 'model', m),
        }, swatch(c.color(m).fill), m)))));
    }

    const dayAt = (ev) => {
      const r = wrap.getBoundingClientRect();
      const x = ((ev.clientX - r.left) * geo.m.W) / r.width;
      return Math.min(days.length - 1, Math.max(0, Math.floor((x - geo.m.ml) / geo.m.bw)));
    };
    function showDay(ev, i) {
      geo.hover.setAttribute('x', geo.m.ml + i * geo.m.bw);
      geo.hover.setAttribute('visibility', 'visible');
      const day = days[i], c = colorsFor(last);
      const rows = Object.keys(day.models).sort((a, b) => day.models[b] - day.models[a]).slice(0, 6)
        .map((m) => tipRow(m, f.usd(day.models[m]), c.color(m).fill));
      tip.show(ev, h('div', { class: 'tt-k' }, `${dayLabel(day.t)}${i === days.length - 1 ? ' · so far' : ''}`),
        h('div', { class: 'tt-big' }, `≈${f.usd(day.total)}`), rows.length ? rows : h('span', null, 'No spend'));
    }
    function hideTip() { tip.hide(); if (geo) geo.hover.setAttribute('visibility', 'hidden'); }

    wrap.addEventListener('pointerdown', (ev) => {
      if (!geo || !ev.target.closest('.lo-hit')) return;
      drag = dayAt(ev);
      wrap.setPointerCapture(ev.pointerId);
      hideTip();
    });
    wrap.addEventListener('pointermove', (ev) => {
      if (!geo) return;
      const i = dayAt(ev);
      if (drag === null) { if (ev.pointerType !== 'touch' && ev.target.closest('.lo-hit')) showDay(ev, i); return; }
      bracket(Math.min(drag, i), Math.max(drag, i));
    });
    wrap.addEventListener('pointerup', (ev) => {
      if (drag === null || !geo) return;
      const i = dayAt(ev), lo = Math.min(drag, i), hi = Math.max(drag, i), n = days.length - 1;
      drag = null;
      // A drag that lands on a preset becomes that preset, so the control shows it.
      if (lo === n) setPeriod(store, 'today');
      else if (hi === n && lo === n - 6) setPeriod(store, '7d');
      else if (hi === n && lo === 0) setPeriod(store, '30d');
      else setPeriod(store, 'custom', days[lo].t, hi === n ? null : dayStart(days[hi].t, 1));
      draw(true);
    });
    wrap.addEventListener('pointercancel', () => { drag = null; draw(true); });
    wrap.addEventListener('pointerleave', () => { if (drag === null) hideTip(); });

    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(() => draw(false)) : null;
    if (ro) ro.observe(wrap);

    return {
      update(s) { if (uiOf(s)) draw(false); },
      destroy() { if (ro) ro.disconnect(); tip.destroy(); },
    };
  },
};
