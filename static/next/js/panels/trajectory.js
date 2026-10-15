// Trajectory of the binding window (uPlot): readings so far, even pace, the
// server forecast to reset (pace.projected_pct), the TIGHT point
// (tight_epoch) or the 100% time (reaches_100_epoch). The browser draws
// lines between server values; it never extends a rate on its own.
import uPlot from '../../vendor/uplot/uPlot.esm.js';
import { h, fill, setText, segmented, panelHead } from '../dom.js';
import * as f from '../format.js';
import { bindingWindow, dataOf, nowFrom } from '../select.js';
import { feedLine } from '../panel.js';

const INK = '#161616', BLUE = '#1d4f91', BLUE_TINT = '#cdd8e8', RED = '#d2342a', RED_INK = '#b3261e';
const PAPER = '#f3eee3', PAPER_2 = '#e8e1d2', MUTED = '#5d5950', GRID = 'rgba(22,22,22,.12)';
const FONT = '600 11px "DM Sans", system-ui, sans-serif';
const H = 3600, D = 86400;

// Chart model from the feeds: everything the hooks draw, in data units.
function model(s, zoom) {
  const w = bindingWindow(dataOf(s.runway));
  if (!w) return null;
  const traj = dataOf(s.trajectory) || { points: [], resets: [] };
  const now = nowFrom(s);
  const pts = (traj.points || []).filter((p) => Array.isArray(p) && p[0] <= now + 60);
  const month = w.kind === 'month';
  const p = w.pace || {};
  const day = zoom === 'day' && !month;
  const x0 = day ? Math.max(w.window_start, now - 24 * H) : w.window_start;
  const x1 = w.resets_at;
  let yMin = 0;
  if (day && pts.length) {
    const first = pts.find((q) => q[0] >= x0) || pts[pts.length - 1];
    yMin = Math.max(0, Math.floor((first[1] - 4) / 5) * 5);
  }
  const yMax = Math.max(100, month && !f.missing(p.projected_pct) ? p.projected_pct * 1.06 : 100);
  const budget = month && dataOf(s.runway).budget ? dataOf(s.runway).budget.budget_usd : null;
  return {
    w, now, pts, month, x0, x1, yMin, yMax, budget, day,
    cur: w.pct, projected: p.projected_pct, tight: w.tight_epoch,
    reaches: p.reaches_100_epoch, resets: (traj.resets || []).filter((t) => t > x0 && t <= now),
    yFmt: (v) => (month && budget ? f.usdShort(v / 100 * budget) : `${Math.round(v)}%`),
  };
}

const MONEY_STEPS = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000];

function ySplits(m) {
  const out = [];
  if (m.month && m.budget) {
    // Round dollar steps, so the axis reads $1K, $2K, $3K and not $750, $2.3K.
    const step = MONEY_STEPS.find((v) => m.budget / v <= 4) || MONEY_STEPS[MONEY_STEPS.length - 1];
    for (let usd = 0; usd / m.budget * 100 <= m.yMax; usd += step) out.push(usd / m.budget * 100);
    return out;
  }
  const step = m.day ? 5 : 25;
  for (let v = m.yMin; v <= 100; v += step) out.push(v);
  return out;
}

// Local midnights for a week, Mondays for a month, every 3 or 6 hours for 24 h.
function xSplits(m, width) {
  const out = [];
  if (m.day) {
    const step = (width < 420 ? 6 : 3) * H;
    for (let t = f.localMidnight(m.x0); t < m.x1; t += step) if (t > m.x0) out.push(t);
  } else if (m.x1 - m.x0 > 10 * D) {
    for (let t = f.localMidnight(m.x0) + D; t < m.x1; t = f.localMidnight(t + D + H)) {
      if (new Date(t * 1000).getDay() === 1) out.push(t);
    }
  } else {
    for (let t = f.localMidnight(m.x0) + D; t < m.x1; t += D) out.push(f.localMidnight(t + H));
  }
  return out;
}

function xLabel(m, t, width) {
  if (m.day) return f.localMidnight(t) === t ? f.dow(t) : f.hourShort(t);
  if (m.x1 - m.x0 > 10 * D) return width < 420 ? String(new Date(t * 1000).getDate()) : f.dateShort(t);
  return f.dow(t);
}

export default {
  id: 'chart',
  label: 'Trajectory',
  mount(el, ctx) {
    let zoom = 'window';
    let m = null;
    const head = panelHead('Trajectory');
    const seg = segmented('Chart range', [['window', 'Window'], ['day', 'Last 24 h']], zoom, (z) => { zoom = z; redraw(); });
    seg.el.classList.add('zoom');
    head.el.append(seg.el);
    el.setAttribute('aria-labelledby', head.id);
    const sub = h('div', { class: 'panel-sub chart-sub' });
    const legend = h('div', { class: 'legend' });
    const readout = h('div', { class: 'readout' });
    const empty = h('p', { class: 'chart-empty' });
    const host = h('div', { class: 'plot', role: 'img', 'aria-label': 'Trajectory chart' });
    const note = h('p', { class: 'chart-note' });
    const state = h('div');
    const body = h('div', { style: { display: 'contents' } }, sub, legend, readout, empty, host, note);
    el.append(head.el, state, body);

    const px = (v) => v * uPlot.pxRatio;
    const X = (u, t) => u.valToPos(t, 'x', true);
    const Y = (u, v) => u.valToPos(Math.min(v, m.yMax), 'y', true);

    function line(c, x1, y1, x2, y2, color, width, dash) {
      c.save();
      c.strokeStyle = color; c.lineWidth = px(width); c.setLineDash(dash.map(px));
      c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
      c.restore();
    }
    function label(c, text, x, y, color, align = 'left', font = FONT) {
      c.save();
      c.font = font.replace(/(\d+)px/, (_, n) => `${px(+n)}px`);
      c.fillStyle = color; c.textAlign = align; c.fillText(text, x, y);
      c.restore();
    }

    function drawFuture(u) {
      if (!m) return;
      const c = u.ctx, { top, height } = u.bbox;
      const a = Math.max(X(u, m.now), u.bbox.left);
      c.save(); c.fillStyle = PAPER_2; c.fillRect(a, top, u.bbox.left + u.bbox.width - a, height); c.restore();
    }

    function drawOverlays(u) {
      if (!m) return;
      const c = u.ctx, { top, left, width, height } = u.bbox;
      const bottom = top + height, right = left + width, nx = X(u, m.now);
      if (!m.month) line(c, X(u, m.w.window_start), Y(u, 0), X(u, m.x1), Y(u, 100), INK, 1.6, [6, 5]);
      line(c, left, Y(u, 100), right, Y(u, 100), INK, 3, []);
      label(c, m.month ? 'BUDGET' : 'LIMIT', left + px(4), Y(u, 100) - px(6), INK, 'left', '800 11px "DM Sans", sans-serif');
      const hasCur = !f.missing(m.cur);
      if (hasCur && !f.missing(m.projected)) {
        const ex = X(u, m.x1), ey = Y(u, m.projected);
        line(c, nx, Y(u, m.cur), ex, ey, RED, 3, [7, 5]);
        c.fillStyle = RED; c.fillRect(ex - px(4), ey - px(4), px(8), px(8));
        const lbl = m.month && m.budget ? `≈${f.usdShort(m.projected / 100 * m.budget)}` : `≈${Math.round(m.projected)}%`;
        label(c, lbl, ex + px(8), ey + px(m.projected > 97 ? 14 : 4), RED_INK, 'left', '400 15px "Archivo Black", sans-serif');
      }
      const hit = !f.missing(m.tight) ? m.tight : (!f.missing(m.reaches) && m.reaches < m.x1 ? m.reaches : null);
      if (hasCur && hit !== null) {
        const hx = X(u, hit), hy = Y(u, 100);
        if (!f.missing(m.tight)) line(c, nx, Y(u, m.cur), hx, hy, RED_INK, 2, [2, 4]);
        c.save(); c.beginPath(); c.arc(hx, hy, px(5), 0, Math.PI * 2);
        c.fillStyle = PAPER; c.fill(); c.lineWidth = px(2.5); c.strokeStyle = RED_INK; c.stroke(); c.restore();
        label(c, f.when(hit, m.now), hx, hy - px(9), RED_INK, 'center', '800 11px "DM Sans", sans-serif');
      }
      c.fillStyle = BLUE;
      for (const r of m.resets) c.fillRect(X(u, r) - px(1.5), bottom - px(12), px(3), px(12));
      line(c, nx, top - px(6), nx, bottom, INK, 2, []);
      if (hasCur) { c.fillStyle = INK; c.fillRect(nx - px(5), Y(u, m.cur) - px(5), px(10), px(10)); }
      label(c, 'NOW', nx - px(6), bottom + px(18), INK, 'right', '800 11px "DM Sans", sans-serif');
      if (width > px(320)) label(c, m.month ? f.dateShort(m.x1 - 1) : 'reset', right, bottom + px(18), MUTED, 'right');
    }

    const money = (v) => (m.budget ? f.usd(v / 100 * m.budget, { cents: false }) : `${v.toFixed(1)}%`);

    function onCursor(u) {
      if (!m) return;
      const i = u.cursor.idx;
      const pace = (t) => (t - m.w.window_start) / (m.x1 - m.w.window_start) * 100;
      if (i === null || i === undefined || u.data[0][i] === undefined) return showLatest();
      const t = u.data[0][i], v = u.data[1][i];
      fill(readout, h('b', null, f.dayClock(t)), m.month ? ` · ${money(v)} billed` : ` · ${v.toFixed(1)}% used · even pace ${pace(t).toFixed(1)}%`);
    }
    function showLatest() {
      if (!m || f.missing(m.cur)) return fill(readout);
      const p = m.w.pace || {};
      fill(readout, h('b', null, 'Latest'), m.month ? ` · ${money(m.cur)} billed` : ` · ${m.cur.toFixed(1)}% used`,
        f.missing(p.expected_pct) ? '' : ` · ${m.month ? `workday pace ${money(p.expected_pct)}` : `even pace ${p.expected_pct.toFixed(1)}%`}`);
    }

    const plot = new uPlot({
      width: 400, height: 300,
      // Room on the right for the forecast label ("≈99%" or "≈$3.4K").
      padding: [18, () => (m && m.month ? 72 : 58), 0, 0],
      legend: { show: false },
      select: { show: false },
      cursor: { y: false, drag: { x: false, y: false }, points: { size: 9, fill: BLUE, stroke: PAPER, width: 2 } },
      scales: {
        x: { time: true, range: () => (m ? [m.x0, m.x1] : [0, 1]) },
        y: { range: () => (m ? [m.yMin, m.yMax] : [0, 100]) },
      },
      axes: [
        { stroke: MUTED, font: FONT, size: 30, gap: 6, ticks: { show: false }, grid: { stroke: GRID, width: 1 },
          splits: (u) => (m ? xSplits(m, u.bbox.width / uPlot.pxRatio) : []),
          values: (u, splits) => splits.map((t) => {
            if (!m || Math.abs(u.valToPos(t, 'x') - u.valToPos(m.now, 'x')) < 56 || (t > m.now && !m.month)) return '';
            return xLabel(m, t, u.bbox.width / uPlot.pxRatio);
          }) },
        { stroke: MUTED, font: FONT, size: 46, gap: 6, ticks: { show: false }, grid: { stroke: GRID, width: 1 },
          splits: () => (m ? ySplits(m) : [0, 100]), values: (u, splits) => splits.map((v) => (m ? m.yFmt(v) : '')) },
      ],
      series: [{}, { stroke: INK, width: 2.6, fill: BLUE_TINT, points: { show: false }, spanGaps: true }],
      hooks: { drawClear: [drawFuture], draw: [drawOverlays], setCursor: [onCursor] },
    }, [[], []], host);
    const resize = new ResizeObserver(() => {
      if (host.clientWidth > 0) plot.setSize({ width: host.clientWidth, height: host.clientHeight });
    });
    resize.observe(host);
    host.addEventListener('mouseleave', showLatest);

    let lastState = null;
    let drawnKey = '';
    function redraw(force = true) {
      if (!lastState) return;
      m = model(lastState, zoom);
      if (!m) return;
      // A 5 s poll that changed nothing on the chart must not reset a hover readout.
      const key = JSON.stringify([zoom, m.cur, m.projected, m.tight, m.reaches, m.pts.length, m.pts.at(-1), m.x0, m.x1, Math.floor(m.now / 30)]);
      if (!force && key === drawnKey) return;
      drawnKey = key;
      const xs = m.pts.map((p) => p[0]), ys = m.pts.map((p) => p[1]);
      plot.setData([xs, ys], true);
      plot.redraw(true, true);
      showLatest();
    }

    function chrome(s) {
      const w = m.w;
      setText(head.titleEl, m.month ? w.label : `${String(w.label).replace(' · ', ' ')} window`);
      seg.el.hidden = m.month;
      setText(sub, m.month ? `${f.dateShort(w.window_start + D / 2)} to ${f.dateShort(w.resets_at - D / 2)}`
        : `${f.dayClock(w.window_start)} to ${f.dayClock(w.resets_at)}`);
      const items = m.month
        ? [['area', 'Billed'], ['proj', 'Forecast at month end']]
        : [['area', 'Used'], ['pace', 'Even pace'], ['proj', 'Forecast at reset'], ...(f.missing(w.tight_epoch) ? [] : [['tight', `At the ${w.kind === 'short' ? '1-hour' : '6-hour'} rate`]]), ...(m.resets.length ? [['reset', 'Reset']] : [])];
      fill(legend, items.map(([cls, t]) => h('span', { class: 'lg' }, h('i', { class: cls }), t)));
      const noHistory = !m.pts.length;
      setText(empty, noHistory && dataOf(s.trajectory) ? 'This window has no reading history. The chart shows the forecast only.' : '');
      empty.hidden = !empty.textContent;
      setText(note, m.month
        ? 'Billed spend from the provider meter. The grey part of the month is still ahead. The red dashed line is the server forecast for month end.'
        : 'The grey part of the window is still ahead. The red dashed line is the server forecast at the window average. The dotted line shows when the recent rate reaches the limit.');
    }

    function update(s) {
      lastState = s;
      const line1 = feedLine(s.runway, 'the limit feed');
      if (line1 || !bindingWindow(dataOf(s.runway))) {
        fill(state, line1);
        body.hidden = true; seg.el.hidden = true;
        return;
      }
      fill(state, s.trajectory && s.trajectory.status === 'error' && !s.trajectory.data
        ? h('div', { class: 'p-state' }, 'The reading history is unavailable.', h('small', null, 'The forecast lines still show.')) : null);
      body.hidden = false;
      m = model(s, zoom);
      chrome(s);
      redraw(false);
    }

    return { update, destroy: () => { resize.disconnect(); plot.destroy(); } };
  },
};
