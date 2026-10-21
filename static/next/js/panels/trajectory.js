// Trajectory of the binding window (uPlot): readings so far, even pace, the
// server forecast to reset (pace.projected_pct), the TIGHT point
// (tight_epoch) or the 100% time (reaches_100_epoch). The browser draws
// lines between server values; it never extends a rate on its own.
// The month window fills the area under the billed line by model: the
// server sends the layers (split), the browser only stacks them.
import uPlot from '../../vendor/uplot/uPlot.esm.js';
import { h, patch, setText, segmented, panelHead } from '../dom.js';
import * as f from '../format.js';
import { bindingWindow, dataOf, nowFrom } from '../select.js';
import { feedLine } from '../panel.js';
import { modelColors } from './ledger-colors.js';

const INK = '#161616', BLUE = '#1d4f91', BLUE_TINT = '#cdd8e8', RED = '#d2342a', RED_INK = '#b3261e';
const PAPER = '#f3eee3', PAPER_2 = '#e8e1d2', MUTED = '#5d5950', GRID = 'rgba(22,22,22,.12)';
const FONT = '600 11px "DM Sans", system-ui, sans-serif';
const BOLD = '800 11px "DM Sans", sans-serif';
const H = 3600, D = 86400;

// The 24 h view: the last 24 h of the window, then a short stretch ahead,
// or the rest of the window when the reset comes within a day. Early in a
// window this is the whole window so far plus the next hours, not a sliver.
const AHEAD_MIN_S = 6 * H, DAY_FRAME_S = 12 * H;
// Windows of a day or less already fit the 24 h view: no range switch.
export const zoomable = (w) => Boolean(w) && w.kind !== 'month' && w.window_s > D;

// Default range: a sliver of data (early) or a sliver of window left (late)
// reads better in the 24 h view. The user's pick always wins.
export function autoZoom(w, now) {
  if (!zoomable(w)) return 'window';
  const p = w.pace || {};
  const frac = !f.missing(p.elapsed_fraction) ? p.elapsed_fraction
    : (now - w.window_start) / Math.max(1, w.resets_at - w.window_start);
  return frac < 0.25 || frac > 0.75 ? 'day' : 'window';
}

export function dayFrame(w, now) {
  const x0 = Math.max(w.window_start, now - 24 * H);
  if (w.resets_at - now <= 24 * H) return [x0, w.resets_at];
  return [x0, Math.min(w.resets_at, Math.max(now + AHEAD_MIN_S, x0 + DAY_FRAME_S))];
}

// A server line, clipped to the frame: the value of the straight line from
// (t0, v0) to (t1, v1) at t. Drawing only, no new forecast.
const along = (t0, v0, t1, v1, t) => (t1 === t0 ? v1 : v0 + (v1 - v0) * (t - t0) / (t1 - t0));

const roundUp = (v, step) => Math.ceil(v / step) * step;

// The y range of the 24 h view: from just under the first reading to just
// over the highest line in the frame; the limit (100%) shows when near.
function dayRange(w, pts, x0, x1, now) {
  const p = w.pace || {};
  const first = pts.find((q) => q[0] >= x0) || pts[pts.length - 1];
  const yMin = first ? Math.max(0, Math.floor((first[1] - 4) / 5) * 5) : 0;
  const tops = [w.pct, along(w.window_start, 0, w.resets_at, 100, x1)];
  if (!f.missing(p.projected_pct) && !f.missing(w.pct)) tops.push(along(now, w.pct, w.resets_at, p.projected_pct, x1));
  if (!f.missing(w.tight_epoch) && w.tight_epoch <= x1) tops.push(100);
  const top = Math.max(...tops.filter((v) => !f.missing(v)), ...pts.filter((q) => q[0] >= x0).map((q) => q[1]));
  const yMax = top >= 75 ? 100 : Math.min(100, Math.max(yMin + 20, roundUp(top + 5, 5)));
  return [yMin, yMax];
}

// A reset time can move by seconds between quota readings; a new window
// moves it by hours or days.
const SAME_WINDOW_S = 600;
const near = (a, b) => f.missing(a) || f.missing(b) || Math.abs(a - b) <= SAME_WINDOW_S;

// The trajectory answer if it is for window w, else null. When the binding
// window changes, the refetch keeps the old answer in the store until the
// new one comes: its points belong to another window and must not show.
export function trajectoryFor(traj, w) {
  if (!traj || !w) return null;
  return near(traj.start, w.window_start) && near(traj.end, w.resets_at) ? traj : null;
}

// The legend names the no-record layer only when it is big enough to see.
const UNRECORDED_MIN_PCT = 0.5;
const UNRECORDED = 'No session record';

// The split layers that belong to the kept points, bottom first:
// [{name, v: [pct per point]}]. null when the answer has no usable split.
function splitLayers(split, keep, n) {
  if (!keep.length || !split || !Array.isArray(split.models) || !split.models.length) return null;
  const pick = (arr) => (Array.isArray(arr) && arr.length === n ? keep.map((i) => +arr[i] || 0) : null);
  const layers = split.models.map((m) => ({ name: String(m.model), v: pick(m.pct) }));
  const rest = pick(split.unrecorded);
  if (!rest || layers.some((l) => !l.v)) return null;
  return [...layers, { name: UNRECORDED, v: rest, unrecorded: true }];
}

// Chart model from the feeds: everything the hooks draw, in data units.
function model(s, zoom) {
  const w = bindingWindow(dataOf(s.runway));
  if (!w) return null;
  const answer = trajectoryFor(dataOf(s.trajectory), w);
  const traj = answer || { points: [], resets: [] };
  const now = nowFrom(s);
  const raw = traj.points || [];
  const keep = raw.map((p, i) => i).filter((i) => Array.isArray(raw[i]) && raw[i][0] <= now + 60);
  const pts = keep.map((i) => raw[i]);
  const month = w.kind === 'month';
  const layers = month ? splitLayers(traj.split, keep, raw.length) : null;
  const p = w.pace || {};
  const day = zoom === 'day' && zoomable(w);
  // The server sends no start when its quota reading is too old to prove the
  // window. The frame then uses the nominal start; the even-pace line and the
  // start stamp stay out, because they would claim a start nobody knows.
  const knownStart = !f.missing(w.window_start);
  const fw = knownStart ? w : { ...w, window_start: w.resets_at - w.window_s };
  const [x0, x1] = day ? dayFrame(fw, now) : [fw.window_start, w.resets_at];
  const [yMin, yMax] = day ? dayRange(fw, pts, x0, x1, now)
    : [0, Math.max(100, month && !f.missing(p.projected_pct) ? p.projected_pct * 1.06 : 100)];
  const budget = month && dataOf(s.runway).budget ? dataOf(s.runway).budget.budget_usd : null;
  return {
    w, now, pts, layers, month, x0, x1, yMin, yMax, budget, day, cut: x1 < w.resets_at, knownStart, answered: Boolean(answer),
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
  const span = m.yMax - m.yMin;
  const step = !m.day ? 25 : span <= 30 ? 5 : span <= 60 ? 10 : 25;
  for (let v = Math.ceil(m.yMin / step) * step; v <= m.yMax; v += step) out.push(v);
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

// "Thu Oct 15, 3:00 AM": a 7-day window starts and ends on the same weekday.
const longStamp = (e) => `${f.dow(e)} ${f.dateShort(e)}, ${f.clock(e)}`;

const BIG = '400 15px "Archivo Black", sans-serif';
const forecastLabel = (m) => (m.month && m.budget ? `≈${f.usdShort(m.projected / 100 * m.budget)}` : `≈${Math.round(m.projected)}%`);
// Right padding that fits the forecast label: "≈211%" is wider than "≈99%".
const measure = document.createElement('canvas').getContext('2d');
function labelRoom(m) {
  if (!m || f.missing(m.projected)) return 16;
  measure.font = BIG;
  const wide = measure.measureText(forecastLabel(m)).width;
  measure.font = BOLD;
  const sub = m.cut ? measure.measureText('at reset').width : 0;
  return Math.max(m.month ? 72 : 58, Math.ceil(Math.max(wide, sub)) + 14);
}

export default {
  id: 'chart',
  label: 'Trajectory',
  mount(el, ctx) {
    // null until the user picks a range; until then autoZoom() decides.
    let picked = null;
    let m = null;
    // Layer colours for the current split (chrome() sets them).
    let colors = null;
    const zoomFor = (s) => picked || autoZoom(bindingWindow(dataOf(s.runway)), nowFrom(s));
    const head = panelHead('Trajectory');
    const seg = segmented('Chart range', [['window', 'Window'], ['day', 'Last 24 h']], 'window', (z) => { picked = z; redraw(); });
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
    const Yraw = (u, v) => u.valToPos(v, 'y', true);

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

    // Diagonal hatch for billed spend that no session record explains. Made
    // once for each pixel ratio, not at each draw.
    let hatchFor = null;
    function hatch(c) {
      if (hatchFor && hatchFor.ratio === uPlot.pxRatio) return hatchFor.pattern;
      const size = Math.max(4, Math.round(px(6)));
      const tile = document.createElement('canvas');
      tile.width = tile.height = size;
      const g = tile.getContext('2d');
      g.fillStyle = PAPER; g.fillRect(0, 0, size, size);
      g.strokeStyle = MUTED; g.lineWidth = px(1);
      g.beginPath(); g.moveTo(0, size); g.lineTo(size, 0); g.stroke();
      hatchFor = { ratio: uPlot.pxRatio, pattern: c.createPattern(tile, 'repeat') };
      return hatchFor.pattern;
    }
    const layerFill = (c, l) => (l.unrecorded ? hatch(c) : colors.color(l.name).fill);

    // The split, stacked bottom first under the billed line (drawn after
    // the grid, before the line).
    function drawLayers(u) {
      if (!m || !m.layers || !colors || !m.pts.length) return;
      const c = u.ctx, xs = m.pts.map((p) => X(u, p[0]));
      let base = m.pts.map(() => 0);
      clipped(c, u, () => {
        for (const l of m.layers) {
          const top = base.map((b, i) => b + l.v[i]);
          c.beginPath();
          xs.forEach((x, i) => (i ? c.lineTo(x, Yraw(u, top[i])) : c.moveTo(x, Yraw(u, top[i]))));
          for (let i = xs.length - 1; i >= 0; i--) c.lineTo(xs[i], Yraw(u, base[i]));
          c.closePath();
          c.fillStyle = layerFill(c, l); c.fill();
          base = top;
        }
      });
    }

    // Lines run between server values in data units and are clipped to the
    // plot, so a frame that cuts a line keeps its true slope.
    function clipped(c, u, draw) {
      const { top, left, width, height } = u.bbox;
      c.save(); c.beginPath(); c.rect(left, top - px(3), width, height + px(6)); c.clip();
      draw(); c.restore();
    }

    function drawForecast(u, c, nx) {
      const w = m.w, right = u.bbox.left + u.bbox.width;
      clipped(c, u, () => line(c, nx, Yraw(u, m.cur), X(u, w.resets_at), Yraw(u, m.projected), RED, 3, [7, 5]));
      if (!m.cut) {
        const ex = X(u, m.x1), ey = Y(u, m.projected);
        c.fillStyle = RED; c.fillRect(ex - px(4), ey - px(4), px(8), px(8));
        label(c, forecastLabel(m), ex + px(8), ey + px(m.projected > 97 ? 14 : 4), RED_INK, 'left', BIG);
        return;
      }
      // The frame ends before the reset: say where the line goes.
      const ve = along(m.now, m.cur, w.resets_at, m.projected, m.x1);
      const ey = Y(u, Math.max(m.yMin, ve));
      label(c, forecastLabel(m), right + px(6), ey + px(5), RED_INK, 'left', BIG);
      label(c, 'at reset', right + px(6), ey + px(19), RED_INK, 'left', BOLD);
    }

    function drawOverlays(u) {
      if (!m) return;
      const c = u.ctx, { top, left, width, height } = u.bbox;
      const bottom = top + height, right = left + width, nx = X(u, m.now);
      const w = m.w, limitIn = m.yMax >= 100;
      if (!m.month && m.knownStart) clipped(c, u, () => line(c, X(u, w.window_start), Yraw(u, 0), X(u, w.resets_at), Yraw(u, 100), INK, 1.6, [6, 5]));
      if (limitIn) {
        line(c, left, Y(u, 100), right, Y(u, 100), INK, 3, []);
        label(c, m.month ? 'BUDGET' : 'LIMIT', left + px(4), Y(u, 100) - px(6), INK, 'left', BOLD);
      }
      const hasCur = !f.missing(m.cur);
      if (hasCur && !f.missing(m.projected)) drawForecast(u, c, nx);
      const hit = !f.missing(m.tight) ? m.tight : (!f.missing(m.reaches) && m.reaches < w.resets_at ? m.reaches : null);
      if (hasCur && hit !== null && limitIn) {
        const hx = X(u, hit), hy = Y(u, 100);
        if (!f.missing(m.tight)) clipped(c, u, () => line(c, nx, Yraw(u, m.cur), hx, hy, RED_INK, 2, [2, 4]));
        if (hit <= m.x1) {
          c.save(); c.beginPath(); c.arc(hx, hy, px(5), 0, Math.PI * 2);
          c.fillStyle = PAPER; c.fill(); c.lineWidth = px(2.5); c.strokeStyle = RED_INK; c.stroke(); c.restore();
          label(c, f.when(hit, m.now), hx, hy - px(9), RED_INK, 'center', BOLD);
        }
      }
      c.fillStyle = BLUE;
      for (const r of m.resets) c.fillRect(X(u, r) - px(1.5), bottom - px(12), px(3), px(12));
      line(c, nx, top - px(6), nx, bottom, INK, 2, []);
      if (hasCur) { c.fillStyle = INK; c.fillRect(nx - px(5), Y(u, m.cur) - px(5), px(10), px(10)); }
      label(c, 'NOW', nx - px(6), bottom + px(18), INK, 'right', BOLD);
      const end = m.month ? f.dateShort(m.x1 - 1) : m.cut ? f.clock(m.x1) : 'reset';
      if (width > px(320)) label(c, end, right, bottom + px(18), MUTED, 'right');
    }

    const money = (v) => (m.budget ? f.usd(v / 100 * m.budget, { cents: false }) : `${v.toFixed(1)}%`);

    // " · Opus 5.5 $1,620 · Sonnet 5.5 $322" at point i, largest first.
    // The no-record part shows only when it rounds to a dollar or more.
    function splitText(i) {
      if (!m.layers || i < 0 || i >= m.pts.length) return '';
      return m.layers
        .filter((l) => (l.unrecorded ? l.v[i] / 100 * (m.budget || 0) >= 1 : l.v[i] > 0))
        .map((l) => ` · ${l.unrecorded ? 'no session record' : l.name} ${money(l.v[i])}`).join('');
    }

    function onCursor(u) {
      if (!m) return;
      const i = u.cursor.idx;
      const pace = (t) => (t - m.w.window_start) / (m.w.resets_at - m.w.window_start) * 100;
      if (i === null || i === undefined || u.data[0][i] === undefined) return showLatest();
      const t = u.data[0][i], v = u.data[1][i];
      patch(readout, h('b', null, f.dayClock(t)), m.month ? ` · ${money(v)} billed${splitText(i)}`
        : ` · ${v.toFixed(1)}% used${m.knownStart ? ` · even pace ${pace(t).toFixed(1)}%` : ''}`);
    }
    function showLatest() {
      if (!m || f.missing(m.cur)) return patch(readout);
      const p = m.w.pace || {};
      // The split belongs to the last meter point; the window pct can be a
      // newer estimate, so the split shows only when the two agree.
      const last = m.pts.length - 1;
      const split = last >= 0 && Math.abs(m.pts[last][1] - m.cur) < 0.01 ? splitText(last) : '';
      patch(readout, h('b', null, 'Latest'), m.month ? ` · ${money(m.cur)} billed${split}` : ` · ${m.cur.toFixed(1)}% used`,
        f.missing(p.expected_pct) ? '' : ` · ${m.month ? `workday pace ${money(p.expected_pct)}` : `even pace ${p.expected_pct.toFixed(1)}%`}`);
    }

    const plot = new uPlot({
      width: 400, height: 300,
      // Room on the right for the forecast label ("≈99%" or "≈$3.4K"), if any.
      padding: [18, () => labelRoom(m), 0, 0],
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
            // Keep clear of the NOW and end labels, which the draw hook paints.
            const x = u.valToPos(t, 'x');
            if (!m || Math.abs(x - u.valToPos(m.now, 'x')) < 56 || Math.abs(x - u.valToPos(m.x1, 'x')) < (m.cut ? 76 : 44)) return '';
            return xLabel(m, t, u.bbox.width / uPlot.pxRatio);
          }) },
        { stroke: MUTED, font: FONT, size: 46, gap: 6, ticks: { show: false }, grid: { stroke: GRID, width: 1 },
          splits: () => (m ? ySplits(m) : [0, 100]), values: (u, splits) => splits.map((v) => (m ? m.yFmt(v) : '')) },
      ],
      // With a model split the layers fill the area (drawLayers), not the series.
      series: [{}, { stroke: INK, width: 2.6, fill: () => (m && m.layers ? null : BLUE_TINT), points: { show: false }, spanGaps: true }],
      hooks: { drawClear: [drawFuture], drawAxes: [drawLayers], draw: [drawOverlays], setCursor: [onCursor] },
    }, [[], []], host);
    const resize = new ResizeObserver(() => {
      if (host.clientWidth > 0) plot.setSize({ width: host.clientWidth, height: host.clientHeight });
    });
    resize.observe(host);
    // While the pointer is on the plot, a redraw for new data re-reads the
    // point under it instead of resetting the readout to the latest value.
    let hovering = false;
    host.addEventListener('mouseenter', () => { hovering = true; });
    host.addEventListener('mouseleave', () => { hovering = false; showLatest(); });

    let lastState = null;
    let drawnKey = '';
    function redraw(force = true) {
      if (!lastState) return;
      const zoom = zoomFor(lastState);
      seg.select(zoom);
      m = model(lastState, zoom);
      if (!m) return;
      // A 5 s poll that changed nothing on the chart must not reset a hover readout.
      const key = JSON.stringify([zoom, m.yMin, m.yMax, m.cur, m.projected, m.tight, m.reaches, m.pts.length, m.pts.at(-1), m.x0, m.x1, Math.floor(m.now / 30),
        m.layers && m.layers.map((l) => [l.name, l.v.at(-1)])]);
      if (!force && key === drawnKey) return;
      drawnKey = key;
      const xs = m.pts.map((p) => p[0]), ys = m.pts.map((p) => p[1]);
      plot.setData([xs, ys], true);
      plot.redraw(true, true);
      const { left, top } = plot.cursor;
      if (hovering && left >= 0) plot.setCursor({ left, top }); else showLatest();
    }

    function chrome() {
      const w = m.w;
      setText(head.titleEl, m.month ? w.label : `${String(w.label).replace(' · ', ' ')} window`);
      seg.el.hidden = !zoomable(w);
      const stamp = w.window_s >= 2 * D ? longStamp : f.dayClock;
      setText(sub, m.month ? `${f.dateShort(w.window_start + D / 2)} to ${f.dateShort(w.resets_at - D / 2)}`
        : m.knownStart ? `${stamp(w.window_start)} to ${stamp(w.resets_at)}` : `Resets ${stamp(w.resets_at)}`);
      const hasProj = !f.missing(m.projected), hasTight = !f.missing(w.tight_epoch);
      const models = m.layers ? m.layers.filter((l) => !l.unrecorded) : [];
      colors = m.layers ? modelColors(Object.fromEntries(models.map((l) => [l.name, l.v.at(-1)]))) : null;
      const rest = m.layers ? m.layers.find((l) => l.unrecorded) : null;
      const showRest = rest && Math.max(...rest.v) >= UNRECORDED_MIN_PCT;
      const items = m.month
        ? [[m.layers ? 'line' : 'area', 'Billed'],
          ...models.map((l) => ['swatch', l.name, colors.color(l.name).fill]),
          ...(showRest ? [['hatch', UNRECORDED]] : []),
          ...(hasProj ? [['proj', 'Forecast at month end']] : [])]
        : [['area', 'Used'], ...(m.knownStart ? [['pace', 'Even pace']] : []), ...(hasProj ? [['proj', 'Forecast at reset']] : []),
          ...(hasTight ? [['tight', `At the ${w.kind === 'short' ? '1-hour' : '6-hour'} rate`]] : []), ...(m.resets.length ? [['reset', 'Reset']] : [])];
      patch(legend, items.map(([cls, t, fill]) => h('span', { class: 'lg', 'data-key': `${cls}:${t}` },
        h('i', { class: cls, style: fill ? { background: fill } : null }), t)));
      const noHistory = !m.pts.length;
      setText(empty, noHistory && m.answered ? 'This window has no reading history. The chart shows the forecast only.'
        : m.yMax < 100 ? 'The limit (100%) is above this 24 h view.' : '');
      empty.hidden = !empty.textContent;
      setText(note, m.month
        ? ['Billed spend from the Anthropic billing meter.',
          m.layers ? 'The colours divide it by model, from the recorded work sessions.' : '',
          showRest ? 'The hatched part is billed spend that has no session record.' : '',
          'The grey part of the month is still ahead.',
          hasProj ? 'The red dashed line is the server forecast for month end.' : ''].filter(Boolean).join(' ')
        : ['The grey part of the window is still ahead.',
          hasProj ? 'The red dashed line is the server forecast at the window average.' : 'The forecast starts when the window has more data.',
          hasTight ? 'The dotted line shows when the recent rate reaches the limit.' : ''].join(' ').trim());
    }

    function update(s) {
      lastState = s;
      const line1 = feedLine(s.runway, 'the limit feed');
      if (line1 || !bindingWindow(dataOf(s.runway))) {
        patch(state, line1);
        body.hidden = true; seg.el.hidden = true;
        return;
      }
      patch(state, s.trajectory && s.trajectory.status === 'error' && !s.trajectory.data
        ? h('div', { class: 'p-state' }, 'The reading history is unavailable.', h('small', null, 'The forecast lines still show.')) : null);
      body.hidden = false;
      m = model(s, zoomFor(s));
      chrome();
      redraw(false);
    }

    return { update, destroy: () => { resize.disconnect(); plot.destroy(); } };
  },
};
