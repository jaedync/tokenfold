// Drawer parts: the fan-out chart (orchestrator bar with prompt ticks, one
// row per parallel subagent run) and the run tree (biggest first, children
// under their parent run).
import { h } from '../dom.js';
import * as f from '../format.js';
import { svg, swatch, money } from './ledger-kit.js';

const INK = '#161616', MUTED = '#5d5950', PAPER = '#f3eee3';
const RH = 9, TOP = 38, MAX_ROWS = 16, TREE_FEW = 8;

// Drawn at once, at `width` (the drawer's gantt width now, 0 before it
// opened: then its max width). The drawer patches the result into its kept
// gantt, so a poll changes the bars in place; its ResizeObserver redraws
// when the measured width differs from data-w.
export function renderGantt(s, promptTimes, c, live, width) {
  const box = h('div', { class: 'gantt' });
  draw(box, s, promptTimes, c, live, width || 480);
  return box;
}

function draw(box, s, promptTimes, c, live, W) {
  box.dataset.w = String(W);
  const t0 = s.start, t1 = Math.max(s.end, s.start + 1);
  const x = (t) => ((Math.min(t1, Math.max(t0, t)) - t0) / (t1 - t0)) * (W - 4) + 2;
  const runs = s.runs.slice().sort((a, b) => a.start - b.start);
  const rows = [];
  const placed = runs.map((r) => {
    let k = rows.findIndex((e) => e <= r.start);
    if (k < 0 && rows.length < MAX_ROWS) { rows.push(0); k = rows.length - 1; }
    if (k < 0) k = rows.indexOf(Math.min(...rows));
    rows[k] = r.end + 60;
    return { r, k };
  });
  const H = TOP + Math.max(1, rows.length) * RH + 30;
  const col = c.color(s.model);
  const g = [
    svg('text', { x: 2, y: 9, class: 'gt-k', text: `ORCHESTRATOR · ${String(s.model).toUpperCase()}` }),
    svg('rect', { x: 2, y: 16, width: W - 4, height: 14, fill: col.fill, stroke: INK, 'stroke-width': 1.5 }),
  ];
  // Your prompts, as ticks on the orchestrator bar.
  for (const t of promptTimes || []) {
    g.push(svg('rect', { x: x(t) - 1, y: 13, width: 2.5, height: 20, fill: col.text === '#ffffff' ? PAPER : INK, stroke: INK, 'stroke-width': 0.5 }));
  }
  for (const { r, k } of placed) {
    const a = x(r.start), b = Math.max(a + 2, x(r.end));
    g.push(svg('rect', { x: a, y: TOP + k * RH, width: b - a, height: RH - 3, fill: c.color(r.model).fill, stroke: b - a > 4 ? INK : null, 'stroke-width': 0.8 },
      svg('title', { text: `${r.label || r.model} · ${r.model} · ${money(r.cost)}` })));
  }
  if (!runs.length) g.push(svg('text', { x: 2, y: TOP + 8, class: 'gt-note', text: 'This session ran alone.' }));
  const yb = H - 12;
  g.push(svg('line', { x1: 0, x2: W, y1: yb - 10, y2: yb - 10, stroke: INK, 'stroke-width': 2 }));
  g.push(svg('text', { x: 2, y: yb + 4, class: 'gt-t', text: f.when(t0, t1) }));
  g.push(svg('text', { x: W - 2, y: yb + 4, 'text-anchor': 'end', class: 'gt-t', text: live ? 'now' : f.when(t1, t0) }));
  if (W > 400) g.push(svg('text', { x: W / 2, y: yb + 4, 'text-anchor': 'middle', class: 'gt-note', text: promptTimes ? 'ticks: your prompts · rows: subagents side by side' : 'rows: subagents side by side' }));
  box.replaceChildren(svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': `Orchestrator and ${runs.length} subagent runs across the session` }, ...g));
}

function ordered(runs) {
  const kids = new Map();
  const ids = new Set(runs.map((r) => r.id));
  for (const r of runs) {
    const p = r.parent && ids.has(r.parent) ? r.parent : null;
    if (!kids.has(p)) kids.set(p, []);
    kids.get(p).push(r);
  }
  const out = [];
  const walk = (parent, depth) => {
    for (const r of (kids.get(parent) || []).sort((a, b) => b.cost - a.cost)) {
      out.push({ r, depth: Math.min(depth, 2) });
      walk(r.id, depth + 1);
    }
  };
  walk(null, 1);
  return out;
}

function row(c, depth, name, sub, active, cost, model) {
  return h('li', { class: depth ? '' : 't-main', 'data-depth': String(depth) }, swatch(c.color(model).fill),
    h('span', { class: 't-name' }, name, h('small', null, sub)),
    h('span', { class: 't-dur' }, f.shortDur(active)), h('span', { class: 't-cost' }, money(cost)));
}

// Claude Code runs have no label (ingest drops the Task text): the model and
// length name them instead. `all`: show every run; `onAll`: the "Show all"
// click. The drawer keeps that state by session id and renders again, so the
// button holds no built node (patch() may keep the old ones instead).
export function runTree(s, c, all = false, onAll = null) {
  const list = ordered(s.runs);
  const ul = h('ul', { class: 'tree' }, row(c, 0, 'Orchestrator', `${s.model}`, s.active, s.orch, s.model),
    ...(all ? list : list.slice(0, TREE_FEW)).map(({ r, depth }) => row(c, depth,
      r.label || `${r.model} run`, `${r.label ? `${r.model} · ` : ''}${f.int(r.tools || 0)} tools`, r.active, r.cost, r.model)));
  const folded = !all && list.length > TREE_FEW;
  return h('div', null, ul, folded ? h('button', { type: 'button', class: 'tree-more', onclick: onAll }, `Show all ${list.length} runs`) : null);
}
