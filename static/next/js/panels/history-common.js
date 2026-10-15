// Shared by the History panels: month labels, the model series of
// /api/spend-history and their colours. Pure functions; no DOM.
import { familyOf, FAMILY_ORDER } from '../palette.js';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// "2026-09" -> {short: "Sep", long: "September 2026"-like "Sep 2026", year: 2026}
export function monthLabel(key) {
  const [y, m] = String(key || '').split('-').map(Number);
  const short = MONTHS[(m || 1) - 1] || String(key);
  return { short, long: `${short} ${y || ''}`.trim(), year: y };
}

// The spend-history keys: Claude Code rows are bare display names ("Opus 5.5"),
// Pi rows say "Provider / Model". The same model from both clients is one series.
const CLAUDE_FAMILIES = new Set(['opus', 'fable', 'sonnet', 'haiku']);

export function splitModelKey(key) {
  const s = String(key || '');
  const i = s.indexOf(' / ');
  if (i > 0) return { provider: s.slice(0, i), model: s.slice(i + 3) };
  return { provider: CLAUDE_FAMILIES.has(familyOf(s)) ? 'Anthropic' : 'Other', model: s };
}

// A series below this share of the whole range folds into "<provider> · other".
const MIN_SHARE = 0.006;

// months[] -> {series: [{id, provider, model, family, total, color}], providers: [{name, total, series}],
// rows: [{month, label, total, partial, values: Map(id -> usd)}]}. Colours come from palette
// variables only: each family gets its own colour, later models of a family get mixed shades.
export function buildSeries(months) {
  const totals = new Map();
  const rows = months.map((m) => {
    const values = new Map();
    for (const [key, usd] of Object.entries(m.by_model || {})) {
      if (typeof usd !== 'number' || !(usd > 0)) continue;
      const { provider, model } = splitModelKey(key);
      const id = `${provider}\u0000${model}`;
      values.set(id, (values.get(id) || 0) + usd);
      totals.set(id, (totals.get(id) || 0) + usd);
    }
    return { month: m.month, label: monthLabel(m.month), total: m.total, partial: Boolean(m.partial), values };
  });
  const grand = [...totals.values()].reduce((a, v) => a + v, 0) || 1;
  const fold = new Map();
  for (const [id, total] of totals) {
    const [provider] = id.split('\u0000');
    if (total / grand >= MIN_SHARE) continue;
    fold.set(id, `${provider}\u0000*`);
  }
  // Fold small series into one "other" series per provider, in every row.
  const folded = rows.map((r) => {
    const values = new Map();
    for (const [id, usd] of r.values) {
      const to = fold.get(id) || id;
      values.set(to, (values.get(to) || 0) + usd);
    }
    return { ...r, values };
  });
  const sums = new Map();
  for (const r of folded) for (const [id, usd] of r.values) sums.set(id, (sums.get(id) || 0) + usd);
  const series = [...sums.entries()].map(([id, total]) => {
    const [provider, model] = id.split('\u0000');
    const other = model === '*';
    return { id, provider, model: other ? 'Other models' : model, other, total, family: other ? 'other' : familyOf(model) };
  });
  const colored = colorSeries(series);
  const providers = groupProviders(colored);
  // Stack order: provider order, then the legend order inside it.
  const ordered = providers.flatMap((p) => p.series);
  return { series: ordered, providers, rows: folded };
}

function familyRank(fam) {
  const i = FAMILY_ORDER.indexOf(fam);
  return i < 0 ? FAMILY_ORDER.length : i;
}

const SHADES = [
  (v) => `var(${v})`,
  (v) => `color-mix(in srgb, var(${v}) 55%, var(--paper))`,
  (v) => `color-mix(in srgb, var(${v}) 60%, var(--ink))`,
  (v) => `color-mix(in srgb, var(${v}) 30%, var(--paper))`,
  (v) => `color-mix(in srgb, var(${v}) 80%, var(--paper))`,
];
const FAMILY_VAR = {
  opus: '--m-opus', fable: '--m-fable', sonnet: '--m-sonnet', haiku: '--m-haiku',
  deepseek: '--m-deepseek', glm: '--m-glm', gpt: '--m-gpt',
};

// Biggest model of a family gets the pure family colour; the next ones get shades.
function colorSeries(series) {
  const byFamily = new Map();
  for (const s of [...series].sort((a, b) => b.total - a.total)) {
    const n = byFamily.get(s.family) || 0;
    byFamily.set(s.family, n + 1);
    s.shade = n;
  }
  return series.map((s) => {
    const v = FAMILY_VAR[s.family] || '--m-other';
    const color = s.other ? 'var(--m-other)' : SHADES[s.shade % SHADES.length](v);
    return { ...s, color, hatch: s.other };
  });
}

function groupProviders(series) {
  const map = new Map();
  for (const s of series) {
    const p = map.get(s.provider) || { name: s.provider, total: 0, series: [] };
    p.total += s.total;
    p.series.push(s);
    map.set(s.provider, p);
  }
  const rank = (p) => (p.name === 'Anthropic' ? 0 : p.name === 'Other' ? 2 : 1);
  return [...map.values()]
    .map((p) => ({ ...p, series: p.series.slice().sort((a, b) => Number(a.other) - Number(b.other)
      || familyRank(a.family) - familyRank(b.family) || b.total - a.total) }))
    .sort((a, b) => rank(a) - rank(b) || b.total - a.total);
}

// A "nice" axis top and step for a maximum: 4 to 6 gridlines.
export function niceScale(max) {
  if (!(max > 0)) return { top: 1, step: 1 };
  const raw = max / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) || 10 * mag;
  return { top: Math.ceil(max / step) * step, step };
}

export const median = (xs) => {
  const s = xs.filter((x) => typeof x === 'number' && Number.isFinite(x)).sort((a, b) => a - b);
  if (!s.length) return null;
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};

// The one line every History and Lab view says about dollars.
export const API_NOTE = 'Dollars are API-equivalent at list price, not billed.';

// Redraw guard: true when any input changed by identity. Feeds are frozen
// objects replaced on every fetch, so identity is the change signal.
export function changed(prev, next) {
  return !prev || prev.length !== next.length || next.some((v, i) => v !== prev[i]);
}
