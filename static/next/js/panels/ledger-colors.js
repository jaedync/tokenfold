// Ledger model colours, from the shared family palette (palette.js). A family
// has one base colour; when a family holds several models (Opus 5.5 and
// Opus 4.8) the one with the most spend keeps the base and the others get
// lighter tints of it, so a migration reads as one hue changing shade.
// Colours never come from a server string.

import { familyColor, familyOf, FAMILY_ORDER } from '../palette.js';

const TINTS = [0, 0.4, 0.6, 0.75];
const INK = '#161616';
const PAPER = '#f3eee3';

function cssVar(expr) {
  const m = /var\((--[\w-]+)\)/.exec(expr);
  if (!m || typeof document === 'undefined') return null;
  return getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim() || null;
}

function rgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || '').trim());
  if (!m) return [154, 147, 133];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const toHex = (c) => `#${c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`;
const mix = (a, b, t) => toHex(rgb(a).map((v, i) => v + (rgb(b)[i] - v) * t));

function luminance(hex) {
  const [r, g, b] = rgb(hex).map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
export const contrast = (a, b) => {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};

const rank = (fam) => { const i = FAMILY_ORDER.indexOf(fam); return i < 0 ? FAMILY_ORDER.length : i; };

// costs: {model: usd} -> {order: [model], color(model) -> {fill, text, line}}
export function modelColors(costs) {
  const byFam = new Map();
  for (const [model, usd] of Object.entries(costs)) {
    const fam = familyOf(model);
    if (!byFam.has(fam)) byFam.set(fam, []);
    byFam.get(fam).push([model, usd || 0]);
  }
  const map = new Map();
  const order = [];
  const fams = [...byFam.keys()].sort((a, b) => rank(a) - rank(b));
  for (const fam of fams) {
    const base = cssVar(familyColor(fam)) || '#9a9385';
    const models = byFam.get(fam).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    models.forEach(([model], i) => {
      const fill = mix(base, PAPER, TINTS[Math.min(i, TINTS.length - 1)] + Math.max(0, i - 3) * 0.04);
      const text = contrast(fill, '#ffffff') >= contrast(fill, INK) ? '#ffffff' : INK;
      // Thin lines on paper need more contrast than a filled bar.
      const line = contrast(fill, PAPER) < 1.8 ? mix(fill, INK, 0.45) : fill;
      map.set(model, { fill, text, line, family: fam });
      order.push(model);
    });
  }
  const fallback = { fill: '#9a9385', text: INK, line: '#6f685c', family: 'other' };
  return { order, color: (m) => map.get(m) || fallback };
}

const GROUP_ORDER = ['Anthropic', 'OpenAI', 'OpenCode Go', 'OpenCode Zen', 'Other'];
const ANTHROPIC = new Set(['opus', 'fable', 'sonnet', 'haiku']);

// Provider group for the legend, from the server's provider name.
export function providerGroup(model, providers) {
  const raw = String((providers && providers[model]) || '').toLowerCase();
  if (raw) {
    if (raw === 'anthropic') return 'Anthropic';
    if (raw.includes('openai') || raw.includes('codex')) return 'OpenAI';
    if (raw.includes('opencode')) return raw.includes('zen') ? 'OpenCode Zen' : 'OpenCode Go';
    return 'Other';
  }
  const fam = familyOf(model);
  if (ANTHROPIC.has(fam)) return 'Anthropic';
  if (fam === 'gpt') return 'OpenAI';
  return 'Other';
}

export const groupRank = (g) => { const i = GROUP_ORDER.indexOf(g); return i < 0 ? GROUP_ORDER.length : i; };

// Group bar colours: one per provider group.
export const GROUP_COLOR = { Anthropic: '#1d4f91', OpenAI: INK, 'OpenCode Go': '#2e7d5b', 'OpenCode Zen': '#93c4a8', Other: '#9a9385' };
