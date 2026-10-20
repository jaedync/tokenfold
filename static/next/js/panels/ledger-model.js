// Ledger data model: pure functions over /api/ledger payloads. No DOM.
// The server clips each session to [from, to) and prices it; the browser
// only groups, filters and sums those numbers (pivots, filters, totals).
// It never makes a forecast.

import { dataOf } from '../select.js';

const D = 86400;
export const QUIET_MAX_USD = 0.5;
export const QUIET_MAX_WALL_S = 600;
export const UNKNOWN_MODEL = 'Unknown model';

export const DEFAULT_UI = Object.freeze({ pkey: '7d', from: null, to: null, filters: Object.freeze([]), showQuiet: false });
export const uiOf = (state) => state.ledgerUi || DEFAULT_UI;

// ---------------------------------------------------------------- time
// Local midnight `n` days after the midnight that holds `e` (DST safe).
export function dayStart(e, n = 0) {
  const d = new Date(e * 1000);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() + n);
  return d.getTime() / 1000;
}

// Local midnights from the day of `from` to the day of `to` (exclusive end).
export function daysBetween(from, to) {
  const out = [];
  for (let t = dayStart(from); t < to; t = dayStart(t, 1)) out.push(t);
  return out;
}

// "Now" for the ledger: the server time of the 30-day payload, else the page clock.
export function ledgerNow(state, fallback) {
  const l = dataOf(state.ledger);
  return l && typeof l.to === 'number' ? l.to : fallback;
}

// {from, to, open}: open = runs up to now (no `to` sent to the server).
export function periodRange(ui, now) {
  const today = dayStart(now);
  if (ui.pkey === 'today') return { from: today, to: now, open: true };
  if (ui.pkey === '30d') return { from: dayStart(now, -29), to: now, open: true };
  if (ui.pkey === 'custom' && typeof ui.from === 'number') {
    const to = Math.min(typeof ui.to === 'number' ? ui.to : now, now);
    return { from: ui.from, to, open: to >= now - 1 };
  }
  return { from: dayStart(now, -6), to: now, open: true };
}

// Query for the period feed, or null when the 30-day feed already covers it.
export function periodParams(state, mock) {
  const ui = uiOf(state);
  if (mock || ui.pkey === '30d' || !dataOf(state.ledger)) return null;
  const r = periodRange(ui, ledgerNow(state, Date.now() / 1000));
  return r.open ? { from: r.from } : { from: r.from, to: r.to };
}

// The payload answers this range (the server echoes from/to).
function answers(data, r) {
  if (!data || typeof data.from !== 'number') return false;
  return Math.abs(data.from - r.from) < 1 && (r.open || Math.abs(data.to - r.to) < 1);
}

// ---------------------------------------------------------------- sessions
// Pi tools are lower case, Claude Code tools capitalised: one row per tool.
export function toolName(name) {
  const n = String(name || '');
  return /^[a-z][a-z0-9_]*$/.test(n) ? n[0].toUpperCase() + n.slice(1) : n;
}

export const isQuiet = (s) => !s.live && s.runs.length === 0 && s.prompts <= 1
  && s.cost < QUIET_MAX_USD && (s.end - s.start) < QUIET_MAX_WALL_S;

function mainParts(raw, model, orch) {
  const entries = Object.entries(raw.orch_models || {}).filter(([, v]) => typeof v === 'number');
  if (!entries.length && (orch > 0 || !(raw.runs || []).length)) entries.push([model, orch]);
  return entries.sort((a, b) => b[1] - a[1]).map(([m, cost], i) => ({
    kind: 'main', model: m || UNKNOWN_MODEL, cost,
    // Active time and prompts belong to the orchestrator: give them to its main model.
    active: i === 0 ? raw.active_s || 0 : 0, prompts: i === 0 ? raw.prompts || 0 : 0,
    start: raw.start,
  }));
}

function normSession(raw) {
  const model = raw.model || UNKNOWN_MODEL;
  const runs = (raw.runs || []).map((r) => ({ ...r, model: r.model || UNKNOWN_MODEL, cost: r.cost_usd || 0,
    active: typeof r.active_s === 'number' ? r.active_s : Math.max(0, (r.end || 0) - (r.start || 0)) }));
  const parts = [...mainParts(raw, model, raw.orch_cost_usd || 0),
    ...runs.map((r) => ({ kind: 'run', model: r.model, cost: r.cost, active: r.active, prompts: 0, start: r.start, run: r }))];
  const tools = {};
  for (const [k, v] of Object.entries(raw.tool_calls || {})) tools[toolName(k)] = (tools[toolName(k)] || 0) + v;
  const s = {
    id: raw.id, machine: raw.machine || 'unknown', project: raw.project || '—', title: raw.title || null, label: raw.label || null,
    client: raw.client, start: raw.start, end: Math.max(raw.end || raw.start, raw.start), live: Boolean(raw.live),
    prompts: raw.prompts || 0, active: raw.active_s || 0, cost: raw.cost_usd || 0,
    orch: raw.orch_cost_usd || 0, agent: raw.agent_cost_usd || 0, model, runs, parts, tools,
    tokens: raw.tokens || null, hasTools: Boolean(raw.tool_calls),
  };
  return { ...s, quiet: isQuiet(s) };
}

const normCache = new WeakMap();
// Payload -> {sessions, byId}; memoised per payload object.
export function normalize(data) {
  if (!data || !Array.isArray(data.sessions)) return { sessions: [], byId: new Map() };
  let out = normCache.get(data);
  if (!out) {
    const sessions = data.sessions.filter((s) => s && s.id && typeof s.start === 'number').map(normSession);
    out = { sessions, byId: new Map(sessions.map((s) => [s.id, s])) };
    normCache.set(data, out);
  }
  return out;
}

export const clientName = (c) => ({ 'claude-code': 'Claude Code', 'pi-agent': 'Pi' }[c] || c || '');
// A session started in the home folder reports the project "~", which names
// nothing; such a session (and one with no project) takes its client's name.
export const HOME_PROJECT = '~';
export const projectLabel = (p) => (p === HOME_PROJECT ? '~ (home folder)' : p);
// The server's label names an untitled session by its facts (its subagents,
// else its project and start); an older server sends none.
export const sessionName = (s) => s.title || s.label || (s.project && s.project !== HOME_PROJECT ? s.project : `Untitled ${clientName(s.client) || 'agent'} session`);
// True when the name already says the project, so a meta line can leave it out.
export const nameShowsProject = (s) => {
  const name = sessionName(s);
  return name === s.project || name.startsWith(`${s.project} · `) || (s.project === HOME_PROJECT && name.startsWith('Home folder · '));
};

// ---------------------------------------------------------------- filters
// Same dimension = OR, across dimensions = AND. skip: a pivot's own dimension.
export function filterSets(filters, skip) {
  const by = {};
  for (const f of filters) {
    if (f.dim === skip) continue;
    (by[f.dim] = by[f.dim] || new Set()).add(f.val);
  }
  return by;
}

export function matchParts(s, by) {
  if (by.project && !by.project.has(s.project)) return [];
  if (by.machine && !by.machine.has(s.machine)) return [];
  if (by.tool && ![...by.tool].some((t) => s.tools[t] > 0)) return [];
  return by.model ? s.parts.filter((p) => by.model.has(p.model)) : s.parts;
}

// [{s, parts, cost}] for sessions with at least one matching part.
export function matched(sessions, filters, skip) {
  const by = filterSets(filters, skip);
  const out = [];
  for (const s of sessions) {
    const parts = matchParts(s, by);
    if (parts.length) out.push({ s, parts, cost: parts.reduce((a, p) => a + p.cost, 0) });
  }
  return out;
}

export const hasFilter = (filters, dim, val) => filters.some((f) => f.dim === dim && f.val === val);

// ---------------------------------------------------------------- derive
let memo = { key: null, out: null };

// Everything the Ledger panels share, recomputed only when a payload or the
// UI state changes (the 5 s live poll does not touch it).
export function derive(state, pageNow) {
  const l30 = dataOf(state.ledger);
  const lp = dataOf(state.ledgerPeriod);
  const ui = uiOf(state);
  const now = ledgerNow(state, pageNow);
  const key = [l30, lp, ui, state.mock, l30 ? 0 : Math.floor(now / 60)];
  if (memo.key && key.every((k, i) => k === memo.key[i])) return memo.out;
  const range = periodRange(ui, now);
  const exact = ui.pkey === '30d' || state.mock ? l30 : (answers(lp, range) ? lp : null);
  // While the period loads, show the 30-day sessions that touch it (unclipped) and say so.
  const source = exact || l30;
  const sessions = normalize(source).sessions.filter((s) => s.end >= range.from && s.start < range.to);
  const quietCount = sessions.reduce((n, s) => n + (s.quiet ? 1 : 0), 0);
  const listed = ui.showQuiet ? sessions : sessions.filter((s) => !s.quiet);
  const out = {
    now, range, ui, exact: Boolean(exact), sessions, listed, quietCount,
    sessions30: normalize(l30).sessions, daily30: (l30 && l30.daily) || null,
    providers: { ...((l30 && l30.model_providers) || {}), ...((source && source.model_providers) || {}) },
    hasData: Boolean(source && Array.isArray(source.sessions)),
    // Work only: days the billing meter rose more than the recorded spend.
    meterGaps: (source && source.meter_gaps) || [],
  };
  memo = { key, out };
  return out;
}

// Peak number of runs at the same time.
export function peakParallel(runs) {
  const ev = [];
  for (const r of runs) { ev.push([r.start, 1]); ev.push([Math.max(r.end, r.start), -1]); }
  ev.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let c = 0, m = 0;
  for (const [, d] of ev) { c += d; m = Math.max(m, c); }
  return m;
}

export const DAY_S = D;
