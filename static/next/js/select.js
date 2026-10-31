import { PROVIDER_ORDER } from './palette.js';

// Pure selectors over the feed payloads. They pick and sort server values;
// they never forecast (the one forecast is the server's pace block).

export const dataOf = (feed) => (feed && feed.data) || null;

// id of the hero window: the owner's pin (pin.js) when the runway has that
// long window, else the server's binding window. The work hero is always the
// billing meter: the pin is a personal-scope choice.
export function heroId(runway, pin = null) {
  if (!runway || !Array.isArray(runway.windows)) return null;
  const pinned = pin && runway.scope !== 'enterprise' ? runway.windows.find((w) => w.id === pin) : null;
  if (pinned && pinned.kind !== 'short') return pin;
  return runway.binding ?? null;
}

export const isPinned = (runway, pin) => Boolean(pin) && heroId(runway, pin) === pin;

export function bindingWindow(runway, pin = null) {
  const id = heroId(runway, pin);
  return (id && runway.windows.find((w) => w.id === id)) || null;
}

export function claudeShortWindow(runway) {
  if (!runway || !Array.isArray(runway.windows)) return null;
  return runway.windows.find((w) => w.provider === 'claude' && w.kind === 'short') || null;
}

const KIND_RANK = { short: 0, long: 1, month: 2 };
const rankOf = (order, key) => { const i = order.indexOf(key); return i < 0 ? order.length : i; };

// Every window in a fixed order that does not move between polls: provider
// (PROVIDER_ORDER, then by name), then short before long before month, then
// window length, then label. Forecasts change each poll, so they never sort.
export function allWindows(runway) {
  if (!runway || !Array.isArray(runway.windows)) return [];
  const str = (v) => String(v ?? '');
  return runway.windows.slice().sort((a, b) =>
    rankOf(PROVIDER_ORDER, a.provider) - rankOf(PROVIDER_ORDER, b.provider)
    || str(a.provider).localeCompare(str(b.provider))
    || (KIND_RANK[a.kind] ?? 3) - (KIND_RANK[b.kind] ?? 3)
    || (a.window_s ?? 0) - (b.window_s ?? 0)
    || str(a.label).localeCompare(str(b.label))
    || str(a.id).localeCompare(str(b.id)));
}

// allWindows split into runs of one provider, for the provider headings.
export function windowsByProvider(runway) {
  const groups = [];
  for (const w of allWindows(runway)) {
    const last = groups[groups.length - 1];
    if (last && last.provider === w.provider) last.windows.push(w);
    else groups.push({ provider: w.provider, windows: [w] });
  }
  return groups;
}

// One window per other provider, the fullest, for "room elsewhere".
export function roomElsewhere(runway) {
  if (!runway || !Array.isArray(runway.windows)) return [];
  const best = new Map();
  for (const w of runway.windows) {
    if (w.provider === 'claude' || typeof w.pct !== 'number') continue;
    const cur = best.get(w.provider);
    if (!cur || w.pct > cur.pct) best.set(w.provider, w);
  }
  return [...best.values()];
}

// Server "now" in epoch seconds: browser time corrected by the feed's skew.
export const nowFrom = (state) => Date.now() / 1000 + (state.skew || 0);

const STATE_RANK = { waiting: 0, working: 1, idle: 2 };

// Sessions in the server's order, made explicit: waiting (longest first), working, idle.
export function sortSessions(list) {
  return (list || []).slice().sort((a, b) =>
    (STATE_RANK[a.state] ?? 3) - (STATE_RANK[b.state] ?? 3) || (a.state_since ?? 0) - (b.state_since ?? 0));
}
