// Pure selectors over the feed payloads. They pick and sort server values;
// they never forecast (the one forecast is the server's pace block).

export const dataOf = (feed) => (feed && feed.data) || null;

export function bindingWindow(runway) {
  if (!runway || !Array.isArray(runway.windows)) return null;
  return runway.windows.find((w) => w.id === runway.binding) || null;
}

export function claudeShortWindow(runway) {
  if (!runway || !Array.isArray(runway.windows)) return null;
  return runway.windows.find((w) => w.provider === 'claude' && w.kind === 'short') || null;
}

const projected = (w) => (w.pace && typeof w.pace.projected_pct === 'number' ? w.pace.projected_pct : -1);

// Every window but the binding one, tightest forecast first (unknown last).
export function otherWindows(runway) {
  if (!runway || !Array.isArray(runway.windows)) return [];
  return runway.windows.filter((w) => w.id !== runway.binding)
    .slice()
    .sort((a, b) => projected(b) - projected(a) || (b.pct ?? -1) - (a.pct ?? -1));
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
