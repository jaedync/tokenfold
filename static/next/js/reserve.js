// Reserved panel heights while a view loads.
//
// A panel that waits for its data shows one loading line (data-loading). Without
// a reserved height the page is then about one screen tall and every panel
// jumps when its data comes in. So each panel keeps the height it had the
// last time the view was loaded (per view, scope and layout width), as the
// CSS variable --reserve. The CSS applies it only while the panel holds a
// [data-loading] line; a panel with no record uses its --reserve-default.

const STORE_KEY = 'tf_panel_heights';
// Widths where a view's grid changes its columns (the @media rules in css/).
const LAYOUT_EDGES = [700, 860, 1100, 1340];
const RECORD_DELAY_MS = 1000;  // at most one record a second, after a change

// isReady(): true when the visible view has every feed it waits for.
// where(): { view, scope, container } of the view on screen, or null.
export function createReserve({ store, root, isReady, where, env = window }) {
  let heights = load();
  let timer = null;

  function load() {
    try {
      const saved = JSON.parse(env.localStorage.getItem(STORE_KEY) || '{}');
      return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
    } catch { return {}; }
  }

  const layout = () => LAYOUT_EDGES.filter((w) => env.innerWidth > w).length;
  const keyOf = (view, scope) => `${view}:${scope}:${layout()}`;

  // Give each panel of a new view its recorded height.
  function apply(view, scope, container) {
    const saved = heights[keyOf(view, scope)] || {};
    for (const el of container.querySelectorAll(':scope > .panel')) {
      const h = saved[el.dataset.panel];
      if (typeof h === 'number' && h > 0) el.style.setProperty('--reserve', `${h}px`);
      else el.style.removeProperty('--reserve');
    }
  }

  // Keep the heights of the loaded panels for the next load.
  function record() {
    timer = null;
    const at = where();
    if (!at || !isReady()) return;
    const key = keyOf(at.view, at.scope);
    const next = { ...(heights[key] || {}) };
    let changed = false;
    for (const el of at.container.querySelectorAll(':scope > .panel')) {
      if (el.hidden || el.querySelector('[data-loading]')) continue;
      const h = el.offsetHeight;
      if (!h || next[el.dataset.panel] === h) continue;
      next[el.dataset.panel] = h;
      changed = true;
    }
    if (!changed) return;
    heights = { ...heights, [key]: next };
    try { env.localStorage.setItem(STORE_KEY, JSON.stringify(heights)); } catch { /* storage off: defaults only */ }
  }

  // A throttle, not a debounce: the 5 s live poll would push a debounce back forever.
  function schedule() {
    if (timer === null) timer = env.setTimeout(record, RECORD_DELAY_MS);
  }

  store.subscribe(schedule);
  if (typeof env.ResizeObserver === 'function') new env.ResizeObserver(schedule).observe(root);

  return { apply };
}
