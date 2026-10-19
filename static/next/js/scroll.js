// Window scroll position across reloads, back/forward and view switches.
//
// The browser's own restoration runs before the feeds answer. Then every
// panel is a short "Loading" line and the page is about one screen tall, so
// the position clamps to the top and stays there when the data comes in.
// Here each history entry gets a key (history.state.tfKey), its position is
// kept under that key (sessionStorage at pagehide, so a reload keeps it), and
// it is put back once the view's data is in and the page is tall enough. Any
// scroll input from the user cancels a restore that is still waiting.

const STORE_KEY = 'tf_scroll';
const MAX_ENTRIES = 50;        // positions kept for this many history entries
const GROW_GRACE_MS = 1000;    // after the data is in, wait this long for the page to grow
const GIVE_UP_MS = 30000;      // a restore never waits longer than this
const INTENT = ['wheel', 'touchstart', 'keydown', 'mousedown'];

// isReady(): true when the visible view has every feed it waits for.
export function createScroll({ store, root, isReady, env = window }) {
  const { history, document } = env;
  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

  const positions = new Map(load());
  let current = null;   // the key of the entry on screen
  let pending = null;   // { y, readyAt, giveUp, grow }

  const maxY = () => document.documentElement.scrollHeight - env.innerHeight;
  const newKey = () => Math.random().toString(36).slice(2, 10);

  function load() {
    try {
      const saved = JSON.parse(env.sessionStorage.getItem(STORE_KEY) || '[]');
      return Array.isArray(saved) ? saved.filter((e) => Array.isArray(e) && typeof e[1] === 'number') : [];
    } catch { return []; }
  }

  function persist() {
    const entries = [...positions].slice(-MAX_ENTRIES);
    try { env.sessionStorage.setItem(STORE_KEY, JSON.stringify(entries)); } catch { /* storage off: no restore after a reload */ }
  }

  // The key of the entry on screen; an entry without one gets one.
  function entryKey() {
    const state = history.state && typeof history.state === 'object' ? history.state : {};
    if (state.tfKey) return state.tfKey;
    const key = newKey();
    history.replaceState({ ...state, tfKey: key }, '');
    return key;
  }

  // Keep the position of the entry on screen. A restore that waits must not
  // write the top of a short page over the saved position.
  function remember() {
    if (pending || current === null) return;
    positions.delete(current);
    positions.set(current, Math.round(env.scrollY));
  }

  function cancel() {
    if (!pending) return;
    clearTimeout(pending.giveUp);
    clearTimeout(pending.grow);
    pending = null;
  }

  function finish() {
    const { y } = pending;
    cancel();
    env.scrollTo(0, Math.min(y, Math.max(0, maxY())));
    remember();
  }

  // Scroll once the data is in and the page is tall enough, or clamp to the
  // bottom when the page stays short.
  function attempt() {
    if (!pending || !isReady()) return;
    const now = env.performance.now();
    if (pending.readyAt === null) pending.readyAt = now;
    if (maxY() < pending.y && now - pending.readyAt < GROW_GRACE_MS) {
      if (!pending.grow) pending.grow = env.setTimeout(() => { if (pending) { pending.grow = null; schedule(); } }, GROW_GRACE_MS);
      return;
    }
    finish();
  }

  let frame = null;
  function schedule() {
    if (!pending || frame !== null) return;
    // After the panels updated and the browser laid them out.
    frame = env.requestAnimationFrame(() => { frame = null; attempt(); });
  }

  // Before the history entry changes (a view tab): keep the old position
  // and give the new entry its key.
  function leave() {
    remember();
    return { tfKey: newKey() };
  }

  // The entry on screen changed (boot, back, forward, a view tab): put back
  // its position. An entry with no position opens at the top.
  function restore() {
    cancel();
    current = entryKey();
    const y = positions.get(current) || 0;
    if (y <= 0) { env.scrollTo(0, 0); return; }
    pending = { y, readyAt: null, grow: null, giveUp: env.setTimeout(() => { if (pending) finish(); }, GIVE_UP_MS) };
    schedule();
  }

  env.addEventListener('scroll', remember, { passive: true });
  env.addEventListener('pagehide', () => { remember(); persist(); });
  // Mobile browsers can drop a tab in the background without a pagehide.
  document.addEventListener('visibilitychange', () => { if (document.hidden) { remember(); persist(); } });
  for (const type of INTENT) env.addEventListener(type, cancel, { passive: true, capture: true });
  store.subscribe(schedule);
  if (typeof env.ResizeObserver === 'function') new env.ResizeObserver(schedule).observe(root);

  return { leave, remember, restore, cancel };
}
