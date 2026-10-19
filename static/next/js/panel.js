// Panel plumbing shared by every view. See static/next/README.md for the API.
// A panel module: export default { id, label, cls, mount(el, ctx) } where
// mount returns { update(state), destroy() }. One failing panel shows a quiet
// line and never takes the page down with it.

import { h, fill } from './dom.js';
import * as f from './format.js';

// Mount `panel` into a new <section> appended to `parent`.
export function mountPanel(panel, parent, ctx) {
  const el = h('section', { class: `panel p-${panel.id} ${panel.cls || ''}`.trim(), 'data-panel': panel.id, 'aria-label': panel.label });
  parent.append(el);
  return mountInto(panel, el, ctx);
}

// Mount `panel` into an existing element (the shell's live bar uses this).
export function mountInto(panel, el, ctx) {
  let inst = null;
  let broken = false;
  const fail = (phase, err) => {
    console.error(`tokenfold: panel ${panel.id} failed to ${phase}`, err);
    if (broken) return;
    broken = true;
    fill(el, stateLine(`${panel.label} failed to load.`, 'Reload the page. If it fails again, see the browser console.'));
  };
  try {
    inst = panel.mount(el, ctx) || {};
  } catch (err) {
    fail('mount', err);
  }
  return {
    el,
    update(state) {
      if (broken || !inst || !inst.update) return;
      try {
        inst.update(state);
      } catch (err) {
        fail('update', err);
      }
    },
    destroy() {
      try {
        if (inst && inst.destroy) inst.destroy();
      } catch (err) {
        console.error(`tokenfold: panel ${panel.id} failed to destroy`, err);
      }
      el.remove();
    },
  };
}

// One quiet line for a panel with no data: loading, unavailable or empty.
// data-loading lets the panel keep its reserved height (reserve.js) while it loads.
export function stateLine(text, sub, { loading = false } = {}) {
  return h('div', { class: 'p-state', role: 'status', 'data-loading': loading }, text, sub ? h('small', null, sub) : null);
}

export const loadingLine = (text, sub) => stateLine(text, sub, { loading: true });

// What a panel should show for a feed: 'ok' (fresh data), 'stale' (old data
// after a failed refresh), 'loading' or 'unavailable'.
export function feedStatus(feed) {
  if (!feed) return 'loading';
  if (feed.status === 'ok') return 'ok';
  if (feed.data) return feed.status === 'error' ? 'stale' : 'ok';
  return feed.status === 'error' ? 'unavailable' : 'loading';
}

const UNAVAILABLE_SUB = {
  404: 'This server does not have this feed yet.',
  401: 'Sign in again to see this data.',
  403: 'This scope is locked on this server.',
};

// A state line for a feed with no data. Returns null when there is data.
export function feedLine(feed, what) {
  const status = feedStatus(feed);
  if (status === 'ok' || status === 'stale') return null;
  if (status === 'loading') return loadingLine(`Loading ${what}…`);
  const code = feed.error && feed.error.status;
  return stateLine(`${what[0].toUpperCase()}${what.slice(1)} is unavailable.`,
    UNAVAILABLE_SUB[code] || 'The page tries again on its own.');
}

// "Data is 4 min old" for a feed that failed after it had data.
export function staleNote(feed, now) {
  if (feedStatus(feed) !== 'stale' || !feed.at) return null;
  return h('span', { class: 'age' }, `Not updated · ${f.age(now - feed.at)}`);
}
