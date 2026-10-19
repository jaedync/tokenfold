// Limit windows (/api/window-history): one report card per ended Claude
// window, newest first, and a strip of the peaks against the 100% line.
// Personal scope only; the work scope has a monthly budget, not windows.
import { h, patch, setText, approx, panelHead, segmented } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { feedLine, feedStatus, staleNote, stateLine, loadingLine } from '../panel.js';
import { median, changed } from './history-common.js';

const BUCKETS = [['seven_day', '7-day'], ['five_hour', '5-hour']];
const FEED = { seven_day: 'windowHistory', five_hour: 'windowHistory5h' };
const FIRST_CARDS = 12;

function rangeText(w, bucket) {
  if (bucket === 'five_hour') return `${f.dow(w.start)} ${f.dateShort(w.start)} · ${f.clock(w.start)} to ${f.clock(w.end)}`;
  return `${f.dateShort(w.start)} to ${f.dateShort(w.end)}`;
}

function card(w, bucket) {
  const grants = w.granted_resets || [];
  const hit = Boolean(w.hit_limit);
  return h('li', { class: 'wc', 'data-key': String(w.start), 'data-hit': hit ? '1' : null },
    h('div', { class: 'wc-range', title: `${f.dayClock(w.start)} to ${f.dayClock(w.end)}` }, rangeText(w, bucket)),
    h('div', { class: 'wc-peak' },
      h('span', { class: 'wc-v' }, f.pct(w.peak_pct)),
      h('span', { class: hit ? 'chip chip-red' : 'chip chip-line' }, hit ? 'Hit the limit' : 'Under the limit')),
    h('div', { class: 'wc-bar', 'aria-hidden': 'true' }, h('i', { style: { width: `${Math.min(100, w.peak_pct || 0)}%` } })),
    h('dl', { class: 'wc-facts' },
      h('div', null, h('dt', null, 'Spend'), h('dd', null, f.missing(w.usd) ? f.DASH : [approx(), f.usd(w.usd, { cents: false })])),
      h('div', null, h('dt', null, 'Per point'), h('dd', null, f.missing(w.usd_per_point) ? f.DASH : [approx(), f.usd(w.usd_per_point, { cents: true })])),
      h('div', { class: 'wc-grants' }, h('dt', null, 'Granted resets'), h('dd', null, grants.length
        ? grants.map((g, i) => [i ? ', ' : '', `${f.dateShort(g)} ${f.clock(g)}`]) : 'none'))));
}

// Peaks oldest -> newest, one column per window. Red: the window hit the limit.
function strip(windows, bucket) {
  const list = windows.slice().sort((a, b) => a.start - b.start);
  const days = new Set();
  let labels = 0;
  const cols = list.map((w) => {
    const day = f.dateShort(w.start);
    const label = bucket === 'seven_day' ? day : (days.has(day) ? null : (days.add(day), day));
    // Every other label hides on a narrow panel (history.css), so dates never overlap.
    const odd = label && labels++ % 2 === 1;
    return h('div', { class: 'ws-col', 'data-key': String(w.start), title: `${rangeText(w, bucket)} · peak ${f.pct(w.peak_pct)}${w.hit_limit ? ' · hit the limit' : ''}` },
      h('span', { class: 'hw-bar', 'data-hit': w.hit_limit ? '1' : null, style: { height: `${Math.min(100, w.peak_pct || 0)}%` } },
        (w.granted_resets || []).length ? h('b', { class: 'ws-grant' }) : null),
      label ? h('span', { class: 'ws-x', 'data-odd': odd ? '1' : null }, label) : null);
  });
  const hits = list.filter((w) => w.hit_limit).length;
  return h('div', { class: 'ws' },
    h('div', { class: 'ws-plot', role: 'img', 'aria-label': `Peak of ${list.length} windows, oldest first. ${hits} hit the limit.` },
      h('span', { class: 'ws-limit' }, h('em', null, '100%')), cols),
    h('div', { class: 'ws-keys' },
      h('span', null, h('i', { style: { background: 'var(--ink)' } }), 'peak of the window'),
      h('span', null, h('i', { style: { background: 'var(--red)' } }), 'hit the limit'),
      h('span', null, h('i', { style: { background: 'var(--blue)', width: '8px', height: '8px' } }), 'granted reset'),
      h('span', null, h('i', { class: 'k-limit' }), '100% of the limit')));
}

// The first read after a restart can be slow (one week of events per window):
// say so while it loads, and after a timeout say that it is still building.
function windowsLine(feed) {
  const status = feedStatus(feed);
  if (status === 'loading') return loadingLine('Loading the window history…', 'The first read after a restart can take up to a minute.');
  if (status === 'unavailable' && feed.error && feed.error.status === 0) {
    return stateLine('The server is still building the window history.', 'The page tries again on its own.');
  }
  return feedLine(feed, 'the window history');
}

function summary(windows) {
  const hits = windows.filter((w) => w.hit_limit).length;
  const grants = windows.reduce((a, w) => a + (w.granted_resets || []).length, 0);
  const fact = (k, v) => h('div', null, h('dt', null, k), h('dd', null, v));
  return h('dl', { class: 'wsum' },
    fact('Windows', f.int(windows.length)),
    fact('Hit the limit', f.int(hits)),
    fact('Median peak', f.pct(median(windows.map((w) => w.peak_pct)))),
    fact('Median spend', windows.length ? [approx(), f.usd(median(windows.map((w) => w.usd)), { cents: false })] : f.DASH),
    fact('Granted resets', f.int(grants)));
}

export default {
  id: 'history-windows',
  label: 'Limit windows',
  cls: 'p-wide',
  mount(el, ctx) {
    const head = panelHead('Limit windows');
    el.setAttribute('aria-labelledby', head.id);
    const seg = segmented('Window length', BUCKETS, ctx.store.get().historyBucket || 'seven_day',
      (v) => ctx.store.set('historyBucket', v));
    head.el.append(seg.el);
    const lead = h('p', { class: 'panel-sub hl-lead' });
    const state = h('div');
    const body = h('div', { class: 'wbody' });
    const more = h('button', { type: 'button', class: 'btn wc-more' });
    let showAll = false;
    more.addEventListener('click', () => { showAll = !showAll; seen = null; update(ctx.store.get()); });
    el.append(head.el, lead, state, body);

    let seen = null;
    function update(s) {
      const bucket = s.historyBucket || 'seven_day';
      const feed = s[FEED[bucket]];
      const key = [feed, s.scope, bucket, showAll];
      if (!changed(seen, key)) return;
      seen = key;
      seg.select(bucket);
      // Work usage has no limit windows (the billing meter panel is its limit).
      el.hidden = s.scope === 'enterprise';
      if (el.hidden) return;
      setText(lead, `Each card is one ended Claude ${bucket === 'five_hour' ? '5-hour' : '7-day'} window: its peak, whether it hit the limit, and its spend. Spend counts the Claude requests inside the window; per point divides it by the points used.`);
      const line = windowsLine(feed);
      patch(state, line, staleNote(feed, nowFrom(s)));
      if (line) { patch(body); return; }
      const windows = (dataOf(feed).windows || []).filter((w) => w && typeof w.start === 'number');
      if (!windows.length) {
        patch(body, stateLine('No window has ended yet.', 'Cards appear after the first window resets.'));
        return;
      }
      const newest = windows.slice().sort((a, b) => b.start - a.start);
      const shown = showAll ? newest : newest.slice(0, FIRST_CARDS);
      setText(more, showAll ? 'Show fewer' : `Show all ${newest.length} windows`);
      more.hidden = newest.length <= FIRST_CARDS;
      patch(body,
        h('div', { class: 'wtop' }, summary(windows), strip(windows, bucket)),
        h('ol', { class: 'wcards', 'aria-label': 'Ended windows, newest first' }, shown.map((w) => card(w, bucket))),
        more);
    }
    return { update };
  },
};
