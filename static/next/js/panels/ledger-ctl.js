// Ledger control bar (sticky under the live bar): the period, the filter
// chips every panel obeys, and the quiet-session toggle.
import { h, fill, setText, segmented } from '../dom.js';
import * as f from '../format.js';
import { derive, dayStart, projectLabel } from './ledger-model.js';
import { setPeriod, setUi, toggleFilter, clearFilters, colorsFor, swatch, plural } from './ledger-kit.js';
import { periodLabel } from './ledger-stats.js';

const DIM = { project: 'Project', model: 'Model', machine: 'Machine', tool: 'Tool' };
const dateFmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });

export default {
  id: 'ledger-ctl',
  label: 'Period and filters',
  cls: 'p-wide',
  mount(el, ctx) {
    const { store } = ctx;
    el.setAttribute('role', 'region');
    const seg = segmented('Period', [['today', 'Today'], ['7d', '7D'], ['30d', '30D'], ['custom', 'Custom']], '7d', (v) => {
      if (v !== 'custom') { setPeriod(store, v); return; }
      const d = derive(store.get(), ctx.now());
      if (d.ui.pkey !== 'custom') setPeriod(store, 'custom', dayStart(d.now, -2), null);
    });
    const customBtn = seg.el.querySelector('[data-value="custom"]');
    const chips = h('div', { class: 'lc-filters', 'aria-live': 'polite' });
    const quiet = h('button', { type: 'button', class: 'lc-quiet', hidden: true,
      onclick: () => setUi(store, { showQuiet: !derive(store.get(), ctx.now()).ui.showQuiet }) });
    const range = h('span', { class: 'lc-range' });
    el.append(
      h('div', { class: 'lc-period' }, h('span', { class: 'lc-label' }, 'Period'), seg.el),
      chips, h('div', { class: 'lc-end' }, quiet, range));

    // Sit right under the shell's sticky live bar, whatever its height.
    const bar = document.getElementById('livebar');
    const place = () => { el.style.top = `${bar ? bar.offsetHeight : 0}px`; };
    place();
    const ro = typeof ResizeObserver === 'function' && bar ? new ResizeObserver(place) : null;
    if (ro) ro.observe(bar);

    function chip(x, d) {
      const name = x.dim === 'tool' && x.val === 'Task' ? 'Subagent (Task)' : x.dim === 'project' ? projectLabel(x.val) : x.val;
      return h('span', { class: 'lc-chip' },
        h('span', { class: 'lc-dim' }, DIM[x.dim] || x.dim),
        x.dim === 'model' ? swatch(colorsFor(d).color(x.val).fill) : null,
        h('span', { class: 'lc-val' }, name),
        h('button', { type: 'button', 'aria-label': `Remove ${DIM[x.dim] || x.dim} filter ${x.val}`, onclick: () => toggleFilter(store, x.dim, x.val) }, '×'));
    }

    let seen = null;
    return {
      update(s) {
        const d = derive(s, ctx.now());
        if (d === seen) return;
        seen = d;
        seg.select(d.ui.pkey);
        setText(customBtn, d.ui.pkey === 'custom' ? periodLabel(d) : 'Custom');
        const { filters } = d.ui;
        fill(chips, filters.length
          ? [...filters.map((x) => chip(x, d)), filters.length > 1 ? h('button', { type: 'button', class: 'lc-clear', onclick: () => clearFilters(store) }, 'Clear all') : null]
          : h('span', { class: 'lc-hint' }, 'No filters. Click a model, a row or a legend item to narrow every panel.'));
        quiet.hidden = !d.quietCount;
        setText(quiet, d.ui.showQuiet ? `Hide ${plural(d.quietCount, 'quiet session')}` : `${plural(d.quietCount, 'quiet session')} hidden · Show`);
        quiet.setAttribute('aria-pressed', String(d.ui.showQuiet));
        quiet.title = 'Quiet: no subagents, one prompt or less, under $0.50 and under 10 minutes (for example a relay\'s own calls). Their dollars always count.';
        const end = d.range.open ? `now, ${f.clock(d.now)}` : dateFmt.format(new Date((d.range.to - 1) * 1000));
        setText(range, `${dateFmt.format(new Date(d.range.from * 1000))} – ${end}`);
      },
      destroy() { if (ro) ro.disconnect(); },
    };
  },
};
