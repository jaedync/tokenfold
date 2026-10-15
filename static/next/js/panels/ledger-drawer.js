// Session drawer: what one session bought. It opens at once from the ledger
// row, then loads /api/sessions/{id} (the whole session, prompts, tools,
// tokens, served model). Esc and the scrim close it; Tab stays inside;
// prev/next walk the period's sessions in time order.
import { h, patch, setText } from '../dom.js';
import * as f from '../format.js';
import { derive, matched, normalize, sessionName, clientName, projectLabel, peakParallel } from './ledger-model.js';
import { colorsFor, swatch, money, pct0, tokens, plural, closeSession, openSession } from './ledger-kit.js';
import { renderGantt, runTree } from './ledger-fanout.js';

const SERVED = {
  as_asked: ['As asked', 'Every signed thinking block names the model that was asked for.'],
  partly_hidden: ['Partly hidden', 'Some blocks use a signature format that names no model. The rest came from the model that was asked for.'],
  rerouted: ['Rerouted', 'Some blocks came from a model other than the one that was asked for.'],
  unknown: ['Not checked', 'This session has no signed thinking blocks. Only Anthropic thinking blocks name the model that served them.'],
};
const FOCUSABLE = 'button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])';

async function loadDetail(ctx, id, signal) {
  if (!ctx.mock) return ctx.api.getJSON('session', { id }, { scope: ctx.store.get().scope, signal });
  const res = await fetch(new URL(`../../fixtures/ledger-sessions/${encodeURIComponent(id)}.json`, import.meta.url), { signal });
  if (!res.ok) throw Object.assign(new Error('no fixture'), { status: res.status });
  return res.json();
}

function sec(title, note, ...body) {
  return h('section', { class: 'dr-sec' }, h('h3', null, title, note ? h('span', null, note) : null), ...body);
}

function costSection(s, row, live) {
  const subShare = s.cost ? s.agent / s.cost : 0;
  const wall = Math.max(0, s.end - s.start);
  const calls = Object.values(s.tools).reduce((a, b) => a + b, 0);
  const clipped = row && Math.abs(row.cost - s.cost) >= 0.01;
  return sec('What it cost', 'API list price, not billed',
    h('div', { class: 'dr-cost' }, h('span', { class: 'dr-cost-v' }, h('span', { class: 'approx' }, '≈'), money(s.cost)),
      h('span', { class: 'dr-cost-note' }, `${live ? 'so far · ' : ''}${s.prompts ? `${money(s.cost / s.prompts)} a prompt · ` : ''}${money(s.cost / Math.max(wall / 3600, 0.25))} an hour`)),
    clipped ? h('p', { class: 'dr-clip' }, `This is the whole session. ≈${money(row.cost)} of it falls in the period.`) : null,
    h('div', { class: 'ws-bar dr-split', role: 'img', 'aria-label': `Orchestrator ${money(s.orch)}, subagents ${money(s.agent)}` },
      h('i', { class: 'sb-orch', style: { width: `${(1 - subShare) * 100}%` } }), s.runs.length ? h('i', { class: 'sb-sub', style: { width: `${subShare * 100}%` } }) : null),
    h('div', { class: 'ws-legend' }, h('span', null, 'Orchestrator ', h('b', null, money(s.orch))),
      h('span', null, `${plural(s.runs.length, 'subagent')} `, h('b', null, money(s.agent)), ` · ${pct0(subShare)}`)),
    h('div', { class: 'dr-bought' },
      h('div', null, h('b', null, f.int(s.prompts)), h('span', null, 'prompts')),
      h('div', null, h('b', null, s.hasTools ? f.int(calls) : f.DASH), h('span', null, 'tool calls')),
      h('div', null, h('b', null, s.fileEdits === undefined ? f.DASH : f.int(s.fileEdits)), h('span', null, 'file edits')),
      h('div', null, h('b', null, s.tokens ? tokens(s.tokens.output) : f.DASH), h('span', null, 'tokens written'))));
}

function modelsSection(s, detail, c) {
  const list = detail && Array.isArray(detail.models) ? detail.models.map((m) => ({ model: m.model || 'Unknown model', usd: m.cost_usd || 0 }))
    : Object.entries(s.parts.reduce((o, p) => ({ ...o, [p.model]: (o[p.model] || 0) + p.cost }), {})).map(([model, usd]) => ({ model, usd }));
  list.sort((a, b) => b.usd - a.usd);
  const tot = list.reduce((a, m) => a + m.usd, 0);
  return sec('Models', 'by cost',
    h('div', { class: 'mix-bar' }, list.map((m) => h('i', { style: { width: `${tot ? (m.usd / tot) * 100 : 100 / list.length}%`, background: c.color(m.model).fill }, title: m.model }))),
    h('div', { class: 'mix-list' }, list.map((m) => h('div', null, swatch(c.color(m.model).fill), h('span', { class: 'ml-n' }, m.model), h('b', null, money(m.usd))))));
}

function toolsTokens(s, detail, state) {
  const tools = detail && Array.isArray(detail.tools)
    ? detail.tools.map((t) => [t.name, t.calls]) : Object.entries(s.tools);
  tools.sort((a, b) => b[1] - a[1]);
  const top = tools.slice(0, 8), max = top.length ? top[0][1] || 1 : 1;
  const t = s.tokens;
  const inTok = t ? (t.input || 0) + (t.cache_write || 0) + (t.cache_read || 0) : 0;
  const tok = (k, v) => h('div', { class: 'tok-row' }, h('span', null, k), h('b', null, v));
  return h('section', { class: 'dr-sec dr-two' },
    h('div', null, h('h3', null, 'Top tools'), top.length ? top.map(([name, n]) => h('div', { class: 'tool-row' },
      h('span', { class: 'tn', title: name }, name === 'Task' ? 'Subagent' : name), h('span', { class: 'pbar' }, h('i', { style: { width: `${(n / max) * 100}%` } })), h('span', { class: 'tv' }, f.int(n))))
      : h('p', { class: 'dr-quiet' }, state === 'loading' ? 'Loading…' : 'No tool calls.')),
    h('div', null, h('h3', null, 'Tokens'), t ? [tok('Input', tokens(t.input)), tok('Output', tokens(t.output)), tok('Cache write', tokens(t.cache_write)),
      tok('Cache read', tokens(t.cache_read)), tok('Input from cache', inTok ? pct0((t.cache_read || 0) / inTok) : f.DASH)]
      : h('p', { class: 'dr-quiet' }, state === 'loading' ? 'Loading…' : 'No token counts.')));
}

function servedSection(detail, state, scope) {
  if (!detail) return sec('Served model', null, h('p', { class: 'dr-quiet' }, state === 'loading' ? 'Loading…' : 'The detail for this session is unavailable.'));
  const sm = detail.served_model || {};
  const key = SERVED[sm.state] ? sm.state : 'unknown';
  const [word, text] = SERVED[key];
  const extra = key === 'unknown' && scope === 'enterprise' ? ' The check runs for the personal scope only.'
    : sm.blocks ? ` Blocks checked: ${f.int(sm.blocks)}.` : '';
  return sec('Served model', null, h('div', { class: 'served', 'data-state': key }, h('div', { class: 'served-word' }, word), h('p', null, text + extra)));
}

export default {
  id: 'ledger-drawer',
  label: 'Session detail',
  mount(el, ctx) {
    const { store } = ctx;
    const scrim = h('div', { class: 'dr-scrim', hidden: true });
    const headEl = h('div', { class: 'dr-head' });
    const bodyEl = h('div', { class: 'dr-body' });
    const drawer = h('aside', { class: 'drawer', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'dr-title', hidden: true }, headEl, bodyEl);
    el.append(scrim, drawer);
    scrim.addEventListener('click', () => closeSession(store));

    let openId = null, detail = null, dstate = 'idle', ctrl = null, seenD = null, hideTimer = null;
    // Sessions whose run tree shows every run ("Show all N runs").
    const allRuns = new Set();
    const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

    function order(d) {
      return matched(d.listed, d.ui.filters).map((m) => m.s).sort((a, b) => a.start - b.start).map((x) => x.id);
    }
    function rowFor(d, id) {
      return d.sessions.find((x) => x.id === id) || normalize(store.get().ledger && store.get().ledger.data).byId.get(id) || null;
    }

    function render(focusNav) {
      const act = document.activeElement;
      const hadFocus = act && drawer.contains(act) ? act.getAttribute('aria-label') || 'Close' : null;
      const st = store.get();
      const d = derive(st, ctx.now());
      const row = rowFor(d, openId);
      const s = detail ? { ...normalize({ sessions: [detail] }).sessions[0], fileEdits: detail.file_edits } : row;
      if (!s) { patch(headEl, h('h2', { class: 'dr-title', id: 'dr-title' }, 'Session'), h('button', { type: 'button', class: 'dr-close', 'aria-label': 'Close', onclick: () => closeSession(store) }, '×')); patch(bodyEl, h('p', { class: 'dr-sec dr-quiet' }, dstate === 'loading' ? 'Loading the session…' : 'This session is not in the data.')); return; }
      const c = colorsFor(d);
      const ids = order(d), i = ids.indexOf(s.id);
      const prev = i > 0 ? ids[i - 1] : null, next = i >= 0 && i < ids.length - 1 ? ids[i + 1] : null;
      const nav = (id, label, sym) => h('button', { type: 'button', 'aria-label': label, disabled: !id, 'data-go': id || null, onclick: () => go(id, label) }, sym);
      const wall = Math.max(0, s.end - s.start);
      const live = s.live || (row && row.live);
      patch(headEl,
        h('div', { class: 'dr-kicker' }, h('span', { class: 'kicker' }, [s.machine, s.project && projectLabel(s.project), clientName(s.client)].filter(Boolean).join(' · ')), live ? h('span', { class: 'live-tag' }, 'Live') : null),
        h('div', { class: 'dr-nav' }, nav(prev, 'Previous session', '‹'), nav(next, 'Next session', '›'),
          h('button', { type: 'button', class: 'dr-close', 'aria-label': 'Close', onclick: () => closeSession(store) }, '×')),
        h('h2', { class: 'dr-title', id: 'dr-title' }, sessionName(s)),
        h('div', { class: 'dr-when' }, h('b', null, `${f.dayClock(s.start)} → ${live ? 'now' : f.when(s.end, s.start)}`), ` · ${f.shortDur(wall)} wall · ${f.shortDur(s.active)} active`));
      const runs = s.runs;
      patch(bodyEl,
        costSection(s, row, live),
        sec('Fan-out', runs.length ? `${plural(runs.length, 'run')} · peak ${peakParallel(runs)} at once` : 'no subagents',
          renderGantt(s, detail ? detail.prompt_times : null, c, live, (bodyEl.querySelector('.gantt') || {}).clientWidth), runTree(s, c, allRuns.has(s.id), () => { allRuns.add(s.id); render(null); })),
        modelsSection(s, detail, c),
        toolsTokens(s, detail, dstate),
        servedSection(detail, dstate, st.scope),
        h('section', { class: 'dr-sec' }, h('div', { class: 'dr-id' }, `session ${s.id} · ${plural(runs.length + 1, 'thread')}`)));
      // A re-render (the detail arrives, the data polls) replaces the head
      // buttons; keep the focus on the same control, or on Close.
      const keep = focusNav || (hadFocus ? hadFocus : null);
      if (keep) {
        const again = headEl.querySelector(`[aria-label="${CSS.escape(keep)}"]`);
        (again && !again.disabled ? again : headEl.querySelector('.dr-close')).focus();
      }
    }

    function fetchDetail(id) {
      if (ctrl) ctrl.abort();
      ctrl = new AbortController();
      const mine = ctrl;
      detail = null; dstate = 'loading';
      loadDetail(ctx, id, mine.signal).then((data) => {
        if (mine.signal.aborted || openId !== id) return;
        detail = data && data.id === id ? data : null;
        dstate = detail ? 'ok' : 'error';
        render(null);
      }).catch((err) => {
        if (mine.signal.aborted || openId !== id) return;
        dstate = 'error';
        if (err.status !== 404) console.warn('tokenfold: session detail failed', err.message);
        render(null);
      });
    }

    function go(id, label) {
      if (!id) return;
      openSession(store, id);
      openId = id;
      fetchDetail(id);
      render(label);
      bodyEl.scrollTop = 0;
    }

    function open(id) {
      clearTimeout(hideTimer);
      openId = id;
      fetchDetail(id);
      scrim.hidden = false; drawer.hidden = false;
      render(null);
      requestAnimationFrame(() => { scrim.classList.add('is-open'); drawer.classList.add('is-open'); headEl.querySelector('.dr-close').focus(); });
    }
    function close() {
      openId = null;
      if (ctrl) ctrl.abort();
      scrim.classList.remove('is-open'); drawer.classList.remove('is-open');
      hideTimer = setTimeout(() => { scrim.hidden = true; drawer.hidden = true; }, reduced() ? 0 : 230);
    }

    function onKey(ev) {
      if (!openId) return;
      if (ev.key === 'Escape') { ev.preventDefault(); closeSession(store); return; }
      if (ev.key !== 'Tab') return;
      const items = [...drawer.querySelectorAll(FOCUSABLE)].filter((n) => n.offsetParent !== null);
      if (!items.length) return;
      const first = items[0], last = items[items.length - 1];
      if (!drawer.contains(document.activeElement)) { ev.preventDefault(); first.focus(); } else if (ev.shiftKey && document.activeElement === first) { ev.preventDefault(); last.focus(); } else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first.focus(); }
    }
    document.addEventListener('keydown', onKey);
    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(() => { if (openId) { const g = bodyEl.querySelector('.gantt'); if (g && g.dataset.w !== String(g.clientWidth)) render(null); } }) : null;
    if (ro) ro.observe(drawer);

    return {
      update(s) {
        const want = s.ledgerDrawer || null;
        if (want !== openId) { if (want) open(want); else if (openId) close(); return; }
        const d = derive(s, ctx.now());
        if (openId && d !== seenD) { seenD = d; render(null); }
      },
      destroy() {
        document.removeEventListener('keydown', onKey);
        if (ctrl) ctrl.abort();
        if (ro) ro.disconnect();
        clearTimeout(hideTimer);
        // Leaving the view closes the drawer; coming back must not reopen it.
        if (store.get().ledgerDrawer) store.set('ledgerDrawer', null);
      },
    };
  },
};

