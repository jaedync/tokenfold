// Hero: "can I keep working?" The binding window (personal) or the monthly
// billing meter against the budget (work), its verdict, the pace facts and
// what burns it now. Every forecast here is a server field (pace block,
// tight_epoch, verdict); the browser only picks words for them.
import { h, patch, setText, setAttr, approx } from '../dom.js';
import * as f from '../format.js';
import { bindingWindow, claudeShortWindow, dataOf, roomElsewhere, nowFrom } from '../select.js';
import { familyColor, splitLabel } from '../palette.js';
import { feedLine, staleNote } from '../panel.js';

const PACE_WORD = { above: 'over pace', on: 'on pace', below: 'under pace', too_early: 'too early' };
const clamp = (v) => Math.max(0, Math.min(100, v));

const fact = (k, v, sub) => h('div', null, h('dt', null, k), h('dd', null, h('span', { class: 'fv' }, v), h('span', { class: 'fs' }, sub)));

function verdictPersonal(w, now) {
  const p = w.pace || {};
  const rate = w.kind === 'short' ? '1-hour' : '6-hour';
  switch (w.verdict) {
    case 'runs_out':
      return ['out', `Runs out ${f.when(p.reaches_100_epoch, now)}`,
        `At the window average, this window reaches 100% ${f.dur(w.resets_at - p.reaches_100_epoch)} before it resets.`];
    case 'tight':
      return ['tight', 'Tight',
        `At the window average you reach ≈${f.pct(p.projected_pct)} at reset. At the ${rate} rate you reach 100% at ≈${f.when(w.tight_epoch, now)}, ${f.dur(w.resets_at - w.tight_epoch)} before the reset.`];
    case 'clear':
      return ['clear', 'Clear to reset', `At the window average this window ends near ≈${f.pct(p.projected_pct)}.`];
    case 'too_early':
      return ['none', 'Too early', `The window started ${f.dur(now - w.window_start)} ago. The forecast needs more data.`];
    default:
      return ['none', 'No forecast', 'The data for this window is old, or the window ended.'];
  }
}

function verdictWork(w, budget) {
  const proj = budget.projected_usd;
  const over = f.missing(proj) ? null : proj - budget.budget_usd;
  const allowance = f.missing(budget.per_workday_left_usd) ? '' : ` Keep under ${f.usd(budget.per_workday_left_usd)} per workday to land on budget.`;
  switch (w.verdict) {
    case 'runs_out':
      return ['out', 'Over budget pace', `At this pace the month ends near ≈${f.usd(proj)}, ≈${f.usd(over)} over budget.${allowance}`];
    case 'tight':
      return ['tight', 'Tight', `At this pace the month ends near ≈${f.usd(proj)}.${allowance}`];
    case 'clear':
      return ['clear', 'Within budget', `At this pace the month ends near ≈${f.usd(proj)}.`];
    case 'too_early':
      return ['none', 'Too early', 'The month is new. The forecast needs more workdays.'];
    default:
      return ['none', 'No forecast', 'The billing meter is old or missing.'];
  }
}

// Pace facts. Too early: the elapsed time and even pace only; an over/under
// figure this early is noise, and the server says it is too early.
function paceSub(w, now) {
  const p = w.pace || {};
  if (w.verdict === 'too_early' || p.pace_state === 'too_early') {
    return [`${f.dur(now - w.window_start)} into the window`,
      f.missing(p.expected_pct) ? null : `even pace ${f.pct(p.expected_pct)}`].filter(Boolean).join(' · ');
  }
  const diff = !f.missing(w.pct) && !f.missing(p.expected_pct) ? w.pct - p.expected_pct : null;
  const h6 = w.burn_pct_per_h ? w.burn_pct_per_h.h6 : null;
  return [diff === null ? null : `${Math.abs(diff).toFixed(1)} pts ${diff < 0 ? 'under' : 'over'} even pace`,
    f.missing(h6) ? null : `6 h rate ${h6.toFixed(2)}%/h`].filter(Boolean).join(' · ');
}

// What the live sessions add to the meter rate. The rate above is the quota
// meter's; the per-session split shows only when agents report live state.
function burnDetail(now, meterRate) {
  if (!now) return null;
  const sessions = Array.isArray(now.sessions) ? now.sessions : [];
  if (!sessions.length) {
    return h('p', { class: 'burn-none' }, 'No live session data. Agents report live state through the notify hooks.');
  }
  const rows = sessions.filter((x) => x.state === 'working' && x.share_of_binding_pct_per_h > 0)
    .sort((a, b) => b.share_of_binding_pct_per_h - a.share_of_binding_pct_per_h);
  if (!rows.length) return h('p', { class: 'burn-none' }, 'No live session uses this window now.');
  // Full bar = the meter rate; the paper rest is what no live session explains.
  const sum = rows.reduce((a, x) => a + x.share_of_binding_pct_per_h, 0);
  const scale = Math.max(sum, f.missing(meterRate) ? 0 : meterRate) || 1;
  return [
    h('div', { class: 'burn-sub' }, 'Live sessions, share of the meter rate'),
    h('div', { class: 'burnbar', role: 'img', 'aria-label': `Live sessions explain ≈${Math.round(sum / scale * 100)}% of the meter rate` },
      rows.map((x) => h('span', { 'data-key': x.id, style: { width: `${x.share_of_binding_pct_per_h / scale * 100}%`, background: familyColor(x.family) } }))),
    h('ul', { class: 'burn-list' }, rows.map((x) => h('li', { 'data-key': x.id },
      h('span', { class: 'swatch', style: { background: familyColor(x.family) } }),
      h('span', { class: 'bn' }, h('b', null, x.machine), ` ${x.project || ''}`, x.subagents ? ` · ${x.subagents} sub` : ''),
      h('span', { class: 'bv' }, `≈+${x.share_of_binding_pct_per_h.toFixed(2)}%/h`),
      h('span', { class: 'bd' }, `≈${f.usd(x.burn_usd_per_h, { cents: false })}/h`)))),
  ];
}

export default {
  id: 'hero',
  label: 'Binding window',
  mount(el, ctx) {
    const kick = h('span', { class: 'kicker', id: 'hero-k' }, 'Binding window');
    const name = h('b', null, f.DASH);
    const big = h('div', { class: 'hero-v' }, f.DASH);
    const unit = h('div', { class: 'hero-unit' });
    const used = h('i'), fore = h('u'), notch = h('b');
    const meter = h('div', { class: 'meter', role: 'img' }, used, fore, notch);
    const s0 = h('span'), sPace = h('span'), s1 = h('span');
    const word = h('div', { class: 'verdict-word' }), text = h('div', { class: 'verdict-text' });
    const verdict = h('div', { class: 'verdict', 'data-tone': 'none' }, word, text);
    const facts = h('dl', { class: 'facts' });
    const burn = h('div', { class: 'burn' });
    const short = h('div', { class: 'short' });
    const note = h('p', { class: 'api-note' });
    const age = h('span');
    const body = h('div', { class: 'hero-body', style: { display: 'contents' } },
      h('div', { class: 'big-wrap' }, h('div', { class: 'hero-big' }, big, unit)), meter,
      h('div', { class: 'meter-scale' }, s0, sPace, s1), verdict, facts, burn, short, note);
    const state = h('div');
    el.setAttribute('aria-labelledby', 'hero-k');
    el.append(h('div', { class: 'hero-k' }, kick, name, age), state, body);

    function setMeter(usedPct, projPct, expectedPct, label) {
      used.style.width = `${clamp(usedPct)}%`;
      fore.style.left = `${clamp(usedPct)}%`;
      fore.style.width = f.missing(projPct) ? '0%' : `${Math.max(0, clamp(projPct) - clamp(usedPct))}%`;
      notch.hidden = f.missing(expectedPct);
      if (!f.missing(expectedPct)) notch.style.left = `${clamp(expectedPct)}%`;
      setAttr(meter, 'aria-label', label);
    }

    function setVerdict([tone, w, t]) {
      setAttr(verdict, 'data-tone', tone);
      setText(word, w);
      setText(text, t);
    }

    function renderPersonal(w, s, now) {
      const p = w.pace || {};
      setText(kick, 'Binding window');
      setText(name, w.label);
      patch(big, f.missing(w.pct) ? f.DASH : String(Math.floor(w.pct)), h('small', null, '%'));
      patch(unit, 'used', h('span', null, 'of the window'));
      setMeter(w.pct, p.projected_pct, p.expected_pct,
        `${f.pct(w.pct)} used. Even pace is ${f.pct(p.expected_pct)} now. Forecast at reset: ${f.pct(p.projected_pct)}.`);
      setText(s0, '0');
      setText(sPace, f.missing(p.expected_pct) ? '' : `pace now ${f.pct(p.expected_pct)}`);
      setText(s1, '100%');
      setVerdict(verdictPersonal(w, now));
      patch(facts,
        fact('Pace', PACE_WORD[p.pace_state] || f.DASH, paceSub(w, now) || ' '),
        fact('Forecast', f.missing(p.projected_pct) ? f.DASH : [approx(), `${f.pct(p.projected_pct)} at reset`],
          f.missing(p.projected_pct) ? 'needs more data' : 'if the window average holds'),
        fact('Resets', f.dayClock(w.resets_at), `in ${f.dur(w.resets_at - now)}`),
        fact('Left', f.missing(w.usd_left) ? f.DASH : [approx(), f.usd(w.usd_left)],
          f.missing(w.usd_per_point) ? 'value per point unknown' : `${(100 - w.pct).toFixed(1)} pts × ≈${f.usd(w.usd_per_point, { cents: true })} per point`));
      renderBurn(w, s);
      renderShort(s);
      patch(note, h('b', null, 'Dollars are API-equivalent at list price, not billed.'),
        ' Your subscription pays for them. They show what the rest of the quota is worth.');
    }

    function renderBurn(w, s) {
      const h1 = w.burn_pct_per_h ? w.burn_pct_per_h.h1 : null;
      burn.hidden = false;
      patch(burn,
        h('div', { class: 'burn-head' },
          h('span', { class: 'kicker' }, 'Meter rate, last hour'),
          h('b', null, f.missing(h1) ? f.DASH : `+${h1.toFixed(2)}%/h`)),
        burnDetail(dataOf(s.now), h1));
    }

    function renderShort(s) {
      const runway = dataOf(s.runway);
      const fh = claudeShortWindow(runway);
      short.hidden = !fh;
      if (!fh) return;
      const now = nowFrom(s);
      const chip = fh.waitable ? h('span', { class: 'chip chip-yellow', title: 'If it fills, a wait for its reset is short' }, 'Waitable')
        : (fh.verdict === 'tight' || fh.verdict === 'runs_out') ? h('span', { class: 'chip chip-red' }, 'Watch') : null;
      const room = roomElsewhere(runway).map((w) => h('b', null, `${splitLabel(w).prov} ${splitLabel(w).name.toLowerCase()} ${f.pct(w.pct)}`));
      patch(short, h('div', { class: 'kicker' }, 'Short window'),
        h('div', { class: 'short-row' }, h('span', null, `${splitLabel(fh).prov} ${splitLabel(fh).name}`),
          h('span', { class: 'num' }, f.pct(fh.pct)),
          h('span', null, `resets ${f.when(fh.resets_at, now)} · in ${f.dur(fh.resets_at - now, true)}`), chip),
        room.length ? h('div', { class: 'elsewhere' }, 'Room elsewhere: ', room.flatMap((r, i) => (i ? [' · ', r] : [r])), '.') : null);
    }

    function renderWork(w, budget, now) {
      const p = w.pace || {};
      const b = budget || {};
      const projPct = !f.missing(b.projected_usd) && b.budget_usd ? b.projected_usd / b.budget_usd * 100 : p.projected_pct;
      setText(kick, 'Billing meter');
      setText(name, w.label);
      patch(big, f.missing(w.pct) ? f.DASH : String(Math.floor(w.pct)), h('small', null, '%'));
      patch(unit, 'of budget', h('span', null, `${f.usd(b.spent_usd ?? w.usd_spent)} of ${f.usd(b.budget_usd)}`));
      setMeter(w.pct, projPct, p.expected_pct,
        `${f.pct(w.pct)} of the budget is billed. Workday pace is ${f.pct(p.expected_pct)} now.`);
      const expectedUsd = !f.missing(p.expected_pct) && b.budget_usd ? p.expected_pct / 100 * b.budget_usd : null;
      setText(s0, '$0');
      setText(sPace, f.missing(expectedUsd) ? '' : `pace now ${f.usd(expectedUsd)}`);
      setText(s1, f.usd(b.budget_usd));
      setVerdict(verdictWork(w, b));
      const ahead = !f.missing(expectedUsd) ? (b.spent_usd ?? w.usd_spent) - expectedUsd : null;
      patch(facts,
        fact('Pace', PACE_WORD[b.pace_state || p.pace_state] || f.DASH,
          f.missing(ahead) ? ' ' : `${f.usd(Math.abs(ahead))} ${ahead >= 0 ? 'ahead of' : 'behind'} an even workday spread`),
        fact('Forecast', f.missing(b.projected_usd) ? f.DASH : [approx(), f.usd(b.projected_usd)], `at month end, ${f.dateShort(w.resets_at - 1)}`),
        fact('Allowance', f.missing(b.per_workday_left_usd) ? f.DASH : `${f.usd(b.per_workday_left_usd)} / workday`, 'for the rest of the month'),
        fact('Meter', f.age(now - w.observed_at), 'Anthropic billing meter'));
      burn.hidden = true;
      short.hidden = false;
      patch(short, h('p', { class: 'burn-none' }, 'The billing meter is the limit. Work usage has no quota windows.'));
      patch(note, h('b', null, 'Work dollars on the meter are billed dollars.'),
        ' Session and daily figures are list-price estimates (≈) until the meter catches up.');
    }

    function update(s) {
      const runway = dataOf(s.runway);
      const line = feedLine(s.runway, 'the limit feed');
      const w = bindingWindow(runway);
      if (line || !w) {
        patch(state, line || (runway ? h('div', { class: 'p-state' }, 'No limit window has data yet.',
          h('small', null, 'The quota feed reports every 2 min.')) : null));
        body.hidden = true;
        setText(name, '');
        return;
      }
      patch(state);
      body.hidden = false;
      const now = nowFrom(s);
      if (w.kind === 'month') renderWork(w, runway.budget, now); else renderPersonal(w, s, now);
      big.classList.toggle('is-stale', Boolean(w.stale));
      patch(age, w.stale ? h('span', { class: 'age' }, `Window data ${f.age(w.age_s)}`) : null, staleNote(s.runway, now));
    }

    return { update };
  },
};
