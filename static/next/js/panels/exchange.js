// Exchange rate (/api/limit-rate): what one point of a limit window costs in
// API-equivalent dollars, by model family, the peak-hour factor on the 5-hour
// window and how much cache reads weigh. All values are the server's fit.
import { h, fill, approx, panelHead } from '../dom.js';
import * as f from '../format.js';
import { dataOf, nowFrom } from '../select.js';
import { familyColor, FAMILY_NAME } from '../palette.js';
import { feedLine, staleNote } from '../panel.js';

const CAUSE = { granted_reset: 'granted reset', rate_change: 'rate change' };
const BUCKET = { seven_day: '7-day', five_hour: '5-hour' };
const HOUR = 3600;
// Fallback for a payload without peak.min_factor: app/limit_peak.py PEAK_MIN_FACTOR.
const PEAK_MIN_FACTOR = 1.15;

// Day of week (0 = Monday, as the server) and hour of `epoch` in time zone `tz`.
function zonedParts(epoch, tz) {
  const parts = new Intl.DateTimeFormat('en-US', { timeZone: tz, weekday: 'short', hour: 'numeric', hourCycle: 'h23' })
    .formatToParts(new Date(epoch * 1000));
  const get = (type) => (parts.find((p) => p.type === type) || {}).value;
  const dow = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].indexOf(get('weekday'));
  return { dow, hour: Number(get('hour')) % 24 };
}

const bandEl = (lo, hi, value, max) => h('span', { class: 'band' },
  f.missing(lo) || f.missing(hi) ? null : h('u', { style: { left: `${lo / max * 100}%`, width: `${Math.max(0.5, (hi - lo) / max * 100)}%` } }),
  f.missing(value) ? null : h('b', { style: { left: `${Math.min(100, value / max * 100)}%` } }));

function sizeBlock(d) {
  const seven = d.usd_per_point && d.usd_per_point.seven_day;
  const five = d.usd_per_point && d.usd_per_point.five_hour;
  const steps = (d.steps || []).filter((s) => s.bucket === 'seven_day').slice().sort((a, b) => b.at - a.at).slice(0, 3);
  const q = d.fit_quality || {};
  return h('div', { class: 'ex-block' },
    h('div', { class: 'kicker' }, 'Limit size now'),
    h('div', { class: 'ex-big' }, h('span', { class: 'ex-v' }, seven && !f.missing(seven.value) ? [approx(), f.usd(seven.value * 100, { cents: false })] : f.DASH),
      h('span', { class: 'ex-unit' }, 'per 7-day window')),
    h('p', { class: 'ex-text' },
      seven && !f.missing(seven.value) ? [h('b', null, `${f.usd(seven.value, { cents: true })} per point`), ` (${f.usd(seven.lo, { cents: true })} to ${f.usd(seven.hi, { cents: true })}). `] : null,
      five && !f.missing(five.value) ? [`A 5-hour window holds ≈${f.usd(five.value * 100, { cents: false })}.`] : null),
    steps.length ? h('ol', { class: 'ex-steps', 'aria-label': 'Last changes of the limit size' }, steps.map((s) => h('li', null,
      h('span', { class: 'sd' }, f.dateShort(s.at)),
      h('span', { class: 'sc' }, `${CAUSE[s.cause] || s.cause} · ${f.usd(s.from_usd_per_point, { cents: true })} → ${f.usd(s.to_usd_per_point, { cents: true })} per point`)))) : null,
    h('p', { class: 'ex-text' }, `Fit from ${f.int(q.intervals)} reading intervals since ${f.dateShort(d.period_start)}${f.missing(q.r2) ? '' : `, R² ${q.r2.toFixed(2)}`}.`));
}

function familyBlock(d) {
  const fams = d.families || [];
  const max = Math.max(10, ...fams.map((x) => x.hi ?? x.usd_per_point ?? 0)) * 1.05;
  return h('div', { class: 'ex-block' },
    h('div', { class: 'kicker' }, 'Dollars per 7-day point, by model'),
    fams.length ? h('ul', { class: 'fam' }, fams.map((x) => h('li', null,
      h('span', { class: 'swatch', style: { background: familyColor(x.family) } }),
      h('span', { class: 'fn' }, FAMILY_NAME[x.family] || x.family),
      f.missing(x.usd_per_point) ? h('span', { class: 'fnote' }, x.note || 'not enough data') : bandEl(x.lo, x.hi, x.usd_per_point, max),
      h('span', { class: 'fv' }, f.missing(x.usd_per_point) ? f.DASH : [approx(), f.usd(x.usd_per_point, { cents: true })]))))
      : h('p', { class: 'ex-text' }, 'No model has enough spend for a fit.'),
    h('p', { class: 'ex-text' }, 'More dollars per point means the model uses less limit for each dollar. Bars show the 90% range.'));
}

function peakBlock(d, now) {
  const peak = d.peak;
  if (!peak) return h('div', { class: 'ex-block' }, h('div', { class: 'kicker' }, 'Peak hours'), h('p', { class: 'ex-text' }, 'No peak data.'));
  const cells = new Map((peak.hours || []).map((c) => [`${c.dow}:${c.hour}`, c.factor]));
  // The server's peak rule (limit_peak.PEAK_MIN_FACTOR), so a red hour here
  // is an hour the server counts as peak.
  const minFactor = f.missing(peak.min_factor) ? PEAK_MIN_FACTOR : peak.min_factor;
  const midnight = f.localMidnight(now);
  const curHour = Math.floor((now - midnight) / HOUR);
  const strip = Array.from({ length: 24 }, (_, i) => {
    const t = midnight + i * HOUR + 1;
    const z = zonedParts(t, peak.tz || 'America/Los_Angeles');
    const factor = cells.get(`${z.dow}:${z.hour}`);
    const known = !f.missing(factor);
    const height = known ? Math.max(8, Math.min(100, (factor - 0.6) / 0.9 * 100)) : 100;
    const cls = [known ? (factor >= minFactor ? 'pk' : '') : 'nd', i === curHour ? 'cur' : ''].join(' ').trim();
    return h('i', { class: cls || null, style: { height: `${height}%` }, title: `${f.hourShort(t)}: ${known ? `×${factor.toFixed(2)}` : 'no data'}` });
  });
  const status = peak.active
    ? [h('span', { class: 'chip chip-red' }, `Peak now ×${f.fixed(peak.factor, 2)}`), f.missing(peak.until) ? null : h('span', null, `ends ${f.clock(peak.until)}`)]
    : [h('span', { class: 'chip chip-line' }, 'Off peak'), f.missing(peak.next_start) ? null : h('span', null, `next peak ${f.when(peak.next_start, now)}`)];
  return h('div', { class: 'ex-block' },
    h('div', { class: 'kicker' }, 'Peak hours, today'),
    h('div', { class: 'peak-now' }, status),
    h('div', { class: 'hours', role: 'img', 'aria-label': `Limit use per hour today. Red hours are peak hours, about ×${f.fixed(peak.factor, 2)}.` }, strip),
    h('div', { class: 'hours-x' }, ['12a', '6a', '12p', '6p'].map((t) => h('span', null, t))),
    h('p', { class: 'ex-text' }, `In peak hours the 5-hour window fills ≈${f.fixed(peak.factor, 2)} times as fast. The 7-day window does not change.`));
}

function cacheBlock(d) {
  const c = d.cache_read_weight;
  return h('div', { class: 'ex-block' },
    h('div', { class: 'kicker' }, 'Cache reads'),
    h('div', { class: 'ex-big' }, h('span', { class: 'ex-v' }, c && !f.missing(c.value) ? [approx(), `×${c.value.toFixed(2)}`] : f.DASH)),
    c && !f.missing(c.value) ? bandEl(c.lo, c.hi, c.value, 1) : null,
    h('p', { class: 'ex-text' }, c && !f.missing(c.value)
      ? `Cache reads count at about ${Math.round(c.value * 100)}% of their list price (${f.fixed(c.lo, 2)} to ${f.fixed(c.hi, 2)}).`
      : 'Not enough data to weigh cache reads.'));
}

// Server notes arrive as lower-case fragments; print them as a sentence.
const sentence = (t) => { const v = String(t).trim(); return v ? `${v[0].toUpperCase()}${v.slice(1)}${/[.!?]$/.test(v) ? '' : '.'}` : v; };

export default {
  id: 'exchange',
  label: 'Exchange rate',
  mount(el) {
    const head = panelHead('Exchange rate');
    const sub = h('span', { class: 'panel-sub' }, 'limit points in API-equivalent dollars');
    head.el.append(sub);
    el.setAttribute('aria-labelledby', head.id);
    const state = h('div');
    const body = h('div', { class: 'ex-body' });
    el.append(head.el, state, body);

    let seen = null, seenMinute = null, seenScope = null;
    function update(s) {
      // Only the personal subscription has limit points: no rate can exist for work.
      el.hidden = s.scope === 'enterprise';
      if (el.hidden) return;
      const now = nowFrom(s);
      // Redraw only for new data or a new minute: a 5 s poll elsewhere must not reset hover titles.
      if (s.limitRate === seen && Math.floor(now / 60) === seenMinute && s.scope === seenScope) return;
      seen = s.limitRate; seenMinute = Math.floor(now / 60); seenScope = s.scope;
      const d = dataOf(s.limitRate);
      const line = feedLine(s.limitRate, 'the exchange rate');
      const q = d && d.fit_quality ? d.fit_quality : {};
      if (line || (q.status && q.status !== 'ok' && q.status !== 'stale')) {
        fill(state, line || h('div', { class: 'p-state' },
          q.status === 'computing' ? 'The fit is in progress.' : 'No exchange rate for this scope.', q.note ? h('small', null, sentence(q.note)) : null));
        fill(body);
        return;
      }
      fill(state, staleNote(s.limitRate, now), q.status === 'stale' ? h('span', { class: 'age' }, q.note || 'The fit is old.') : null);
      fill(body, sizeBlock(d), familyBlock(d), peakBlock(d, now), cacheBlock(d));
    }

    return { update };
  },
};
