// Number, money and time formatting. Every function returns DASH for a missing
// value instead of inventing a zero, so "unknown" never reads as "none".
// Clock times use the browser's local time zone.

export const DASH = '—';
export const missing = (v) => v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v));

const intFmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
const centsFmt = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// Money: whole dollars from $100 up, cents below (a session's $6.12 matters, a day's $412.30 does not).
export function usd(v, { cents } = {}) {
  if (missing(v)) return DASH;
  const a = Math.abs(v);
  const sign = v < 0 ? '−' : '';
  const useCents = cents === undefined ? a < 100 : cents;
  return `${sign}$${useCents ? centsFmt.format(a) : intFmt.format(Math.round(a))}`;
}

// Compact money for axis labels: $950, $1.5K, $12K.
export function usdShort(v) {
  if (missing(v)) return DASH;
  v = Math.round(v);
  if (Math.abs(v) < 1000) return `$${intFmt.format(Math.round(v))}`;
  const k = v / 1000;
  return `$${k >= 10 ? Math.round(k) : k.toFixed(1).replace(/\.0$/, '')}K`;
}

export function pct(v, dp = 0) {
  if (missing(v)) return DASH;
  return `${v.toFixed(dp)}%`;
}

export const int = (v) => (missing(v) ? DASH : intFmt.format(v));
export const fixed = (v, dp = 1) => (missing(v) ? DASH : v.toFixed(dp));

// Long spans: "3 d 4 h", "12 h 15 min", "15 min 48 s" (withSecs), "40 s".
export function dur(s, withSecs = false) {
  if (missing(s)) return DASH;
  const t = Math.max(0, Math.round(s));
  const d = Math.floor(t / 86400), h = Math.floor((t % 86400) / 3600);
  const m = Math.floor((t % 3600) / 60), sec = t % 60;
  if (d) return `${d} d ${h} h`;
  if (h) return `${h} h ${m} min`;
  if (withSecs && m) return `${m} min ${String(sec).padStart(2, '0')} s`;
  return m ? `${m} min` : `${sec} s`;
}

// Short spans for dense rows: "<1m", "4m", "1h 12m", "2d 3h".
export function shortDur(s) {
  if (missing(s)) return DASH;
  const m = Math.floor(Math.max(0, s) / 60);
  if (m < 1) return '<1m';
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${m % 60}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

export function age(s) {
  if (missing(s)) return DASH;
  if (s < 90) return `${Math.max(0, Math.round(s))} s ago`;
  return `${shortDur(s)} ago`;
}

const clockFmt = new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit' });
const clockSecFmt = new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', second: '2-digit' });
const dowFmt = new Intl.DateTimeFormat('en-US', { weekday: 'short' });
const dateFmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });
const hourFmt = new Intl.DateTimeFormat('en-US', { hour: 'numeric' });

const ms = (epoch) => new Date(epoch * 1000);
export const clock = (e, secs = false) => (missing(e) ? DASH : (secs ? clockSecFmt : clockFmt).format(ms(e)));
export const dow = (e) => (missing(e) ? DASH : dowFmt.format(ms(e)));
export const dayClock = (e) => (missing(e) ? DASH : `${dow(e)} ${clock(e)}`);
export const dateShort = (e) => (missing(e) ? DASH : dateFmt.format(ms(e)));
// "5 AM" -> "5a": axis labels need the short form.
export const hourShort = (e) => hourFmt.format(ms(e)).replace(' AM', 'a').replace(' PM', 'p');

const sameDay = (a, b) => ms(a).toDateString() === ms(b).toDateString();

// "3:00 PM" today, "Thu 3:00 AM" within six days, "Nov 1" later.
export function when(e, ref) {
  if (missing(e)) return DASH;
  if (sameDay(e, ref)) return clock(e);
  if (Math.abs(e - ref) < 6 * 86400) return dayClock(e);
  return dateShort(e);
}

// Local midnight (epoch seconds) of the day that holds `e`.
export function localMidnight(e) {
  const d = ms(e);
  d.setHours(0, 0, 0, 0);
  return d.getTime() / 1000;
}

// Plain text escaping for the rare string that must go through innerHTML.
// Panels build DOM with textContent (dom.js h()); prefer that.
export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
