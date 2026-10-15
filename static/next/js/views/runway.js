// Runway (landing view): "can I keep working, and what burns my limit now?"
// Panel order in the DOM is the phone reading order: hero, Now, other
// windows, chart, Today, Exchange rate. The desktop grid places them by area
// (css: .p-hero, .p-now, .p-rail, .p-chart, .p-today, .p-exchange). The work
// scope has no quota windows and no exchange rate: those two panels hide
// and Now takes the rail's place.
import hero from '../panels/hero.js';
import now from '../panels/now.js';
import rail from '../panels/rail.js';
import trajectory from '../panels/trajectory.js';
import today from '../panels/today.js';
import exchange from '../panels/exchange.js';
import { dataOf } from '../select.js';

export default {
  id: 'runway',
  label: 'Runway',
  // `now` and `runway` are shell feeds (main.js); a view may want them again.
  feeds: {
    now: { endpoint: 'now', intervalMs: 5000 },
    runway: { endpoint: 'runway', intervalMs: 30000, bump: true },
    today: { endpoint: 'today', intervalMs: 60000, bump: true },
    limitRate: { endpoint: 'limitRate', bump: true, params: (s) => (s.scope === 'enterprise' ? null : {}) },
    // Waits for the binding window, then refetches when it changes.
    trajectory: {
      endpoint: 'trajectory', bump: true, intervalMs: 120000,
      params: (s) => { const r = dataOf(s.runway); return r && r.binding ? { window: r.binding } : null; },
    },
  },
  panels: [hero, now, rail, trajectory, today, exchange],
};
