// Lab: the checks and reference tables that are not part of the daily loop.
// DOM order is the phone order: served-model check, energy and water, price
// table, benchmarks and cost per hour. Desktop placement: css/lab.css.
import served from '../panels/lab-served.js';
import energy from '../panels/lab-energy.js';
import prices from '../panels/lab-prices.js';
import bench from '../panels/lab-bench.js';

export default {
  id: 'lab',
  label: 'Lab',
  feeds: {
    // /api/stats is the classic dashboard payload; the server caches it per scope.
    stats: { endpoint: 'stats', bump: true },
    // Personal data only (the server answers 404 for work): never ask from the work scope.
    servedTimeline: {
      endpoint: 'servedTimeline', bump: true,
      params: (s) => (s.scope === 'enterprise' ? null : { days: s.labServedDays || '30', sessions: 0 }),
    },
  },
  panels: [served, energy, prices, bench],
};
