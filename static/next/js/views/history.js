// History: how past limit windows ended and how the spend changed month by
// month. DOM order is the phone order: billing meter (work scope only), limit
// windows, spend by month, value. The desktop grid places them in css/history.css
// (linked from templates/next.html).
import meter from '../panels/history-meter.js';
import windows from '../panels/history-windows.js';
import months from '../panels/history-months.js';
import value from '../panels/history-value.js';

const personal = (s) => s.scope !== 'enterprise';


export default {
  id: 'history',
  label: 'History',
  feeds: {
    // Limit windows are personal (Max) data: the work scope never asks for them.
    windowHistory: { endpoint: 'windowHistory', bump: true, params: (s) => (personal(s) ? { bucket: 'seven_day' } : null) },
    windowHistory5h: {
      endpoint: 'windowHistory5h', bump: true,
      params: (s) => (personal(s) && s.historyBucket === 'five_hour' ? { bucket: 'five_hour' } : null),
    },
    spendHistory: { endpoint: 'spendHistory', bump: true },
    // The meter lives in the cached /api/stats payload; only the work scope reads it.
    stats: { endpoint: 'stats', bump: true, params: (s) => (personal(s) ? null : {}) },
    enterpriseBudget: { endpoint: 'enterpriseBudget', params: (s) => (personal(s) ? null : {}) },
  },
  panels: [meter, windows, months, value],
};
