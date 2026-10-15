// Ledger: where the dollars went over a period, and what each session bought
// (/api/ledger, /api/sessions/{id}). Panel order in the DOM is the phone
// reading order: summary, controls, chart, split, biggest sessions,
// timeline, breakdown. The desktop grid moves the timeline up (ledger.css).
import stats from '../panels/ledger-stats.js';
import ctl from '../panels/ledger-ctl.js';
import overview from '../panels/ledger-overview.js';
import split from '../panels/ledger-split.js';
import top from '../panels/ledger-top.js';
import timeline from '../panels/ledger-timeline.js';
import pivot from '../panels/ledger-pivot.js';
import drawer from '../panels/ledger-drawer.js';
import { periodParams } from '../panels/ledger-model.js';
import { ensureCss } from '../panels/ledger-kit.js';

ensureCss();

const POLL_MS = 120000;

export default {
  id: 'ledger',
  label: 'Ledger',
  feeds: {
    // 30 days (the server default): the chart, the legend and the model shift.
    ledger: { endpoint: 'ledger', intervalMs: POLL_MS, bump: true },
    // The period, clipped by the server. It waits for the 30-day read, so the
    // closed days are cached when it runs; until then the panels show the
    // 30-day sessions that touch the period and say so.
    ledgerPeriod: { endpoint: 'ledger', intervalMs: POLL_MS, bump: true, params: (s) => periodParams(s, s.mock) },
  },
  panels: [stats, ctl, overview, split, top, timeline, pivot, drawer],
};
