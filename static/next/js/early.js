// Early reads: the shell's first feeds start as soon as this module and
// api.js arrive, not after the whole module graph (about 40 modules) loads.
// getJSON (api.js) takes each response once. Only reads whose params need
// no other feed are listed here. They must match the view feeds exactly:
// app/tests/next_stable.test.cjs checks that no read happens twice at boot.
import { startEarly } from './api.js';

const personal = (scope) => scope !== 'enterprise';
const VIEW_READS = {
  runway: (scope) => [['today', {}], personal(scope) ? ['limitRate', {}] : null],
  ledger: () => [['ledger', {}]],
  history: (scope) => (personal(scope)
    ? [['windowHistory', { bucket: 'seven_day' }], ['spendHistory', {}]]
    : [['spendHistory', {}], ['stats', {}], ['enterpriseBudget', {}]]),
  lab: (scope) => [['stats', {}], personal(scope) ? ['servedTimeline', { days: '30', sessions: 0 }] : null],
};

const body = document.body;
const scope = body.dataset.scope || 'personal';
const mock = new URLSearchParams(location.search).get('mock') === '1';
const hash = location.hash.slice(1);
const known = (v) => Object.hasOwn(VIEW_READS, v);
const view = known(hash) ? hash : (body.dataset.view || 'runway');
const reads = [['now', {}], ['runway', {}], ...(known(view) ? VIEW_READS[view](scope) : [])];
for (const read of reads) if (read) startEarly(read[0], read[1], { scope, mock });
