// Ledger: sessions and projects over a period (/api/ledger, /api/sessions/{id}).
// Placeholder until the Ledger build replaces `panels` (see README.md).
import { placeholder } from '../panels/placeholder.js';

export default {
  id: 'ledger',
  label: 'Ledger',
  feeds: {},
  panels: [placeholder({
    id: 'ledger-placeholder',
    title: 'Ledger',
    lead: 'The Ledger shows where the spend went: one row per session, with its subagent runs, cost and active time.',
    items: ['Sessions and subagent runs over a period', 'Pivots by project, model, machine, tool and day', 'A session drawer with prompts, tools and tokens'],
  })],
};
