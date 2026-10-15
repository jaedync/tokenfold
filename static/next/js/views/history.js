// History: monthly spend and past limit windows (/api/window-history).
// Placeholder until the History build replaces `panels` (see README.md).
import { placeholder } from '../panels/placeholder.js';

export default {
  id: 'history',
  label: 'History',
  feeds: {},
  panels: [placeholder({
    id: 'history-placeholder',
    title: 'History',
    lead: 'History shows how past limit windows ended and how the spend changed month by month.',
    items: ['Peak of each 7-day window, and the windows that hit the limit', 'Granted resets and changes of the limit size', 'Spend per month and per window'],
  })],
};
