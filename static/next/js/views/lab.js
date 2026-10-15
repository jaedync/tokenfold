// Lab: the checks and reference tables that are not part of the daily loop.
// Placeholder until the Lab build replaces `panels` (see README.md).
import { placeholder } from '../panels/placeholder.js';

export default {
  id: 'lab',
  label: 'Lab',
  feeds: {},
  panels: [placeholder({
    id: 'lab-placeholder',
    title: 'Lab',
    lead: 'The Lab holds the checks and tables that you do not need every day.',
    items: ['Served-model check', 'Energy and water estimates', 'Price table', 'Benchmarks'],
  })],
};
