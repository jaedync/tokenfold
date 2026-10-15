// Model family colours and provider names. Colours come from a fixed list of
// CSS variables keyed by family, never from a server string.

const FAMILY_VAR = {
  opus: '--m-opus', fable: '--m-fable', sonnet: '--m-sonnet', haiku: '--m-haiku',
  deepseek: '--m-deepseek', glm: '--m-glm', gpt: '--m-gpt',
};

// Display order of families in mixes and legends.
export const FAMILY_ORDER = ['opus', 'fable', 'sonnet', 'haiku', 'deepseek', 'glm', 'gpt'];

export function familyColor(family) {
  const key = String(family || '').toLowerCase();
  return `var(${FAMILY_VAR[key] || '--m-other'})`;
}

// Best guess at a family from a model display name, for payloads without one.
export function familyOf(model) {
  const m = String(model || '').toLowerCase();
  return FAMILY_ORDER.find((fam) => m.includes(fam)) || 'other';
}

export const FAMILY_NAME = {
  opus: 'Opus', fable: 'Fable', sonnet: 'Sonnet', haiku: 'Haiku', deepseek: 'DeepSeek', glm: 'GLM', gpt: 'GPT',
};

const PROVIDER_NAME = { claude: 'Claude', 'opencode-go': 'OpenCode Go', codex: 'Codex' };
export const providerName = (p) => PROVIDER_NAME[p] || String(p || '');

// "Claude · 7-day" -> {prov: "Claude", name: "7-day"}
export function splitLabel(w) {
  const label = String(w.label || w.key || '');
  const i = label.indexOf(' · ');
  return i >= 0 ? { prov: label.slice(0, i), name: label.slice(i + 3) } : { prov: providerName(w.provider), name: label };
}

// Short window tag for the tab title and live bar: 7d, 5h, budget.
export function windowTag(w) {
  if (!w) return '';
  if (w.kind === 'month') return 'budget';
  return { seven_day: '7d', five_hour: '5h' }[w.key] || splitLabel(w).name.toLowerCase();
}
