// A clean placeholder for a view that is not built yet. Build the real view
// from panels like the Runway ones (README.md) and remove this panel from it.
import { h, panelHead } from '../dom.js';

// placeholder({id, title, lead, items}) -> a panel module.
export function placeholder({ id, title, lead, items = [] }) {
  return {
    id,
    label: title,
    cls: 'p-wide',
    mount(el, ctx) {
      const head = panelHead(title);
      el.setAttribute('aria-labelledby', head.id);
      const scopeLink = h('a', { class: 'btn', href: `/?scope=${encodeURIComponent(ctx.store.get().scope)}` },
        'Open the classic dashboard ', h('span', { 'aria-hidden': 'true' }, '→'));
      el.append(head.el, h('div', { class: 'placeholder' },
        h('p', null, lead),
        items.length ? h('ul', null, items.map((t) => h('li', null, t))) : null,
        h('p', { class: 'panel-sub' }, 'This view is not ready yet. The classic dashboard has these figures until then.'),
        scopeLink));
      return {
        update(s) { scopeLink.setAttribute('href', `/?scope=${encodeURIComponent(s.scope)}`); },
      };
    },
  };
}
