// Tiny DOM helpers. Text goes through textContent and attributes through
// setAttribute, never innerHTML: machine, project and session names are
// untrusted strings from many hosts.

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') {
      for (const [prop, val] of Object.entries(v)) el.style.setProperty(prop, val);
    } else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, String(v));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

// Replace every child of el.
export function fill(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

// Writes only on change, so a 5 s poll does not restyle text that stayed the same.
export function setText(el, text) {
  const s = text === null || text === undefined ? '' : String(text);
  if (el.textContent !== s) el.textContent = s;
}

export function setAttr(el, name, value) {
  if (value === null || value === undefined || value === false) {
    if (el.hasAttribute(name)) el.removeAttribute(name);
    return;
  }
  const s = value === true ? '' : String(value);
  if (el.getAttribute(name) !== s) el.setAttribute(name, s);
}

// The "≈" mark for every estimate.
export const approx = () => h('span', { class: 'approx', title: 'Estimate' }, '≈');

// Segmented button group. aria-pressed carries the state for screen readers.
export function segmented(label, options, current, onPick) {
  const group = h('div', { class: 'seg', role: 'group', 'aria-label': label });
  const buttons = options.map(([value, text]) => h('button', {
    type: 'button', 'data-value': value, 'aria-pressed': String(value === current),
    onclick: () => { select(value); onPick(value); },
  }, text));
  function select(value) {
    for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.value === value));
  }
  group.append(...buttons);
  return { el: group, select };
}

let ids = 0;
export const uid = (prefix) => `${prefix}-${++ids}`;

// Standard panel heading: red square, display title, optional sub line and extras.
export function panelHead(title, ...extra) {
  const id = uid('pt');
  const titleEl = h('h2', { class: 'panel-title', id }, title);
  return { el: h('div', { class: 'panel-head' }, titleEl, ...extra), id, titleEl };
}
