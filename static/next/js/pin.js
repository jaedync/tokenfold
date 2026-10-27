// The window the owner pinned as the hero. This browser keeps it
// (localStorage); without a pin, or when the pinned window is not in the
// runway answer, the server's binding window is the hero (select.heroId).

const KEY = 'tf_hero_window';
// Same shape as the server's window ids (app/runway.py _WINDOW_ID_RE).
const ID_RE = /^(claude|codex|opencode-go):[a-z0-9_:-]{1,64}$/;

export function loadPin(storage = globalThis.localStorage) {
  try {
    const id = storage.getItem(KEY);
    return id && ID_RE.test(id) ? id : null;
  } catch {
    return null; // storage off: no pin
  }
}

// setPin(store, id) pins a window; setPin(store, null) goes back to auto.
export function setPin(store, id, storage = globalThis.localStorage) {
  const pin = id && ID_RE.test(id) ? id : null;
  try {
    if (pin) storage.setItem(KEY, pin); else storage.removeItem(KEY);
  } catch { /* storage off: the pin lasts until a reload */ }
  store.set('pin', pin);
}
