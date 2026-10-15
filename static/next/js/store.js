// Tiny immutable store. State objects are frozen and replaced on every set, so
// a panel that holds an old reference still sees one consistent picture.
// Subscribers run once per microtask, however many keys changed in between.

const deepFreeze = (v) => {
  if (v && typeof v === 'object' && !Object.isFrozen(v)) {
    Object.freeze(v);
    for (const k of Object.keys(v)) deepFreeze(v[k]);
  }
  return v;
};

export function createStore(initial = {}) {
  let state = deepFreeze({ ...initial });
  const subs = new Set();
  let queued = false;

  function flush() {
    queued = false;
    const snapshot = state;
    for (const fn of subs) {
      try {
        fn(snapshot);
      } catch (err) {
        console.error('tokenfold: a store subscriber failed', err);
      }
    }
  }

  const get = () => state;

  // set(key, value) -> the new state. Values are frozen, never copied back.
  function set(key, value) {
    if (state[key] === value) return state;
    state = Object.freeze({ ...state, [key]: deepFreeze(value) });
    if (!queued) {
      queued = true;
      queueMicrotask(flush);
    }
    return state;
  }

  function subscribe(fn) {
    subs.add(fn);
    return () => subs.delete(fn);
  }

  return { get, set, subscribe };
}
