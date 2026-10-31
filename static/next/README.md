# Tokenfold `/next` front end

The dashboard at `/` (`/next` is an alias; the old one is at `/classic`). Plain ES modules, no build step, no CDN. The
contract for the endpoints is `docs/ux-2026-10/BUILD.md`. The visual spec is the
approved Runway mockup (`.ux/mockups/runway/index.html`).

## Files

| Path | Purpose |
|------|---------|
| `templates/next.html` | Jinja shell: masthead, live bar, `<main id="view">`, footer. Sets `data-scope`, `data-scope-locked`, `data-owner`, `data-view`, `data-version`, `data-static` on `<body>`. |
| `app/next_page.py` | `GET /` and `GET /next`: dashboard auth, scope precedence as `/classic` (lock > `?scope=` > `tf_scope` cookie > default). Also serves `static/next` under `/static/next-<version>/` (see Cache busting), and lists every module of the static import graph as a `modulepreload` link (`module_graph`). |
| `css/tokenfold.css` | Tokens, atoms, layout, then one section per panel. |
| `js/main.js` | Boot, hash router (`#runway`, `#ledger`, `#history`, `#lab`), scope switch, tab title, SSE bump, idle prefetch of the other views' feeds. |
| `js/early.js` | Starts the shell feeds and the first view's feeds before the rest of the modules load. Its URLs must match the view feeds (the boot checks in `app/tests/next_stable.test.cjs`). |
| `js/api.js` | `getJSON(name, params, {scope, mock, signal})`, the endpoint table `ENDPOINTS`, `backoffMs()`. |
| `js/feeds.js` | Shared polled feeds with backoff; pauses in a hidden tab. A refetch keeps the old data on screen. A scope switch sets `switching` and keeps the old content until the new data is in. `prefetch()` reads a feed once. A body the server marks stale (`X-Tokenfold-Stale: 1`, built before the newest ingest) is read again after 4 s, once per bump. |
| `js/scroll.js` | Window scroll position per history entry: back, forward and a reload put it back once the view's data is in; a view tab opens at the top. The browser's own restoration is off (it runs while the page is one screen of "Loading" lines). |
| `js/reserve.js` | Reserved panel heights while a view loads: each panel keeps the height it had at the last load (`localStorage`, per view, scope and layout width) as `--reserve`; the CSS applies it while the panel holds a `[data-loading]` line, else the panel's `--reserve-default`. |
| `js/store.js` | Immutable store: `get()`, `set(key, value)`, `subscribe(fn)`. |
| `js/format.js` | `usd`, `pct`, `dur`, `shortDur`, `age`, `clock`, `dayClock`, `when`, `esc`. Missing values print `—`. |
| `js/dom.js` | `h()`, `fill()`, `patch()`, `setText()`, `approx()`, `segmented()`, `panelHead()`. |
| `js/panel.js` | `mountPanel()`, `feedLine()`, `feedStatus()`, `staleNote()`, `stateLine()`. |
| `js/select.js` | Pure selectors: `heroId`, `bindingWindow`, `allWindows`, `windowsByProvider`, `sortSessions`, `nowFrom`. The hero is the pinned window when the runway has it, else the server's `binding`. `allWindows` has a fixed order (provider by `PROVIDER_ORDER` in `palette.js`, then short, long, month, then window length, then label), so the All windows rows do not move between polls. |
| `js/pin.js` | The hero window the owner pinned (a click on a long window's row in All windows; Back to auto clears it). This browser keeps it in `localStorage` (`tf_hero_window`); the store key is `pin`. |
| `js/palette.js` | Model family colours (CSS variables only), provider names. |
| `js/views/*.js` | One module per view: the feeds it wants and its panels. |
| `js/panels/*.js` | One module per panel. |
| `fixtures/*.json` | One fixture per endpoint (`-work` variants for the work scope). Made by `scripts/make_next_fixtures.py`. |
| `vendor/uplot/` | uPlot 1.6.32, MIT (`LICENSE`). ESM build. |
| `fonts/` | Archivo Black, DM Sans (OFL text files beside them). |

## Cache busting

The shell loads every file of `static/next` from `/static/next-<version>/…`,
where `<version>` is a content hash of the folder (`app/next_page.py`). Relative
imports inherit the prefix, so a nested module is never stale after a deploy,
and the current version is served `immutable`. A URL with an older version
still loads (the current files, `no-cache`).

- In a module, build asset URLs relative to the module:
  `new URL('../../fixtures/now.json', import.meta.url)`.
- Link every view stylesheet in the template, so no view restyles after the
  first paint. Use `{{ static_base }}/css/<view>.css`. Do not write
  `/static/next/…` or `?v=`: the plain path still works but skips the version.

## Mock mode

Open `/next?mock=1`. Every feed then reads `fixtures/<name>.json` (or
`<name>-work.json` in the work scope) instead of the API. The page clock stops at
the fixtures' `generated_at`, so relative times ("in 12 h") match the fixtures.
Change the world in `scripts/make_next_fixtures.py`, then run it:

```sh
.venv/bin/python scripts/make_next_fixtures.py
```

## Panel API

A panel is a module with a default export:

```js
export default {
  id: 'today',            // data-panel="today" and class p-today on its <section>
  label: 'Today',         // aria-label and the failure message
  cls: 'p-wide',          // optional extra classes (p-wide spans the full grid)
  mount(el, ctx) {        // el: the panel's empty <section>
    // Build the DOM once. Return the instance:
    return {
      update(state) {},   // called after every store change, with the frozen state
      destroy() {},       // optional: stop timers, observers, charts
    };
  },
};
```

- `ctx.store`: the store. `ctx.store.get().scope` is `personal` or `enterprise`.
- `ctx.now()`: server time in epoch seconds. Use it for every relative time.
- `ctx.feeds`, `ctx.api`, `ctx.mock`: data access (views normally declare feeds).
- `ctx.announce(text)`: a polite screen-reader message.
- `ctx.navigate(viewId, panelId)`: go to a view and focus a panel.

`mountPanel()` wraps `mount` and `update` in try/catch. A panel that throws
shows one quiet line ("… failed to load"). The other panels keep working.

A panel with no data yet shows `feedLine()` (or `loadingLine()`), which carries
`data-loading`: the panel then keeps its reserved height (`reserve.js`). A panel
that shows nothing while it loads makes the page jump when the data comes in.
Give a new panel a `--reserve-default` near its usual height in its view's CSS.

### Updates in place

A poll must not reload what the user looks at. In `update`, build the new
content with `h()` and give it to `patch(el, ...)`, not `fill()`. `patch()`
keeps every node that can stay and changes only text and attributes, so
hover, focus, selection, scroll and open `<details>` stay. Give each row of a
list that can change order a `data-key` (a session id, a model name, a
month). A keyed row that moves keeps its focus, selection and inner scroll:
`patch()` moves only the rows out of order, with `moveBefore` where the
browser has it. Keep no reference to a node built for `patch()`: when an old node
matches, the new one is dropped. `app/tests/next_stable.test.cjs` checks
every view over three poll cycles: no layout shift, no replaced node, no
"Loading".

### Feed state

Each feed is a store key with `{status, data, error, at}`. `status` is
`loading`, `ok` or `error`. After a failure the last good `data` stays. Use the
helpers so every panel speaks the same way:

```js
const line = feedLine(state.today, "today's spend");   // null when there is data
patch(stateEl, line, staleNote(state.today, nowFrom(state)));
if (line) return;                                      // loading or unavailable
```

`feedLine` says "Loading …" or "… is unavailable." with a reason (404: the
server does not have this feed yet). Never let a missing feed break the page.

## Views

```js
export default {
  id: 'ledger',
  label: 'Ledger',
  feeds: {
    ledger: { endpoint: 'ledger', bump: true, params: (s) => ({ from: s.ledgerFrom }) },
  },
  panels: [ledgerTable, sessionDrawer],
};
```

- `feeds`: store key -> `{endpoint, intervalMs, bump, params(state)}`.
  `endpoint` is a key of `ENDPOINTS` in `api.js`. `intervalMs` polls; `bump`
  refetches on the SSE version bump (at most once per 10 s). `params(state)`
  returns the query (without `scope`, which the feed adds), or `null` to wait.
  When the params change, the feed fetches again.
- `panels`: mounted in this order into `<div class="grid">`. Put them in the
  phone reading order; place them on the desktop grid in CSS (`.p-<id>`).
- `main.js` releases the view's feeds and destroys its panels when the user
  leaves the view. The shell always keeps `now` (5 s) and `runway` (30 s).

To build a view, replace its placeholder panel in `js/views/<view>.js` with
real panels. To add a fixture, add the endpoint to `ENDPOINTS` and to the
generator.

## Rules

- Forecasts come from the server only: `pace.*`, `tight_epoch`, `verdict`,
  `budget.*`. The browser draws lines between server values; it never makes a
  rate, a projection or a time to 100% of its own.
- Build DOM with `h()` and `setText()`. Never put a server string into
  `innerHTML`. Colours come from `palette.js`, never from a server string.
- No inline event handlers, no CDN, no new dependencies.
- Every estimate carries `≈` (`approx()`). Say once per view that dollars are
  API-equivalent at list price, not billed.
- Stale data greys out with its age (`.is-stale`, `staleNote`). Do not use
  opacity on text: it fails WCAG AA.
- No yellow text on paper (use `--ochre`). Small red text uses `--red-ink`.
- Copy follows ASD-STE100: short sentences, active voice, plain words.

## Tests

- `app/tests/test_next_page.py`: route, auth and scope.
- `app/tests/test_next_dom.py` + `next_dom.test.cjs`: Chromium checks in mock
  mode (regions, phone order, scope switch, real mode with 404 feeds, view
  tabs, the waiting announcement).
