# Browser smoke lane

Deterministic Playwright smoke tests for the static viewer in
`apps/new_england_north`. They drive the real `viewer.js` and real MapLibre GL
(WebGL via SwiftShader) against a local fake API. No test touches the
external network.

## Run

```sh
cd tests/browser
npm ci                            # pinned @playwright/test 1.60.0 + maplibre-gl 3.6.2
npx playwright install chromium   # one-time browser download (Chromium build 1223)
npm test                          # same as: node run.mjs
```

`node tests/browser/run.mjs` also works from the repo root. Extra arguments go
to `playwright test`, for example `node run.mjs -g "deferred"` or
`node run.mjs --repeat-each=3`.

If `@playwright/test`/`maplibre-gl` are not installed, or Playwright's Chromium
build is missing, `run.mjs` prints `browser smoke lane: SKIPPED — <reason>`
and exits 0. Any other failure, including a Chromium launch error that is not
a missing install, fails the run.

Environment:

- `PLAYWRIGHT_BROWSERS_PATH`: standard Playwright browser cache override.
  Pointing it at an empty directory exercises the skip path.
- No API keys or services are needed. The fixture server binds `127.0.0.1` on
  an ephemeral port and lives only for the Playwright worker.

## How it stays offline

- `fake-api.mjs` serves `index.html`, `viewer.js` and `styles.css` from
  `apps/new_england_north`. It also serves the `/api/v1/*` endpoints viewer.js
  calls: regions, healthz, training results, heatmap metadata/image/score grid,
  validated route, search suggest/retrieve, and route compare. Payload shapes
  follow `src/scenicdriver/app_api/main.py`. Route payloads are computed only
  from the request body.
- `fixtures.mjs` routes every browser request. The unpkg MapLibre JS/CSS is
  served from the pinned `node_modules/maplibre-gl` package, CARTO basemap
  tiles get a local grey PNG, and any other off-origin request is aborted and
  fails the test. Uncaught page errors also fail the test.

## Scenarios (`smoke.spec.mjs`)

| Scenario | What it proves |
| --- | --- |
| startup | MapLibre canvas, controls and attribution render. Region, training and heatmap status populate. The default form is ready. The validated route appears as green WebGL pixels. |
| coordinate submit | The POST body matches the form. The comparison dialog shows the expected metrics and focus. The route is drawn on the map. |
| deferred-render Share | After a prior route succeeds, basemap tile responses are held and the real map is zoomed, so `map.isStyleLoaded()` returns false. A new request then stays in "Waiting for the map…" with Share disabled ("Share pending…"). Once the tiles are released, the new route renders and the copied share URL contains the new coordinates, not the prior ones. |
| shared URL replay | `?start=&end=&max_detour=&avoid_highways=` restores the inputs, replays exactly that request without loading the validated route, and shares the same URL again. |
| invalid input | Unparseable input gives a client-side error with `aria-invalid` set and no request sent. An out-of-coverage endpoint shows the public message and hint, but not the server `reason`. |
| keyboard autocomplete | Typing, then ArrowDown ×2 and Enter, retrieves the second suggestion. Enter then submits using the retrieved point. |
| mobile width | At 390×844 (touch, mobile), the map fills the width, there is no horizontal overflow, and the controls and dialog fit the viewport. |

Failure traces are written to `tests/browser/test-results/` (git-ignored).
