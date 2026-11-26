// Browser smoke lane for apps/new_england_north: real viewer.js, real
// MapLibre GL (pinned 3.6.2, served locally), fake deterministic API.
import { test, expect } from "./fixtures.mjs";
import { ADDRESS_FIXTURES, REGION, RUN_NAME, routeComparison } from "./fake-api.mjs";

const COMPARE_PATH = "/api/v1/route/compare";

async function gotoReady(page, url = "/") {
  await page.goto(url);
  // Plan route enables only after MapLibre fires `load` and region capabilities arrive.
  await expect(page.locator("#submitRoute")).toBeEnabled({ timeout: 30_000 });
}

async function waitForStartupRoute(page) {
  await expect(page.locator("#routeStatus")).toHaveText("New England North validated route");
  await page.waitForLoadState("networkidle");
}

function comparePosts(page) {
  const bodies = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && new URL(request.url()).pathname === COMPARE_PATH) {
      bodies.push(request.postDataJSON());
    }
  });
  return bodies;
}

async function submitCoordinates(page, start, end) {
  await page.locator("#startInput").fill(`${start.lat}, ${start.lon}`);
  await page.locator("#endInput").fill(`${end.lat}, ${end.lon}`);
  const response = page.waitForResponse((res) => new URL(res.url()).pathname === COMPARE_PATH);
  await page.locator("#submitRoute").click();
  return (await response).request().postDataJSON();
}

const km = (value) => `${value.toFixed(1)} km`;

/** Count scenic-route-green pixels in a screenshot of the real WebGL map. */
async function scenicGreenPixels(page) {
  const png = await page.locator("#map").screenshot();
  return page.evaluate(async (base64) => {
    const blob = await (await fetch(`data:image/png;base64,${base64}`)).blob();
    const bitmap = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d");
    context.drawImage(bitmap, 0, 0);
    const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
    let count = 0;
    for (let i = 0; i < data.length; i += 4) {
      const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
      if (g > 150 && g > r + 50 && g > b + 25) count += 1;
    }
    return count;
  }, png.toString("base64"));
}

async function readShareUrl(page) {
  const share = page.locator("#shareRouteBtn");
  await expect(share).toBeEnabled();
  await expect(share).toHaveText("Share");
  await share.click();
  await expect(share).toHaveText("Copied");
  return new URL(await page.evaluate(() => navigator.clipboard.readText()));
}

function expectShareMatches(url, request) {
  expect(url.searchParams.get("start")).toBe(`${request.start.lat},${request.start.lon}`);
  expect(url.searchParams.get("end")).toBe(`${request.end.lat},${request.end.lon}`);
  expect(url.searchParams.get("max_detour")).toBe(String(request.max_detour_factor));
  expect(url.searchParams.get("avoid_highways")).toBe(String(request.avoid_highways));
  expect(url.searchParams.get("source")).toBe(REGION);
  expect(url.searchParams.get("run")).toBe(RUN_NAME);
}

test.describe("desktop", () => {
  test("startup renders the real MapLibre map, heatmap and route form", async ({ page, tiles }) => {
    await gotoReady(page);
    await waitForStartupRoute(page);

    const canvas = page.locator("#map canvas.maplibregl-canvas");
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    expect(box.width).toBeGreaterThan(300);
    expect(box.height).toBeGreaterThan(300);
    await expect(page.locator(".maplibregl-ctrl-zoom-in")).toBeVisible();
    await expect(page.locator(".maplibregl-ctrl-attrib")).toContainText("CARTO");
    expect(tiles.served).toBeGreaterThan(0);

    await expect(page.locator("#apiStatus")).toHaveText("API: online");
    await expect(page.locator("#regionStatus")).toHaveText(`New England North · ${RUN_NAME}`);
    await expect(page.locator("#trainingResults")).toContainText(RUN_NAME);
    await expect(page.locator("#inspectorScore")).toHaveText(/tiles$/);
    await expect(page.locator("#startInput")).toHaveValue("44.475884, -73.214003");
    await expect(page.locator("#endInput")).toHaveValue("44.801616, -68.771305");
    await expect(page.locator("#detourOut")).toHaveText("1.80×");
    await expect(page.locator("#avoidHighways")).toBeChecked();
    await expect(page.locator("#routeResultsDialog")).not.toHaveAttribute("open");
    // The validated startup route is drawn by WebGL on the real map.
    await expect.poll(() => scenicGreenPixels(page)).toBeGreaterThan(50);
  });

  test("coordinate submit opens the route comparison dialog", async ({ page }) => {
    await gotoReady(page);
    await waitForStartupRoute(page);
    const start = { lat: 44.26, lon: -72.58 };
    const end = { lat: 44.55, lon: -69.65 };
    const body = await submitCoordinates(page, start, end);

    expect(body).toEqual({
      region: REGION,
      run_name: RUN_NAME,
      start,
      end,
      max_detour_factor: 1.8,
      avoid_highways: true,
      include_baseline: true,
    });
    const expected = routeComparison(body);
    const dialog = page.locator("#routeResultsDialog");
    await expect(dialog).toBeVisible();
    await expect(page.locator("#scenicDistance")).toHaveText(km(expected.routes.scenic.total_distance_km));
    await expect(page.locator("#baselineDistance")).toHaveText(km(expected.routes.baseline.total_distance_km));
    await expect(page.locator("#scenicScore")).toHaveText("7.42 / 10");
    await expect(page.locator("#baselineScore")).toHaveText("6.18 / 10");
    await expect(page.locator("#scenicScoreDelta")).toHaveText("+1.24");
    await expect(page.locator("#closeRouteDialogBtn")).toBeFocused();
    await expect(page.locator("#routeStatus")).toHaveText("Route ready. Scenic and baseline routes are shown on the map.");
    await expect(page.locator("#routeOutput")).toContainText("Dashed lines connect selected points to the road network.");

    await page.locator("#closeRouteDialogBtn").click();
    await expect(dialog).toBeHidden();
    await expect.poll(() => scenicGreenPixels(page)).toBeGreaterThan(50);
  });

  test("deferred render keeps Share disabled until the NEW route is displayed", async ({ page, tiles }) => {
    await gotoReady(page);
    await waitForStartupRoute(page);
    const share = page.locator("#shareRouteBtn");
    const dialog = page.locator("#routeResultsDialog");

    // A prior route renders and becomes shareable normally.
    const prior = await submitCoordinates(page, { lat: 44.1, lon: -72.9 }, { lat: 44.3, lon: -71.4 });
    await expect(dialog).toBeVisible();
    await expect(share).toBeEnabled();
    await page.locator("#closeRouteDialogBtn").click();
    await expect(dialog).toBeHidden();
    await page.waitForLoadState("networkidle");

    // Hold basemap tile responses, then zoom the real map: MapLibre requests
    // new raster tiles, so map.isStyleLoaded() stays false until release.
    tiles.hold();
    await page.locator("#map canvas.maplibregl-canvas").focus();
    await page.keyboard.press("Equal");
    await expect.poll(() => tiles.held.length).toBeGreaterThan(0);

    const next = await submitCoordinates(page, { lat: 45.2, lon: -70.6 }, { lat: 44.9, lon: -68.4 });
    expect(next.start).not.toEqual(prior.start);
    await expect(page.locator("#routeStatus")).toHaveText("Route computed. Waiting for the map to finish loading…");
    await expect(share).toBeDisabled();
    await expect(share).toHaveText("Share pending…");
    await expect(dialog).toBeHidden();
    // Still deferred while the style keeps loading.
    await page.waitForTimeout(750);
    await expect(share).toBeDisabled();
    await expect(dialog).toBeHidden();
    expect(tiles.held.length).toBeGreaterThan(0);

    expect(await tiles.release()).toBeGreaterThan(0);
    await expect(dialog).toBeVisible();
    await expect(page.locator("#routeStatus")).toHaveText("Route ready. Scenic and baseline routes are shown on the map.");
    await expect(page.locator("#scenicDistance")).toHaveText(km(routeComparison(next).routes.scenic.total_distance_km));

    const shared = await readShareUrl(page);
    expectShareMatches(shared, next);
    expect(shared.searchParams.get("start")).not.toBe(`${prior.start.lat},${prior.start.lon}`);
    expect(shared.searchParams.get("end")).not.toBe(`${prior.end.lat},${prior.end.lon}`);
  });

  test("shared URL replays and restores the shared request", async ({ page }) => {
    const requested = [];
    page.on("request", (request) => requested.push(new URL(request.url()).pathname));
    const bodies = comparePosts(page);
    const shared = { start: { lat: 44.31, lon: -71.05 }, end: { lat: 45.12, lon: -69.33 } };
    const query = new URLSearchParams({
      start: `${shared.start.lat},${shared.start.lon}`,
      end: `${shared.end.lat},${shared.end.lon}`,
      max_detour: "2.3",
      avoid_highways: "false",
      source: REGION,
      run: RUN_NAME,
    });
    await page.goto(`/?${query}`);

    await expect(page.locator("#routeResultsDialog")).toBeVisible({ timeout: 30_000 });
    expect(bodies).toEqual([
      { region: REGION, run_name: RUN_NAME, ...shared, max_detour_factor: 2.3, avoid_highways: false, include_baseline: true },
    ]);
    expect(requested).not.toContain("/api/v1/validated-route");
    await expect(page.locator("#startInput")).toHaveValue("44.31,-71.05");
    await expect(page.locator("#endInput")).toHaveValue("45.12,-69.33");
    await expect(page.locator("#detourOut")).toHaveText("2.30×");
    await expect(page.locator("#avoidHighways")).not.toBeChecked();
    await expect(page.locator("#scenicDistance")).toHaveText(km(routeComparison(bodies[0]).routes.scenic.total_distance_km));

    // Sharing the replayed route round-trips the same request.
    expectShareMatches(await readShareUrl(page), bodies[0]);
  });

  test("invalid input shows public errors without leaking server reasons", async ({ page }) => {
    await gotoReady(page);
    await waitForStartupRoute(page);
    const bodies = comparePosts(page);
    const status = page.locator("#routeStatus");
    const startInput = page.locator("#startInput");

    // Client-side validation: unparseable start, no address selected.
    await startInput.fill("not a coordinate");
    await page.locator("#submitRoute").click();
    await expect(status).toHaveText("Enter valid coordinates as lat, lon or choose an address suggestion.");
    await expect(startInput).toHaveAttribute("aria-invalid", "true");
    await expect(page.locator("#endInput")).toHaveAttribute("aria-invalid", "false");
    await expect(startInput).toBeFocused();
    await expect(page.locator("#shareRouteBtn")).toBeDisabled();
    expect(bodies).toEqual([]);

    // Server-side public error: start outside route coverage.
    await submitCoordinates(page, { lat: 40.7128, lon: -74.006 }, { lat: 44.8, lon: -68.77 });
    await expect(status).toHaveText(
      "Outside route coverage: The start point is too far from the supported road network. — Choose a point within the selected region's route coverage."
    );
    await expect(status).not.toContainText("snap distance");
    await expect(startInput).toHaveAttribute("aria-invalid", "true");
    await expect(page.locator("#endInput")).toHaveAttribute("aria-invalid", "false");
    await expect(page.locator("#apiStatus")).toHaveText("API: outside route coverage");
    await expect(page.locator("#routeResultsDialog")).toBeHidden();
    await expect(page.locator("#shareRouteBtn")).toBeDisabled();
    await expect(page.locator("#submitRoute")).toBeEnabled();
  });

  test("keyboard autocomplete selects a suggestion with ArrowDown and Enter", async ({ page }) => {
    await gotoReady(page);
    await waitForStartupRoute(page);
    const endInput = page.locator("#endInput");
    const options = page.locator("#endSuggestions .address-suggestion");

    await endInput.fill("");
    await endInput.pressSequentially("Port");
    await expect(options).toHaveCount(3);
    await expect(endInput).toHaveAttribute("aria-expanded", "true");

    await endInput.press("ArrowDown");
    await endInput.press("ArrowDown");
    await expect(endInput).toHaveAttribute("aria-activedescendant", "end-suggestion-1");
    await expect(options.nth(1)).toHaveAttribute("aria-selected", "true");
    await expect(options.nth(0)).toHaveAttribute("aria-selected", "false");

    const portsmouth = ADDRESS_FIXTURES[1];
    const retrieve = page.waitForRequest((request) => new URL(request.url()).pathname === "/api/v1/search/retrieve");
    await endInput.press("Enter");
    expect(new URL((await retrieve).url()).searchParams.get("mapbox_id")).toBe(portsmouth.mapbox_id);
    await expect(endInput).toHaveValue(portsmouth.full_address);
    await expect(endInput).toHaveAttribute("aria-expanded", "false");
    await expect(page.locator("#endSuggestions .address-suggestion")).toHaveCount(0);

    // Enter with no open suggestion submits the form using the retrieved point.
    const compare = page.waitForRequest((request) => new URL(request.url()).pathname === COMPARE_PATH);
    await endInput.press("Enter");
    expect((await compare).postDataJSON().end).toEqual({ lat: portsmouth.lat, lon: portsmouth.lon });
    await expect(page.locator("#routeResultsDialog")).toBeVisible();
  });
});

test.describe("mobile", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });

  test("mobile-width load fits the map and route controls", async ({ page }) => {
    await gotoReady(page);
    await waitForStartupRoute(page);

    const canvas = page.locator("#map canvas.maplibregl-canvas");
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    expect(Math.round(box.width)).toBe(390);
    expect(box.height).toBeGreaterThan(200);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(0);
    for (const id of ["#startInput", "#endInput", "#submitRoute"]) {
      await expect(page.locator(id)).toBeInViewport();
    }

    await submitCoordinates(page, { lat: 44.26, lon: -72.58 }, { lat: 44.55, lon: -69.65 });
    const dialog = page.locator("#routeResultsDialog");
    await expect(dialog).toBeVisible();
    const dialogBox = await dialog.boundingBox();
    expect(dialogBox.x).toBeGreaterThanOrEqual(0);
    expect(dialogBox.x + dialogBox.width).toBeLessThanOrEqual(390);
    await expect(page.locator("#shareRouteBtn")).toBeInViewport();
  });
});
