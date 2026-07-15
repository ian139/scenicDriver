import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";

const viewerSource = fs.readFileSync(
  new URL("../apps/new_england_north/viewer.js", import.meta.url),
  "utf8"
);
const installStart = viewerSource.indexOf("function installAddressSearch");
const installEnd = viewerSource.indexOf("\n\nasync function planRoute", installStart);
assert.ok(installStart >= 0 && installEnd > installStart);
const installSource = viewerSource.slice(installStart, installEnd);
const failureStart = viewerSource.indexOf("function routeFailurePresentation");
const failureEnd = viewerSource.indexOf("\n\nconst REASON_PROSE", failureStart);
assert.ok(failureStart >= 0 && failureEnd > failureStart);
const failureSource = viewerSource.slice(failureStart, failureEnd);

class FakeElement {
  constructor(tag = "div") {
    this.tagName = tag;
    this.value = "";
    this.children = [];
    this.listeners = new Map();
    this.attributes = new Map();
    this.classList = {
      add: () => {},
      remove: () => {},
    };
  }

  addEventListener(type, callback) {
    const callbacks = this.listeners.get(type) || [];
    callbacks.push(callback);
    this.listeners.set(type, callbacks);
  }

  dispatch(type, event = {}) {
    for (const callback of this.listeners.get(type) || []) {
      callback({ target: this, relatedTarget: null, preventDefault() {}, ...event });
    }
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  replaceChildren(...children) {
    this.children = children;
  }

  append(...children) {
    this.children.push(...children);
  }

  appendChild(child) {
    this.children.push(child);
  }
  contains() {
    return false;
  }

  focus() {}
}

function makeHarness() {
  const input = new FakeElement("input");
  const suggestions = new FakeElement("div");
  const clear = new FakeElement("button");
  const selectedRoutePoints = { start: null, end: null };
  const pending = [];
  let sequence = 0;
  const context = {
    CONFIG: { sourceRegion: "test-region" },
    selectedRoutePoints,
    crypto: { randomUUID: () => `session-${++sequence}` },
    AbortController,
    URL: class TestURL extends URL {
      constructor(path, base) {
        super(path, base || "http://test.local");
      }
    },
    setTimeout,
    clearTimeout,
    Number,
    JSON,
    String,
    Math,
    console,
    document: {
      createElement: (tag) => new FakeElement(tag),
    },
    api: (path) => path,
    parsePoint: (value) => {
      const match = String(value).match(/^(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)$/);
      return match ? { lat: Number(match[1]), lon: Number(match[2]) } : null;
    },
    fetch: (url, options) => {
      const record = { url: String(url), options };
      pending.push(record);
      return new Promise((resolve, reject) => {
        record.resolve = resolve;
        record.reject = reject;
        options.signal.addEventListener("abort", () => {
          const error = new Error("aborted");
          error.name = "AbortError";
          reject(error);
        });
      });
    },
  };
  vm.runInNewContext(
    `function newSearchSessionToken() { return "session-fixed"; }\n${installSource}\nglobalThis.installAddressSearch = installAddressSearch;`,
    context
  );
  context.el = {
    startInput: input,
    startSuggestions: suggestions,
    clearStartBtn: clear,
    endInput: new FakeElement("input"),
    endSuggestions: new FakeElement("div"),
    clearEndBtn: new FakeElement("button"),
  };
  context.installAddressSearch("start");
  return { context, input, suggestions, clear, pending, selectedRoutePoints };
}

const response = (payload) => ({ ok: true, json: async () => payload });
const tick = (ms = 230) => new Promise((resolve) => setTimeout(resolve, ms));

test("route failure presentation distinguishes no-route constraints", () => {
  const context = {};
  vm.runInNewContext(
    `${failureSource}\nglobalThis.routeFailurePresentation = routeFailurePresentation;`,
    context
  );
  const noRoute = context.routeFailurePresentation("no_route_found");
  assert.equal(noRoute.status, "API: no route found");
  assert.equal(noRoute.title, "No route found");
  const other = context.routeFailurePresentation("server_error");
  assert.equal(other.status, "API: route failed");
  assert.equal(other.title, "Route failed");
});

test("address suggest/retrieve guards reject stale responses and close invalidates", async () => {
  const harness = makeHarness();
  const { input, suggestions, pending, selectedRoutePoints } = harness;

  input.value = "old query";
  input.dispatch("input");
  await tick();
  assert.equal(pending.length, 1);
  input.value = "new query";
  input.dispatch("input");
  await tick();
  assert.equal(pending.length, 2);

  pending[0].resolve(response({ suggestions: [{ mapbox_id: "old", name: "Old" }] }));
  pending[1].resolve(
    response({
      suggestions: [
        { mapbox_id: "first", name: "First" },
        { mapbox_id: "second", name: "Second" },
      ],
    })
  );
  await tick(20);
  assert.equal(suggestions.children.length, 2);
  assert.equal(suggestions.children[0].children[0].textContent, "First");

  const firstSuggestion = suggestions.children[0];
  const secondSuggestion = suggestions.children[1];
  firstSuggestion.dispatch("mousedown");
  secondSuggestion.dispatch("mousedown");
  assert.equal(pending.length, 4);
  pending[2].resolve(
    response({ result: { lat: 1, lon: 1, full_address: "Old address" } })
  );
  pending[3].resolve(
    response({ result: { lat: 2, lon: 2, full_address: "New address" } })
  );
  await tick(0);
  assert.equal(selectedRoutePoints.start.lat, 2);
  assert.equal(selectedRoutePoints.start.lon, 2);
  assert.equal(input.value, "New address");

  // A delayed suggest response after Escape must not reopen the list.
  input.value = "delayed query";
  input.dispatch("input");
  await tick();
  const delayed = pending.at(-1);
  input.dispatch("keydown", { key: "Escape" });
  delayed.resolve(response({ suggestions: [{ mapbox_id: "late", name: "Late" }] }));
  await tick(0);
  assert.equal(suggestions.children.length, 0);
});
