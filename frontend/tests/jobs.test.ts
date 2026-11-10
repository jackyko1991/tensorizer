// Detached jobs in the browser: the visitor id and the reconnecting event reader. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { JobGone, readJobEvents, visitorId, type JobEvent } from "../src/jobs.ts";

function memory() {
  const data = new Map<string, string>();
  return { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v), data };
}

const ndjson = (events: object[], close = true) =>
  new Response(
    new ReadableStream({
      start(c) {
        for (const e of events) c.enqueue(new TextEncoder().encode(JSON.stringify(e) + "\n"));
        if (close) c.close();
      },
      pull(c) {
        c.error(new TypeError("network error")); // after the events: the connection drops mid-stream
      },
    }),
  );

const noSleep = async () => {};

test("visitor id is made once and kept", () => {
  const store = memory();
  const id = visitorId(store);
  assert.match(id, /^[0-9a-f]{32}$/);
  assert.equal(visitorId(store), id);
  assert.equal(store.data.get("tensorizer.visitor"), id);
});

test("visitor id survives blocked storage", () => {
  const blocked = {
    getItem: () => {
      throw new Error("SecurityError");
    },
    setItem: () => {
      throw new Error("SecurityError");
    },
  };
  const id = visitorId(blocked);
  assert.match(id, /^[0-9a-f]{32}$/);
  assert.equal(visitorId(blocked), id); // the same id for the page's lifetime
});

test("reader reconnects from the last event after a drop, without repeats", async () => {
  const asked: number[] = [];
  const seen: JobEvent[] = [];
  const replies = [
    () => ndjson([{ type: "line", text: "a", seq: 0 }, { type: "line", text: "b", seq: 1 }], false),
    () => {
      throw new TypeError("Failed to fetch"); // the backend is restarting
    },
    // the server replays from 1 although 2 was asked: the reader skips what it already has
    () => ndjson([{ type: "line", text: "b", seq: 1 }, { type: "line", text: "c", seq: 2 }, { type: "result", seq: 3 }]),
  ];
  const final = await readJobEvents({
    open: async (from) => (asked.push(from), replies[asked.length - 1]()),
    onEvent: (e) => seen.push(e),
    sleep: noSleep,
  });
  assert.equal(final.type, "result");
  assert.deepEqual(asked, [0, 2, 2]);
  assert.deepEqual(
    seen.map((e) => e.text ?? e.type),
    ["a", "b", "c", "result"],
  );
});

test("reader starts from N and stops at an error event", async () => {
  let from = -1;
  const final = await readJobEvents({
    open: async (n) => ((from = n), ndjson([{ type: "error", message: "cancelled", seq: 5 }])),
    onEvent: () => {},
    from: 5,
    sleep: noSleep,
  });
  assert.equal(from, 5);
  assert.equal(final.message, "cancelled");
});

test("reader gives up on 404 at once and after repeated failures", async () => {
  await assert.rejects(
    readJobEvents({ open: async () => new Response("", { status: 404 }), onEvent: () => {}, sleep: noSleep }),
    JobGone,
  );
  let tries = 0;
  await assert.rejects(
    readJobEvents({
      open: async () => {
        tries++;
        throw new TypeError("Failed to fetch");
      },
      onEvent: () => {},
      retries: 3,
      sleep: noSleep,
    }),
    /Failed to fetch/,
  );
  assert.equal(tries, 4);
});

test("each backend gets its own visitor id, starting from the old single one", () => {
  const store = memory();
  const legacy = visitorId(store); // before ids were per backend
  assert.equal(visitorId(store, "local"), legacy); // runs made earlier stay listed
  assert.equal(visitorId(store, "remote:a"), legacy);
  store.data.set("tensorizer.visitor.remote:b", "b".repeat(32));
  assert.equal(visitorId(store, "remote:b"), "b".repeat(32));
  const fresh = memory();
  assert.notEqual(visitorId(fresh, "remote:x"), visitorId(memory(), "remote:x")); // no old id: a new random one
  assert.ok(fresh.data.has("tensorizer.visitor.remote:x") && !fresh.data.has("tensorizer.visitor"));
});

test("an agent turn's log ends when its stream closes and the server says the turn is over", async () => {
  const seen: JobEvent[] = [];
  let opens = 0;
  let over = false;
  await readJobEvents({
    open: async (from) => {
      opens += 1;
      if (opens === 1) return ndjson([{ type: "turn", seq: 0 }, { type: "error", message: "bad notebook", seq: 1 }]);
      over = true;
      return ndjson([{ type: "text", delta: "ok", seq: from }, { type: "done", seq: from + 1 }]);
    },
    onEvent: (e) => seen.push(e),
    isFinal: () => false, // an error mid-turn does not end it
    finished: async () => over,
    sleep: noSleep,
  });
  assert.deepEqual(
    seen.map((e) => e.type),
    ["turn", "error", "text", "done"],
  );
  assert.equal(opens, 2);
});
