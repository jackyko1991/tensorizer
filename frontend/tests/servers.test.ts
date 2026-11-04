// The visitor's own servers: storage of the list and tokens, URL rules, and Test connection. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  activeServer,
  addServer,
  authHeaderFor,
  getToken,
  loadServers,
  mixedContent,
  normalizeUrl,
  removeServer,
  setActiveServerId,
  setToken,
  testConnection,
  updateServer,
  type Stores,
} from "../src/servers.ts";

function memory() {
  const data = new Map<string, string>();
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
    data,
  };
}
const stores = () => {
  const local = memory();
  const session = memory();
  return { local, session, stores: { local, session } as Stores };
};
const blocked = {
  getItem: () => {
    throw new Error("SecurityError");
  },
  setItem: () => {
    throw new Error("SecurityError");
  },
  removeItem: () => {
    throw new Error("SecurityError");
  },
};

test("normalizeUrl accepts http(s) runner addresses and rejects the rest", () => {
  assert.equal(normalizeUrl("https://runner.example.org/"), "https://runner.example.org");
  assert.equal(normalizeUrl("  runner.example.org "), "https://runner.example.org");
  assert.equal(normalizeUrl("localhost:8765"), "http://localhost:8765");
  assert.equal(normalizeUrl("https://host.example/sub/path/"), "https://host.example/sub/path");
  for (const bad of ["", "ftp://x.org", "javascript:alert(1)", "https://user:pw@x.org", "https://x.org/?token=1", "https://x.org/#a"])
    assert.equal(normalizeUrl(bad), null, bad);
});

test("mixed content: an https page can't call plain http, except this computer", () => {
  assert.equal(mixedContent("http://203.0.113.7:8765", "https:"), true);
  assert.equal(mixedContent("http://localhost:8765", "https:"), false);
  assert.equal(mixedContent("http://127.0.0.1:8765", "https:"), false);
  assert.equal(mixedContent("https://runner.example.org", "https:"), false);
  assert.equal(mixedContent("http://203.0.113.7:8765", "http:"), false);
});

test("the list lives in localStorage, the token in sessionStorage unless remembered", () => {
  const { local, session, stores: s } = stores();
  const a = addServer({ name: "VPS", url: "https://a.example", token: "tok-a" }, s)!;
  const b = addServer({ name: "", url: "https://b.example", token: "tok-b", remember: true }, s)!;
  assert.equal(b.name, "b.example");
  assert.deepEqual(
    loadServers(s).map((x) => x.url),
    ["https://a.example", "https://b.example"],
  );
  assert.ok(!local.data.get("tensorizer.servers")!.includes("tok-"), "tokens never go into the list");
  assert.equal(session.data.get(`tensorizer.server.token.${a.id}`), "tok-a");
  assert.equal(local.data.get(`tensorizer.server.token.${a.id}`), undefined);
  assert.equal(local.data.get(`tensorizer.server.token.${b.id}`), "tok-b");
  assert.equal(getToken(a.id, s), "tok-a");
  assert.equal(getToken(b.id, s), "tok-b");
});

test("remember on/off moves the token; a new URL drops the old token; remove clears everything", () => {
  const { local, session, stores: s } = stores();
  const a = addServer({ name: "A", url: "https://a.example", token: "tok" }, s)!;
  updateServer(a.id, { remember: true }, s);
  assert.equal(local.data.get(`tensorizer.server.token.${a.id}`), "tok");
  assert.equal(session.data.get(`tensorizer.server.token.${a.id}`), undefined);
  updateServer(a.id, { name: "Renamed" }, s); // token kept
  assert.equal(getToken(a.id, s), "tok");
  updateServer(a.id, { url: "https://evil.example" }, s); // the token was for the old address
  assert.equal(getToken(a.id, s), null);
  updateServer(a.id, { token: "new" }, s);
  setActiveServerId(a.id, s);
  assert.equal(activeServer(s)?.name, "Renamed");
  removeServer(a.id, s);
  assert.equal(loadServers(s).length, 0);
  assert.equal(getToken(a.id, s), null);
  assert.equal(activeServer(s), null);
});

test("blocked or corrupt storage never throws", () => {
  const s = { local: blocked, session: blocked } as Stores;
  assert.deepEqual(loadServers(s), []);
  assert.ok(addServer({ name: "x", url: "https://x.example", token: "t" }, s));
  setToken("id", "t", true, s);
  assert.equal(getToken("id", s), null);
  const { local, stores: ok } = stores();
  local.setItem("tensorizer.servers", "{not json");
  assert.deepEqual(loadServers(ok), []);
  assert.deepEqual(loadServers({ local: null, session: null }), []);
});

test("the token is sent only to its own server", () => {
  const server = { id: "1", name: "a", url: "https://a.example/runner", remember: false };
  assert.equal(authHeaderFor("https://a.example/runner/api/queue", server, "t"), "Bearer t");
  assert.equal(authHeaderFor("https://a.example/runner-other/api/queue", server, "t"), null);
  assert.equal(authHeaderFor("https://a.example.evil.org/runner/api/queue", server, "t"), null);
  assert.equal(authHeaderFor("https://a.example/runner/api/queue", server, null), null);
  assert.equal(authHeaderFor("https://a.example/runner/api/queue", null, "t"), null);
});

// --- Test connection with a fake fetch ------------------------------------------------------------------------------

type Handler = (url: string, init?: RequestInit) => Response | Promise<Response>;
const fakeFetch = (h: Handler) => (async (url: string, init?: RequestInit) => h(url, init)) as unknown as typeof fetch;
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const HEALTH = { status: "ok", runner: "0.1.0", devices: ["cpu"], auth: "token", shared: false, code_exec: true };

test("test connection: health, then an authenticated call with the bearer header", async () => {
  const seen: (string | null)[] = [];
  const f = fakeFetch((url, init) => {
    if (url.endsWith("/api/health")) return json(HEALTH);
    seen.push(new Headers(init?.headers).get("Authorization"));
    return json({ running: 1, waiting: 2 });
  });
  const r = await testConnection("https://a.example", "tok", { fetch: f, pageProtocol: "https:" });
  assert.deepEqual(r, { ok: true, health: HEALTH, queue: { running: 1, waiting: 2 } });
  assert.deepEqual(seen, ["Bearer tok"]);
});

test("test connection: clear reasons for each failure", async () => {
  const opts = (h: Handler) => ({ fetch: fakeFetch(h), pageProtocol: "https:" });
  const health = (status: number) => (url: string) => (url.endsWith("/api/health") ? json(HEALTH) : json({}, status));
  assert.equal(((await testConnection("http://203.0.113.7:8765", "t", opts(health(200)))) as { problem: string }).problem, "mixed");
  assert.equal(((await testConnection("ftp://x", "t", opts(health(200)))) as { problem: string }).problem, "url");
  assert.equal(((await testConnection("https://a.example", "bad", opts(health(401)))) as { problem: string }).problem, "token");
  assert.equal(((await testConnection("https://a.example", "bad", opts(health(429)))) as { problem: string }).problem, "rate");
  assert.equal(((await testConnection("https://a.example", "t", opts(health(500)))) as { problem: string }).problem, "http");
  const notRunner = await testConnection("https://a.example", "t", opts(() => new Response("<html>", { status: 200 })));
  assert.equal((notRunner as { problem: string }).problem, "not-runner");
  const down = await testConnection("https://a.example", "t", opts(() => Promise.reject(new TypeError("failed"))));
  assert.equal((down as { problem: string }).problem, "unreachable");
  // a cors request fails but an opaque one gets through: the server is there, it just doesn't allow this site
  const cors = await testConnection(
    "https://a.example",
    "t",
    opts((_u, init) => (init?.mode === "no-cors" ? new Response(null, { status: 200 }) : Promise.reject(new TypeError("cors")))),
  );
  assert.equal((cors as { problem: string }).problem, "cors");
});
