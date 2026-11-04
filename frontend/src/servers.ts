/** The visitor's own runners (Backend → Remote), with no page or build dependencies (unit-tested with node --test).
 *
 *  - The list (name, URL, "remember") is kept in localStorage.
 *  - Each pairing token is kept in sessionStorage (gone when the tab closes), or in localStorage when the visitor
 *    ticks "Remember on this device". Every storage access is wrapped: a blocked storage only loses the setting.
 *  - A token is only ever sent to its own server, as `Authorization: Bearer` (never in a URL, never logged). */

export interface Server {
  id: string;
  name: string;
  /** normalised base URL, no trailing slash */
  url: string;
  /** token kept in localStorage (true) or sessionStorage (false) */
  remember: boolean;
}

type KeyValue = Pick<Storage, "getItem" | "setItem" | "removeItem">;
export interface Stores {
  local: KeyValue | null;
  session: KeyValue | null;
}

const LIST_KEY = "tensorizer.servers";
const ACTIVE_KEY = "tensorizer.server.active";
const TOKEN_PREFIX = "tensorizer.server.token.";

function browser(name: "localStorage" | "sessionStorage"): KeyValue | null {
  try {
    return (globalThis as unknown as Record<string, KeyValue | undefined>)[name] ?? null;
  } catch {
    return null;
  }
}
export const browserStores = (): Stores => ({ local: browser("localStorage"), session: browser("sessionStorage") });

function read(s: KeyValue | null, key: string): string | null {
  try {
    return s?.getItem(key) ?? null;
  } catch {
    return null;
  }
}
function write(s: KeyValue | null, key: string, value: string | null) {
  try {
    if (value === null) s?.removeItem(key);
    else s?.setItem(key, value);
  } catch {
    // storage blocked or full: the setting lasts for this page only
  }
}

/** A runner's base URL, or null if it can't be one: http(s) only, no user:password, query or fragment.
 *  "runner.example.org" becomes "https://runner.example.org"; a trailing slash is dropped. */
export function normalizeUrl(raw: string): string | null {
  let text = raw.trim();
  if (!text) return null;
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(text)) text = (isLoopbackHost(text.split(/[/:]/)[0]) ? "http://" : "https://") + text;
  let u: URL;
  try {
    u = new URL(text);
  } catch {
    return null;
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") return null;
  if (u.username || u.password || u.search || u.hash || !u.hostname) return null;
  return (u.origin + u.pathname).replace(/\/+$/, "");
}

export function isLoopbackHost(host: string): boolean {
  const h = host.toLowerCase().replace(/^\[|\]$/g, "");
  return h === "localhost" || h === "::1" || /^127\./.test(h) || h.endsWith(".localhost");
}

/** An https page can't call a plain-http server, except one on this computer (browsers block it as mixed content). */
export function mixedContent(url: string, pageProtocol: string): boolean {
  try {
    const u = new URL(url);
    return pageProtocol === "https:" && u.protocol === "http:" && !isLoopbackHost(u.hostname);
  } catch {
    return false;
  }
}

// --- the list ---------------------------------------------------------------------------------------------------

export function loadServers(stores: Stores = browserStores()): Server[] {
  try {
    const list = JSON.parse(read(stores.local, LIST_KEY) ?? "[]");
    if (!Array.isArray(list)) return [];
    return list
      .filter((s) => s && typeof s.id === "string" && typeof s.url === "string" && normalizeUrl(s.url))
      .map((s) => ({ id: s.id, name: String(s.name || s.url), url: normalizeUrl(s.url)!, remember: !!s.remember }));
  } catch {
    return [];
  }
}

function saveServers(list: Server[], stores: Stores) {
  // tokens never go into the list
  const plain = list.map(({ id, name, url, remember }) => ({ id, name, url, remember }));
  write(stores.local, LIST_KEY, JSON.stringify(plain));
}

function newId(): string {
  const bytes = new Uint8Array(8);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Add a server (the URL must pass normalizeUrl); returns it, or null for a bad URL. */
export function addServer(
  input: { name: string; url: string; token?: string; remember?: boolean },
  stores: Stores = browserStores(),
): Server | null {
  const url = normalizeUrl(input.url);
  if (!url) return null;
  const server: Server = { id: newId(), name: input.name.trim() || new URL(url).host, url, remember: !!input.remember };
  saveServers([...loadServers(stores), server], stores);
  if (input.token) setToken(server.id, input.token, server.remember, stores);
  return server;
}

/** Change a server. `token` undefined keeps the current one, "" removes it. Moving between "remember" and not moves
 *  the token to the other storage. A changed URL drops the token unless a new one is given (it was for that URL). */
export function updateServer(
  id: string,
  changes: { name?: string; url?: string; token?: string; remember?: boolean },
  stores: Stores = browserStores(),
): Server | null {
  const list = loadServers(stores);
  const old = list.find((s) => s.id === id);
  if (!old) return null;
  const url = changes.url === undefined ? old.url : normalizeUrl(changes.url);
  if (!url) return null;
  const next: Server = {
    ...old,
    name: changes.name === undefined ? old.name : changes.name.trim() || new URL(url).host,
    url,
    remember: changes.remember ?? old.remember,
  };
  let token = changes.token ?? getToken(id, stores);
  if (changes.token === undefined && url !== old.url) token = null;
  saveServers(
    list.map((s) => (s.id === id ? next : s)),
    stores,
  );
  setToken(id, token || null, next.remember, stores);
  return next;
}

export function removeServer(id: string, stores: Stores = browserStores()) {
  saveServers(
    loadServers(stores).filter((s) => s.id !== id),
    stores,
  );
  setToken(id, null, false, stores);
  if (activeServerId(stores) === id) setActiveServerId(null, stores);
}

// --- tokens -----------------------------------------------------------------------------------------------------

export function getToken(id: string, stores: Stores = browserStores()): string | null {
  return read(stores.session, TOKEN_PREFIX + id) ?? read(stores.local, TOKEN_PREFIX + id);
}

/** Keep `token` in one storage only: localStorage when `remember`, else sessionStorage. null removes it. */
export function setToken(id: string, token: string | null, remember: boolean, stores: Stores = browserStores()) {
  const key = TOKEN_PREFIX + id;
  const value = token?.trim() || null;
  write(remember ? stores.session : stores.local, key, null);
  write(remember ? stores.local : stores.session, key, value);
}

// --- the active server ------------------------------------------------------------------------------------------

export const activeServerId = (stores: Stores = browserStores()) => read(stores.local, ACTIVE_KEY);
export const setActiveServerId = (id: string | null, stores: Stores = browserStores()) =>
  write(stores.local, ACTIVE_KEY, id);
export function activeServer(stores: Stores = browserStores()): Server | null {
  const id = activeServerId(stores);
  return loadServers(stores).find((s) => s.id === id) ?? null;
}

/** The Authorization header for a request to `url`, only when `url` is under `server`'s own address. */
export function authHeaderFor(url: string, server: Server | null, token: string | null): string | null {
  if (!server || !token) return null;
  return url === server.url || url.startsWith(server.url + "/") ? `Bearer ${token}` : null;
}

// --- Test connection --------------------------------------------------------------------------------------------

export type Problem = "url" | "mixed" | "unreachable" | "cors" | "not-runner" | "token" | "rate" | "http";
export interface HealthInfo {
  status: string;
  runner?: string;
  devices?: string[];
  gpu?: string | null;
  auth?: "token" | "none";
  shared?: boolean;
  code_exec?: boolean;
  queue?: { running: number; waiting: number };
}
export type TestResult =
  | { ok: true; health: HealthInfo; queue: { running: number; waiting: number } | null }
  | { ok: false; problem: Problem; status?: number; health?: HealthInfo };

/** Health (public), then one authenticated call (GET /api/queue), and a precise reason when either fails. */
export async function testConnection(
  rawUrl: string,
  token: string | null,
  opts: { fetch?: typeof fetch; pageProtocol?: string; timeoutMs?: number } = {},
): Promise<TestResult> {
  const f = opts.fetch ?? fetch;
  const url = normalizeUrl(rawUrl);
  if (!url) return { ok: false, problem: "url" };
  if (mixedContent(url, opts.pageProtocol ?? globalThis.location?.protocol ?? "https:"))
    return { ok: false, problem: "mixed" };
  const timeout = () => AbortSignal.timeout(opts.timeoutMs ?? 6000);
  let res: Response;
  try {
    res = await f(`${url}/api/health`, { signal: timeout() });
  } catch {
    // The request failed. If an opaque (no-cors) request gets through, the server is there but doesn't allow this
    // site (CORS); otherwise it can't be reached at all.
    try {
      await f(`${url}/api/health`, { mode: "no-cors", signal: timeout() });
      return { ok: false, problem: "cors" };
    } catch {
      return { ok: false, problem: "unreachable" };
    }
  }
  const health = (await res.json().catch(() => null)) as HealthInfo | null;
  if (!res.ok || !health || health.status !== "ok") return { ok: false, problem: "not-runner", status: res.status };
  let q: Response;
  try {
    q = await f(`${url}/api/queue`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      signal: timeout(),
    });
  } catch {
    return { ok: false, problem: "cors", health };
  }
  if (q.status === 401) return { ok: false, problem: "token", status: 401, health };
  if (q.status === 429) return { ok: false, problem: "rate", status: 429, health };
  if (!q.ok) return { ok: false, problem: "http", status: q.status, health };
  const queue = await q.json().catch(() => null);
  return { ok: true, health, queue: queue ? { running: queue.running, waiting: queue.waiting } : null };
}
