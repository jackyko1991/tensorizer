/** Detached jobs, the parts with no page or build dependencies (unit-tested with node --test):
 *  the anonymous visitor id and the reconnecting reader of a job's event log. */

const VISITOR_KEY = "tensorizer.visitor";
let memoryVisitor: string | null = null; // storage blocked: one id for this page's lifetime

type KeyValue = Pick<Storage, "getItem" | "setItem">;
const browserStorage = (): KeyValue | null => {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
};

function randomId(): string {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

const VALID = /^[A-Za-z0-9_-]{16,128}$/;
const memoryScoped = new Map<string, string>();

/** This browser's anonymous visitor id (random, kept in localStorage), sent as X-Visitor so the server shows each
 *  visitor only their own jobs and agent chats. Not a secret: it only scopes those lists; the server stores its hash.
 *  `scope` (e.g. "local" or "remote:<server id>") gives each backend its own id, so two servers can't link one
 *  browser's runs; a scope without an id yet starts from the old single id, so runs made before are still listed. */
export function visitorId(storage: KeyValue | null = browserStorage(), scope?: string): string {
  const key = scope ? `${VISITOR_KEY}.${scope}` : VISITOR_KEY;
  try {
    const saved = storage?.getItem(key);
    if (saved && VALID.test(saved)) return saved;
  } catch {
    // storage blocked
  }
  let id: string;
  if (scope) {
    const legacy = (() => {
      try {
        return storage?.getItem(VISITOR_KEY) ?? null;
      } catch {
        return null;
      }
    })();
    id = memoryScoped.get(scope) ?? (legacy && VALID.test(legacy) ? legacy : randomId());
  } else {
    id = memoryVisitor ?? randomId();
  }
  try {
    if (!storage) throw new Error("no storage");
    storage.setItem(key, id);
  } catch {
    // private window: the id lasts for this page only
    if (scope) memoryScoped.set(scope, id);
    else memoryVisitor = id;
  }
  return id;
}

export interface JobEvent {
  type: string;
  seq?: number;
  [key: string]: unknown;
}

export class JobGone extends Error {}

export interface ReadOptions {
  /** fetch(`…/events?from=${from}`) for the job; given the next event number to ask for */
  open: (from: number, signal?: AbortSignal) => Promise<Response>;
  /** every event, in order and once each, also across reconnects */
  onEvent: (event: JobEvent) => void;
  from?: number;
  signal?: AbortSignal;
  /** reconnect attempts in a row without receiving anything, before giving up */
  retries?: number;
  /** wait before reconnect attempt n (ms) */
  delay?: (attempt: number) => number;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  /** the event that ends the log (default: a job's "result" or "error") */
  isFinal?: (event: JobEvent) => boolean;
  /** asked when a stream closes without a final event: true = the run has ended, stop (an agent turn's log has no
   *  single last event; its stream simply closes when the turn ends) */
  finished?: () => Promise<boolean>;
}

const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => (clearTimeout(timer), resolve()), { once: true });
  });

/** Follow a job's events until its result or error, reconnecting from the last event received when the stream
 *  breaks (the backend restarted, the network dropped, the laptop slept). Returns the final event.
 *  Throws JobGone on 404 (expired, or not this visitor's), the abort reason when aborted, and the last error after
 *  `retries` failed attempts in a row. */
export async function readJobEvents(opts: ReadOptions): Promise<JobEvent> {
  const { open, onEvent, signal, retries = 8 } = opts;
  const isFinal = opts.isFinal ?? ((e: JobEvent) => e.type === "result" || e.type === "error");
  let last: JobEvent | null = null;
  const delay = opts.delay ?? ((n: number) => Math.min(1000 * 2 ** n, 15000));
  const sleep = opts.sleep ?? wait;
  let next = opts.from ?? 0;
  let failures = 0;
  let lastError: unknown = new Error("the job's event stream ended");
  for (;;) {
    if (signal?.aborted) throw signal.reason ?? new Error("aborted");
    let received = false;
    try {
      const res = await open(next, signal);
      if (res.status === 404) throw new JobGone("this job no longer exists");
      if (!res.ok || !res.body) throw new Error(`request failed (${res.status})`);
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        let newline: number;
        while ((newline = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (!line) continue;
          const event = JSON.parse(line) as JobEvent;
          const seq = typeof event.seq === "number" ? event.seq : next;
          if (seq < next) continue; // already seen (a replay overlapping the last connection)
          next = seq + 1;
          received = true;
          last = event;
          onEvent(event);
          if (isFinal(event)) {
            void reader.cancel().catch(() => {});
            return event;
          }
        }
      }
      if (opts.finished && (await opts.finished())) return last ?? { type: "end", seq: next - 1 };
      lastError = new Error("the job's event stream ended without a result");
    } catch (e) {
      if (e instanceof JobGone) throw e;
      if (signal?.aborted) throw signal.reason ?? e;
      lastError = e;
    }
    failures = received ? 0 : failures + 1;
    if (failures > retries) throw lastError;
    await sleep(received ? 0 : delay(failures - 1), signal);
  }
}
