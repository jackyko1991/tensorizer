import type { Manifest, Params, Problem } from "./types";

// Where runs go. "server": VITE_API_URL (the public Hugging Face Space on GitHub Pages), else the same origin
// (Vite proxies /api in dev; uvicorn serves both in `pixi run serve`). "local": a backend on the visitor's own
// machine (`pixi run api`), reachable from the public site because it allows Private Network Access.
export type BackendChoice = "server" | "local";
export const LOCAL_API = "http://localhost:8000";
const SERVER_API = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");
const CHOICE_KEY = "tensorizer.backend";

export function backendChoice(): BackendChoice {
  try {
    return localStorage.getItem(CHOICE_KEY) === "local" ? "local" : "server";
  } catch {
    return "server";
  }
}

export function setBackendChoice(choice: BackendChoice) {
  try {
    localStorage.setItem(CHOICE_KEY, choice);
  } catch {
    // private window: the choice lasts for this page only
  }
  API = choice === "local" ? LOCAL_API : SERVER_API;
}

let API = backendChoice() === "local" ? LOCAL_API : SERVER_API;
const GENERATED = `${import.meta.env.BASE_URL}generated/`;

export const generatedUrl = (path: string) => GENERATED + path;

export async function loadManifest(): Promise<Manifest> {
  const res = await fetch(generatedUrl("manifest.json"));
  if (!res.ok) throw new Error("manifest.json not found — run `pixi run build-assets` first");
  return res.json();
}

export interface Health {
  status: string;
  allow_code_exec: boolean;
  queue?: { running: number; waiting: number; concurrent: number };
}

export async function health(): Promise<Health | null> {
  try {
    const res = await fetch(`${API}/api/health`);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

export interface RunResult {
  output: string;
  console: string;
  cached: boolean;
}

export class RunFailed extends Error {
  constructor(
    message: string,
    public console = "",
  ) {
    super(message);
  }
}

export interface LineEvent {
  text: string;
  /** true for a progress-bar redraw: overwrite the previous redraw instead of adding a line */
  replace: boolean;
}

/** Run a problem, calling onLine for every printed line as it happens (newline-delimited JSON stream).
 *  `id` is a library problem id, or a Workspace problem's agent session ({session} → its sandbox). */
export async function runProblemStream(
  id: string | { session: string },
  params: Params,
  files: Record<string, string> | undefined,
  verbose: boolean,
  onLine: (line: LineEvent) => void,
  onQueued: (position: number) => void = () => {},
): Promise<RunResult> {
  let res: Response;
  try {
    const path = typeof id === "string" ? `/api/run-stream/${id}` : `/api/workspace/${id.session}/run-stream`;
    res = await fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, files, verbose }),
    });
  } catch {
    throw new RunFailed(unreachable());
  }
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new RunFailed(typeof body.detail === "string" ? body.detail : `request failed (${res.status})`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const event = JSON.parse(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      if (event.type === "line") onLine({ text: event.text, replace: !!event.replace });
      else if (event.type === "queued") onQueued(event.position);
      else if (event.type === "result") return { ...event, output: API + event.output };
      else if (event.type === "error") throw new RunFailed(event.message, event.console);
    }
  }
  throw new RunFailed("the run ended without a result");
}

export async function runProblem(
  id: string,
  params: Params,
  files?: Record<string, string>,
  verbose = false,
): Promise<RunResult> {
  let res: Response;
  try {
    res = await fetch(`${API}/api/run/${id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, files, verbose }),
    });
  } catch {
    throw new RunFailed(unreachable());
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = body.detail;
    if (detail && typeof detail === "object") throw new RunFailed(detail.message, detail.console);
    throw new RunFailed(typeof detail === "string" ? detail : `request failed (${res.status})`);
  }
  return { ...body, output: API + body.output };
}

function unreachable(): string {
  return API === LOCAL_API
    ? "can't reach this computer's backend — start it with `pixi run api` (port 8000)"
    : "can't reach the backend — the server may be waking up; try again in a minute";
}

// --- Agent ("Ask Claude") ---------------------------------------------------------------------------

export type AgentId = "claude-code" | "claude-api";

export interface AgentStatus {
  enabled: boolean;
  /** Claude Code (local): this machine's own login, no setup. API: the visitor's key for a provider. */
  agents: {
    id: AgentId;
    label: string;
    available: boolean;
    logged_in?: boolean | null;
    providers?: { id: string; label: string; available: boolean }[];
  }[];
  default: AgentId;
  models: string[];
  workspace: boolean;
}

export async function agentStatus(): Promise<AgentStatus | null> {
  try {
    const res = await fetch(`${API}/api/agent/status`);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

export type AgentEvent =
  | { type: "session"; id: string }
  | { type: "text"; delta: string }
  | { type: "tool"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; id: string; ok: boolean; text: string }
  | { type: "files"; problem: string; files: Record<string, string> }
  | { type: "problem"; entry: Problem }
  | { type: "run"; output: string | null; console: string; ok: boolean }
  | { type: "queued"; position: number }
  | { type: "done"; cost_usd?: number }
  | { type: "error"; message: string };

/** One user turn: streams the agent's events until "done" or "error". The API key, if any, goes in a header
 *  and is never stored by the server. */
export async function agentChat(
  body: { session: string | null; problem: string; message: string; model?: string; agent: AgentId },
  apiKey: string | null,
  onEvent: (e: AgentEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) headers["X-Anthropic-Key"] = apiKey;
  let res: Response;
  try {
    res = await fetch(`${API}/api/agent/chat`, { method: "POST", headers, body: JSON.stringify(body), signal });
  } catch (e) {
    if (signal.aborted) return;
    throw new RunFailed(unreachable());
  }
  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => ({}));
    throw new RunFailed(typeof detail.detail === "string" ? detail.detail : `request failed (${res.status})`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const event = JSON.parse(buffer.slice(0, newline)) as AgentEvent;
      buffer = buffer.slice(newline + 1);
      if (event.type === "run" && event.output) event.output = API + event.output;
      onEvent(event);
    }
  }
}

/** Image URLs in a problem entry: prebuilt files under generated/, run outputs from the API, or inline data. */
export function assetUrl(path: string): string {
  if (/^(data:|https?:)/.test(path)) return path;
  return path.startsWith("/api/") ? API + path : generatedUrl(path);
}

/** A blank new problem in the Workspace (a copy of the template), to edit and run by hand. */
export async function newWorkspace(): Promise<Problem> {
  const res = await fetch(`${API}/api/workspace/new`, { method: "POST" });
  if (!res.ok) throw new RunFailed(`request failed (${res.status})`);
  return res.json();
}
