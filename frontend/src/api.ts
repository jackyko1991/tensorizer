import { AGENT_STRINGS } from "./strings.agent";
import { WORKSPACE_STRINGS } from "./strings.workspace";
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
  if (!res.ok)
    throw new Error(
      "manifest.json not found — run `pixi run build-assets` first",
    );
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
    const path =
      typeof id === "string"
        ? `/api/run-stream/${id}`
        : `/api/workspace/${id.session}/run-stream`;
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
    throw new RunFailed(
      typeof body.detail === "string"
        ? body.detail
        : `request failed (${res.status})`,
    );
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read().catch((e) => {
      throw interrupted(WORKSPACE_STRINGS["ws.streamCut"], e);
    });
    if (done) break;
    buffer += value;
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const event = JSON.parse(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      if (event.type === "line")
        onLine({ text: event.text, replace: !!event.replace });
      else if (event.type === "queued") onQueued(event.position);
      else if (event.type === "result")
        return { ...event, output: API + event.output };
      else if (event.type === "error")
        throw new RunFailed(event.message, event.console);
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
    if (detail && typeof detail === "object")
      throw new RunFailed(detail.message, detail.console);
    throw new RunFailed(
      typeof detail === "string" ? detail : `request failed (${res.status})`,
    );
  }
  return { ...body, output: API + body.output };
}

/** The page's language (set by LangProvider on <html lang>), for messages made outside React. */
const ORDER = ["en", "zh-Hant", "zh-Hans", "fr", "de", "es", "ja", "ru", "ko", "hi"];
const inPageLang = (row: readonly string[]) => row[Math.max(0, ORDER.indexOf(document.documentElement.lang))] ?? row[0];
/** A stream that broke mid-way (e.g. the backend restarted): a clear message instead of the raw TypeError. */
function interrupted(row: readonly string[], cause: unknown): RunFailed {
  console.warn("stream interrupted", cause);
  return new RunFailed(inPageLang(row));
}

function unreachable(): string {
  return API === LOCAL_API
    ? "can't reach this computer's backend — start it with `pixi run api` (port 8000)"
    : "can't reach the backend — the server may be waking up; try again in a minute";
}

// --- Agent (AI Assistant: Local | Cloud | API) ------------------------------------------------------------------

/** LOCAL: a coding-agent CLI on the backend's machine. API: Claude Code with the visitor's key for a provider. */
export type LocalAgentId = "claude-code" | "codex" | "opencode" | "gemini";
export type AgentId = LocalAgentId | "claude-api";
export type AgentMode = "local" | "cloud" | "api";

export interface AgentProvider {
  id: string;
  label: string;
  available: boolean;
  models?: string[];
}

export interface AgentStatus {
  enabled: boolean;
  /** Claude Code (local): this machine's own login, no setup. API: the visitor's key for a provider. */
  agents: {
    id: AgentId;
    label: string;
    available: boolean;
    logged_in?: boolean | null;
    providers?: AgentProvider[];
  }[];
  default: AgentId;
  models: string[];
  workspace: boolean;
  /** Which modes this backend offers (LOCAL only on the user's own machine). */
  modes?: { local: boolean; cloud: boolean; api: boolean };
}

export async function agentStatus(): Promise<AgentStatus | null> {
  try {
    const res = await fetch(`${API}/api/agent/status`);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

/** One agent CLI found (or not) by "Search this computer": booleans and a version number only. */
export interface FoundAgent {
  id: LocalAgentId;
  label: string;
  installed: boolean;
  version: string | null;
  logged_in: boolean | null;
  experimental: boolean;
  docs: string;
}

export async function scanAgents(): Promise<FoundAgent[]> {
  let res: Response;
  try {
    res = await fetch(`${API}/api/agents/scan`, { method: "POST" });
  } catch {
    throw new RunFailed(unreachable());
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok)
    throw new RunFailed(
      typeof data.detail === "string"
        ? data.detail
        : `request failed (${res.status})`,
    );
  return data.agents;
}

export type AgentEvent =
  | { type: "session"; id: string }
  | { type: "text"; delta: string }
  | { type: "tool"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; id: string; ok: boolean; text: string }
  | { type: "files"; problem: string; files: Record<string, string>; entry?: WsEntry }
  | { type: "notebook"; problem: string; notebook: string; cells: { id: string; type: "markdown" | "code"; source: string }[] }
  | { type: "problem"; entry: Problem }
  | { type: "run"; output: string | null; console: string; ok: boolean }
  | { type: "queued"; position: number }
  | { type: "done"; cost_usd?: number | null }
  | { type: "error"; message: string };

export interface AgentChatBody {
  session: string | null;
  problem: string;
  message: string;
  model?: string;
  agent: AgentId;
  provider?: string;
  context?: Record<string, unknown>;
  /** A Workspace problem: its sandbox, synced just before the message. */
  sandbox?: string;
}

/** One user turn: streams the agent's events until "done" or "error". The API key, if any, goes in a header
 *  and is never stored by the server. */
export async function agentChat(
  body: AgentChatBody,
  apiKey: string | null,
  onEvent: (e: AgentEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) headers["X-Agent-Key"] = apiKey;
  let res: Response;
  try {
    res = await fetch(`${API}/api/agents/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
  } catch {
    if (signal.aborted) return;
    throw new RunFailed(unreachable());
  }
  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => ({}));
    throw new RunFailed(
      typeof detail.detail === "string"
        ? detail.detail
        : `request failed (${res.status})`,
    );
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read().catch((e) => {
      if (signal.aborted) return { value: undefined, done: true as const }; // the user pressed stop
      throw interrupted(AGENT_STRINGS["agent.streamCut"], e);
    });
    if (done) return;
    buffer += value;
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const event = JSON.parse(buffer.slice(0, newline)) as AgentEvent;
      buffer = buffer.slice(newline + 1);
      if (event.type === "run" && event.output)
        event.output = API + event.output;
      onEvent(event);
    }
  }
}

/** CLOUD: a connector token for this browser and the MCP URL to give the user's own agent. */
export interface Connector {
  token: string;
  url: string;
  url_with_token: string;
  local: boolean;
}

export async function createConnector(
  previous: string | null,
): Promise<Connector> {
  let res: Response;
  try {
    res = await fetch(`${API}/api/agents/connector`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ previous }),
    });
  } catch {
    throw new RunFailed(unreachable());
  }
  if (!res.ok) throw new RunFailed(`request failed (${res.status})`);
  return res.json();
}

export type ConnectorEvent = AgentEvent & { seq: number };

/** What the cloud agent did since `after`; null when the token is no longer valid. */
export async function connectorEvents(
  token: string,
  after: number,
): Promise<{ events: ConnectorEvent[]; seq: number } | null> {
  const res = await fetch(`${API}/api/agents/connector/events?after=${after}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 401) return null;
  if (!res.ok) throw new RunFailed(`request failed (${res.status})`);
  const data = await res.json();
  for (const e of data.events as ConnectorEvent[])
    if (e.type === "run" && e.output) e.output = API + e.output;
  return data;
}

// OpenRouter "Sign in" (OAuth PKCE, https://openrouter.ai/docs/guides/overview/auth/oauth): the page sends the user
// to openrouter.ai/auth with a S256 code challenge, OpenRouter redirects back with ?code=, and the page exchanges it
// (with the verifier) for a user-controlled key at /api/v1/auth/keys. The key stays in this tab's sessionStorage.
const OR_VERIFIER = "tensorizer.openrouter.verifier";
const OR_RETURN = "tensorizer.openrouter.return";

function base64url(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

export async function openRouterSignIn(): Promise<void> {
  const verifier = base64url(crypto.getRandomValues(new Uint8Array(32)));
  const challenge = base64url(
    new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)),
    ),
  );
  sessionStorage.setItem(OR_VERIFIER, verifier);
  sessionStorage.setItem(OR_RETURN, location.search + location.hash);
  const callback = location.origin + location.pathname; // OpenRouter appends ?code=
  location.href =
    "https://openrouter.ai/auth?" +
    new URLSearchParams({
      callback_url: callback,
      code_challenge: challenge,
      code_challenge_method: "S256",
    });
}

/** Back from OpenRouter (?code= in the URL): exchange the code for a key and restore the page's URL. */
export async function openRouterCallback(): Promise<string | null> {
  const code = new URLSearchParams(location.search).get("code");
  let verifier: string | null = null;
  try {
    verifier = sessionStorage.getItem(OR_VERIFIER);
  } catch {
    return null;
  }
  if (!code || !verifier) return null;
  const back = sessionStorage.getItem(OR_RETURN) ?? "";
  sessionStorage.removeItem(OR_VERIFIER);
  sessionStorage.removeItem(OR_RETURN);
  history.replaceState(null, "", location.pathname + back);
  const res = await fetch("https://openrouter.ai/api/v1/auth/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      code,
      code_verifier: verifier,
      code_challenge_method: "S256",
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || typeof data.key !== "string")
    throw new RunFailed(`OpenRouter: request failed (${res.status})`);
  return data.key;
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

// --- Workspace problems (server sandboxes; the browser keeps the files) ----------------------------------

/** Library-shaped entry plus the raw meta.yaml, as the /api/ws routes return it. */
export type WsEntry = Problem & { session: string; meta: string };

async function wsPost<T>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}/api/ws${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new RunFailed(unreachable());
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    if (detail && typeof detail === "object")
      throw new RunFailed(detail.message, detail.console);
    throw new RunFailed(
      typeof detail === "string" ? detail : `request failed (${res.status})`,
    );
  }
  return data as T;
}

/** A new, empty problem: a skeleton that runs. */
export const wsBlank = () => wsPost<WsEntry>("/blank");
/** A new problem copied from the library ("category/slug"). */
export const wsCopy = (problem: string) =>
  wsPost<WsEntry>("/copy", { problem });
/** Write the files into the problem's sandbox (re-created when it expired); returns the parsed entry. */
export const wsSync = (session: string | null, files: Record<string, string>, data?: string[]) =>
  wsPost<WsEntry & { data: { name: string; size: number }[] }>("/sync", { session, files, data });
/** Upload one data file into a sandbox's data/ folder. */
export async function wsPutData(session: string, name: string, blob: Blob): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API}/api/ws/${session}/data/${encodeURIComponent(name)}`, { method: "PUT", body: blob });
  } catch {
    throw new RunFailed(unreachable());
  }
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail;
    throw new RunFailed(typeof detail === "string" ? detail : `upload failed (${res.status})`);
  }
}
/** All files of a sandbox, including meta.yaml (for problems the AI created). */
export const wsFiles = (session: string) =>
  wsPost<{ files: Record<string, string> }>(`/${session}/files`).then(
    (r) => r.files,
  );
export const wsDelete = (session: string) =>
  wsPost<{ ok: boolean }>(`/${session}/delete`);

export interface NotebookCellResult {
  index: number;
  stdout: string;
  figures: string[];
  error: string | null;
}

/** Run code cells 1..k in a fresh process; the last cell's output (or every cell's, with all). */
export const wsNotebook = (session: string, cells: string[], all = false) =>
  wsPost<{ cells: NotebookCellResult[] }>(`/${session}/notebook`, {
    cells,
    all,
  }).then((r) => r.cells);

/** Render a workspace problem in its sandbox, streaming lines like runProblemStream. */
export async function wsRunStream(
  session: string,
  params: Params,
  verbose: boolean,
  onLine: (line: LineEvent) => void,
  onQueued: (position: number) => void = () => {},
): Promise<RunResult> {
  let res: Response;
  try {
    res = await fetch(`${API}/api/ws/${session}/run-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, verbose }),
    });
  } catch {
    throw new RunFailed(unreachable());
  }
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new RunFailed(
      typeof body.detail === "string"
        ? body.detail
        : `request failed (${res.status})`,
    );
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read().catch((e) => {
      throw interrupted(WORKSPACE_STRINGS["ws.streamCut"], e);
    });
    if (done) break;
    buffer += value;
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const event = JSON.parse(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      if (event.type === "line")
        onLine({ text: event.text, replace: !!event.replace });
      else if (event.type === "queued") onQueued(event.position);
      else if (event.type === "result")
        return { ...event, output: API + event.output };
      else if (event.type === "error")
        throw new RunFailed(event.message, event.console);
    }
  }
  throw new RunFailed("the run ended without a result");
}
