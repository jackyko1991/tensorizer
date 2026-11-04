import { AGENT_STRINGS } from "./strings.agent";
import { JobGone, readJobEvents, visitorId, type JobEvent } from "./jobs";
import { activeServer, authHeaderFor, getToken } from "./servers";
import { WORKSPACE_STRINGS } from "./strings.workspace";
import type { Manifest, Params, Problem } from "./types";

// Where runs go (chosen in the Backend panel, kept in localStorage). There is no shared server:
// "local": a runner on the visitor's own machine at a port they can change (reachable from the public site because
// it allows Private Network Access; `pixi run api` listens on 8000, `python -m tensorizer_physics.serve` on 8765).
// "remote": one of the visitor's own servers (servers.ts), with its pairing token sent as `Authorization: Bearer`.
export type BackendChoice = "local" | "remote";
export type Device = "auto" | "cpu" | "cuda" | "mps";
export const DEFAULT_LOCAL_PORT = 8000;
/** The runner's own default port (python -m tensorizer_physics.serve, the server installer): tried after 8000. */
export const RUNNER_PORT = 8765;
const CHOICE_KEY = "tensorizer.backend";
const PORT_KEY = "tensorizer.localPort";
const OLD_URL_KEY = "tensorizer.localUrl"; // before the port-only field
const DEVICE_KEY = "tensorizer.device";

function stored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // private window: the setting lasts for this page only
  }
}

/** True once the visitor chose a port (otherwise 8000 is tried, then 8765). */
export const localPortSaved = () => stored(PORT_KEY) !== null || stored(OLD_URL_KEY) !== null;

/** The local runner's port: Local always means this computer, so only the port can change. */
export function localPort(): number {
  const saved = Number(stored(PORT_KEY) ?? /:(\d+)/.exec(stored(OLD_URL_KEY) ?? "")?.[1]);
  return Number.isInteger(saved) && saved > 0 && saved < 65536 ? saved : DEFAULT_LOCAL_PORT;
}
export const localAddress = (port = localPort()) => `http://localhost:${port}`;
export function setLocalPort(port: number) {
  store(PORT_KEY, String(Number.isInteger(port) && port > 0 && port < 65536 ? port : DEFAULT_LOCAL_PORT));
  if (backendChoice() === "local") API = localAddress();
}

/** The device runs ask for on the local runner (the shared server always uses its default). */
export function deviceChoice(): Device {
  const d = stored(DEVICE_KEY);
  return d === "cpu" || d === "cuda" || d === "mps" ? d : "auto";
}
export const setDeviceChoice = (d: Device) => store(DEVICE_KEY, d);
const runDevice = (): Device => (backendChoice() === "local" ? deviceChoice() : "auto");

/** "remote" only while an own server is chosen and still in the list; otherwise this device (also for the old
 *  "server" choice, the shared demo that no longer exists). */
export function backendChoice(): BackendChoice {
  return stored(CHOICE_KEY) === "remote" && activeServer() ? "remote" : "local";
}

const baseFor = (choice: BackendChoice) => (choice === "remote" ? activeServer()!.url : localAddress());

export function setBackendChoice(choice: BackendChoice) {
  store(CHOICE_KEY, choice);
  API = baseFor(backendChoice());
}

/** Look again at the active server (after it was edited, removed or its token changed). */
export function refreshBackend() {
  API = baseFor(backendChoice());
}

let API = baseFor(backendChoice());

/** fetch to the current backend: adds the pairing token for the visitor's own remote server, and only for URLs
 *  under that server's address (a header already set, e.g. a connector token, is left alone). */
function call(url: string, init: RequestInit = {}): Promise<Response> {
  const server = backendChoice() === "remote" ? activeServer() : null;
  const auth = server ? authHeaderFor(url, server, getToken(server.id)) : null;
  if (!auth) return fetch(url, init);
  const headers = new Headers(init.headers);
  if (!headers.has("Authorization")) headers.set("Authorization", auth);
  return fetch(url, { ...init, headers });
}
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
  /** what the runner computes on (newer runners) */
  runner?: string;
  torch?: string;
  python?: string;
  os?: string;
  devices?: Device[];
  gpu?: string | null;
  /** an NVIDIA GPU this runner's PyTorch can't use (a CPU-only build) */
  nvidia?: string | null;
  /** what the runner offers; "jobs" = detached runs (POST /api/jobs) that keep going after the page closes */
  features?: string[];
}

export const hasJobs = (h: Health | null | undefined | "checking") =>
  !!h && h !== "checking" && !!h.features?.includes("jobs");

/** Check a runner at an address without switching to it (the Backend panel's Test). */
export async function probe(base: string): Promise<Health | null> {
  try {
    const res = await fetch(`${base.replace(/\/$/, "")}/api/health`, { signal: AbortSignal.timeout(4000) });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

/** The run queue as the queue window shows it (problems and times; the caller's own runs marked). */
export interface QueueRun {
  problem: string;
  state: "running" | "waiting";
  position?: number;
  seconds: number;
  mine: boolean;
}
export interface QueueDetails {
  running: number;
  waiting: number;
  concurrent: number;
  max_run_s: number;
  runs: QueueRun[];
}

export async function queueDetails(): Promise<QueueDetails | null> {
  try {
    const res = await call(`${API}/api/queue`);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

/** Drop the caller's own waiting or stuck runs; returns how many. */
export async function cancelMyRuns(): Promise<number> {
  const res = await call(`${API}/api/queue/cancel`, { method: "POST" });
  return res.ok ? (await res.json()).cancelled : 0;
}

export async function health(): Promise<Health | null> {
  try {
    const res = await call(`${API}/api/health`);
    return res.ok ? await res.json() : null;
  } catch {
    // This device, no port chosen yet: the runner's own default port (8765) when 8000 doesn't answer
    if (backendChoice() !== "local" || localPortSaved()) return null;
    const other = await probe(localAddress(RUNNER_PORT));
    if (other) setLocalPort(RUNNER_PORT);
    return other;
  }
}

/** test.py's verdict after a run (TDD): ok = every test passed. */
export interface TestTally {
  passed: number;
  failed: number;
  ok: boolean;
  /** each test's outcome (and failure message), for the TDD window */
  items?: { name: string; outcome: string; message: string }[];
}

export interface RunResult {
  output: string;
  console: string;
  cached: boolean;
  tests?: TestTally | null;
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
// The run stream in flight, so the queue window's Cancel can end it on this page at once (the server stops the
// process too, but an older server would keep streaming).
let current: AbortController | null = null;

/** End this page's run stream now (after the queue window's Cancel). */
export function abortMyRun(): void {
  current?.abort();
}

export async function runProblemStream(
  id: string | { session: string },
  params: Params,
  files: Record<string, string> | undefined,
  verbose: boolean,
  onLine: (line: LineEvent) => void,
  onQueued: (position: number) => void = () => {},
  onProgress: (fraction: number) => void = () => {},
): Promise<RunResult> {
  let res: Response;
  const controller = new AbortController();
  current = controller;
  try {
    const path = typeof id === "string" ? `/api/run-stream/${id}` : `/api/workspace/${id.session}/run-stream`;
    res = await call(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, files, verbose, device: runDevice() }),
      signal: controller.signal,
    });
  } catch {
    throw new RunFailed(controller.signal.aborted ? "cancelled" : unreachable());
  }
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new RunFailed(typeof body.detail === "string" ? body.detail : `request failed (${res.status})`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read().catch((e) => {
      if (controller.signal.aborted) throw new RunFailed("cancelled");
      throw interrupted(WORKSPACE_STRINGS["ws.streamCut"], e);
    });
    if (done) break;
    buffer += value;
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const event = JSON.parse(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      if (event.type === "line") onLine({ text: event.text, replace: !!event.replace });
      else if (event.type === "queued") onQueued(event.position);
      else if (event.type === "progress") onProgress(event.total ? event.done / event.total : 0);
      else if (event.type === "result") return { ...event, output: API + event.output };
      else if (event.type === "error") throw new RunFailed(event.message, event.console);
    }
  }
  throw new RunFailed("the run ended without a result");
}

// --- Detached jobs: runs that keep going after the page closes (POST /api/jobs) ------------------------------------

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";
export interface JobInfo {
  id: string;
  /** library problem id, "category/slug" */
  problem: string;
  status: JobStatus;
  /** epoch seconds */
  created: number;
  started: number | null;
  finished: number | null;
  position: number | null;
  events: number;
  error: string | null;
  result: { output: string; cached: boolean; tests: TestTally | null } | null;
}
export const jobActive = (j: JobInfo) => j.status === "queued" || j.status === "running";

const visitorHeaders = (json = false): Record<string, string> => ({
  "X-Visitor": visitorId(),
  ...(json ? { "Content-Type": "application/json" } : {}),
});
/** Tell the Jobs indicator to look again (a job was started or cancelled on this page). */
export const JOBS_CHANGED = "tensorizer:jobs-changed";
const jobsChanged = () => window.dispatchEvent(new Event(JOBS_CHANGED));

/** Start a run that outlives the page; returns its job id. */
export async function startJob(
  id: string,
  params: Params,
  files: Record<string, string> | undefined,
  verbose: boolean,
): Promise<string> {
  const [category, slug] = id.split("/");
  let res: Response;
  try {
    res = await call(`${API}/api/jobs`, {
      method: "POST",
      headers: visitorHeaders(true),
      body: JSON.stringify({ category, slug, params, files, verbose, device: runDevice() }),
    });
  } catch {
    throw new RunFailed(unreachable());
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new RunFailed(typeof body.detail === "string" ? body.detail : `request failed (${res.status})`);
  jobsChanged();
  return body.job_id;
}

/** The visitor's own jobs, newest first (null: the backend can't be reached or has no jobs). */
export async function listJobs(): Promise<JobInfo[] | null> {
  try {
    const res = await call(`${API}/api/jobs`, { headers: visitorHeaders() });
    return res.ok ? (await res.json()).jobs : null;
  } catch {
    return null;
  }
}

export async function cancelJob(id: string): Promise<boolean> {
  try {
    const res = await call(`${API}/api/jobs/${id}/cancel`, { method: "POST", headers: visitorHeaders() });
    jobsChanged();
    return res.ok && (await res.json()).cancelled;
  } catch {
    return false;
  }
}

/** Follow a job's output from event `from` (0 = replay everything), reconnecting when the stream breaks, with the
 *  same callbacks as runProblemStream. Aborting `signal` stops following; the job itself keeps running. */
export async function jobEvents(
  jobId: string,
  from: number,
  onLine: (line: LineEvent) => void,
  onQueued: (position: number) => void = () => {},
  onProgress: (fraction: number) => void = () => {},
  signal?: AbortSignal,
): Promise<RunResult> {
  let final: JobEvent;
  try {
    final = await readJobEvents({
      from,
      signal,
      open: (n, s) => call(`${API}/api/jobs/${jobId}/events?from=${n}`, { headers: visitorHeaders(), signal: s }),
      onEvent: (event) => {
        if (event.type === "line") onLine({ text: String(event.text), replace: !!event.replace });
        else if (event.type === "queued") onQueued(Number(event.position));
        else if (event.type === "progress") {
          const total = Number(event.total);
          onProgress(total ? Number(event.done) / total : 0);
        }
      },
    });
  } catch (e) {
    if (signal?.aborted) throw new RunFailed("cancelled");
    if (e instanceof JobGone) throw new RunFailed(e.message);
    throw interrupted(WORKSPACE_STRINGS["ws.streamCut"], e);
  }
  jobsChanged();
  if (final.type === "error") throw new RunFailed(String(final.message), String(final.console ?? ""));
  return { ...(final as unknown as RunResult), output: API + String(final.output) };
}

/** Start a job and follow it: runProblemStream's behaviour, but the run survives closing the page.
 *  `onJob` gets the id as soon as the job exists. */
export async function runProblemJob(
  id: string,
  params: Params,
  files: Record<string, string> | undefined,
  verbose: boolean,
  onLine: (line: LineEvent) => void,
  onQueued: (position: number) => void = () => {},
  onProgress: (fraction: number) => void = () => {},
  onJob: (jobId: string) => void = () => {},
  signal?: AbortSignal,
): Promise<RunResult> {
  const jobId = await startJob(id, params, files, verbose);
  onJob(jobId);
  const controller = new AbortController(); // the queue window's Cancel stops following at once (abortMyRun)
  current = controller;
  const both = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  return jobEvents(jobId, 0, onLine, onQueued, onProgress, both);
}

// The Jobs panel's "open": the problem page shows that job's output (now, or once it has opened).
export const OPEN_JOB = "tensorizer:open-job";
let openRequest: { id: string; problem: string } | null = null;
export function requestOpenJob(job: JobInfo) {
  openRequest = { id: job.id, problem: job.problem };
  window.dispatchEvent(new Event(OPEN_JOB));
}
/** The job the Jobs panel asked to show for this problem, once. */
export function takeOpenRequest(problem: string): string | null {
  if (openRequest?.problem !== problem) return null;
  const id = openRequest.id;
  openRequest = null;
  return id;
}

// Jobs whose outcome this browser already showed: a reopened problem page shows a job that finished while the page
// was closed, but not every old one again.
const SEEN_KEY = "tensorizer.jobsSeen";
export const jobSeen = (id: string) => (stored(SEEN_KEY) ?? "").split(",").includes(id);
export function markJobSeen(id: string) {
  const ids = (stored(SEEN_KEY) ?? "").split(",").filter((x) => x && x !== id);
  store(SEEN_KEY, [...ids.slice(-49), id].join(","));
}

export async function runProblem(
  id: string,
  params: Params,
  files?: Record<string, string>,
  verbose = false,
): Promise<RunResult> {
  let res: Response;
  try {
    res = await call(`${API}/api/run/${id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, files, verbose, device: runDevice() }),
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

/** The page's language (set by LangProvider on <html lang>), for messages made outside React. */
const ORDER = ["en", "zh-Hant", "zh-Hans", "fr", "de", "es", "ja", "ru", "ko", "hi"];
const inPageLang = (row: readonly string[]) => row[Math.max(0, ORDER.indexOf(document.documentElement.lang))] ?? row[0];
/** A stream that broke mid-way (e.g. the backend restarted): a clear message instead of the raw TypeError. */
function interrupted(row: readonly string[], cause: unknown): RunFailed {
  console.warn("stream interrupted", cause);
  return new RunFailed(inPageLang(row));
}

function unreachable(): string {
  return backendChoice() === "local"
    ? `can't reach this computer's runner at ${localAddress()}: start it (see ⚙ Backend → Local)`
    : `can't reach your server at ${API}: check it is running and your VPN or tunnel is up (⚙ Backend → Remote → Test connection)`;
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
    const res = await call(`${API}/api/agent/status`);
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
    res = await call(`${API}/api/agents/scan`, { method: "POST" });
  } catch {
    throw new RunFailed(unreachable());
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new RunFailed(typeof data.detail === "string" ? data.detail : `request failed (${res.status})`);
  return data.agents;
}

export type AgentEvent =
  | { type: "session"; id: string }
  | { type: "text"; delta: string }
  | { type: "tool"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; id: string; ok: boolean; text: string }
  | { type: "files"; problem: string; files: Record<string, string>; entry?: WsEntry }
  | {
      type: "notebook";
      problem: string;
      notebook: string;
      cells: { id: string; type: "markdown" | "code"; source: string }[];
    }
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
    res = await call(`${API}/api/agents/chat`, {
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
    throw new RunFailed(typeof detail.detail === "string" ? detail.detail : `request failed (${res.status})`);
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
      if (event.type === "run" && event.output) event.output = API + event.output;
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

export async function createConnector(previous: string | null): Promise<Connector> {
  let res: Response;
  try {
    res = await call(`${API}/api/agents/connector`, {
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
  for (const e of data.events as ConnectorEvent[]) if (e.type === "run" && e.output) e.output = API + e.output;
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
    new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier))),
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
  if (!res.ok || typeof data.key !== "string") throw new RunFailed(`OpenRouter: request failed (${res.status})`);
  return data.key;
}

/** Image URLs in a problem entry: prebuilt files under generated/, run outputs from the API, or inline data. */
export function assetUrl(path: string): string {
  if (/^(data:|https?:)/.test(path)) return path;
  return path.startsWith("/api/") ? API + path : generatedUrl(path);
}

/** A blank new problem in the Workspace (a copy of the template), to edit and run by hand. */
export async function newWorkspace(): Promise<Problem> {
  const res = await call(`${API}/api/workspace/new`, { method: "POST" });
  if (!res.ok) throw new RunFailed(`request failed (${res.status})`);
  return res.json();
}

// --- Workspace problems (server sandboxes; the browser keeps the files) ----------------------------------

/** Library-shaped entry plus the raw meta.yaml, as the /api/ws routes return it. */
export type WsEntry = Problem & { session: string; meta: string };

async function wsPost<T>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await call(`${API}/api/ws${path}`, {
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
    if (detail && typeof detail === "object") throw new RunFailed(detail.message, detail.console);
    throw new RunFailed(typeof detail === "string" ? detail : `request failed (${res.status})`);
  }
  return data as T;
}

/** A new, empty problem: a skeleton that runs. */
export const wsBlank = () => wsPost<WsEntry>("/blank");
/** A new problem copied from the library ("category/slug"). */
export const wsCopy = (problem: string) => wsPost<WsEntry>("/copy", { problem });
/** Write the files into the problem's sandbox (re-created when it expired); returns the parsed entry. */
export const wsSync = (session: string | null, files: Record<string, string>, data?: string[]) =>
  wsPost<WsEntry & { data: { name: string; size: number }[] }>("/sync", { session, files, data });
/** Upload one data file into a sandbox's data/ folder. */
export async function wsPutData(session: string, name: string, blob: Blob): Promise<void> {
  let res: Response;
  try {
    res = await call(`${API}/api/ws/${session}/data/${encodeURIComponent(name)}`, { method: "PUT", body: blob });
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
  wsPost<{ files: Record<string, string> }>(`/${session}/files`).then((r) => r.files);
export const wsDelete = (session: string) => wsPost<{ ok: boolean }>(`/${session}/delete`);

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
  onProgress: (fraction: number) => void = () => {},
): Promise<RunResult> {
  let res: Response;
  try {
    res = await call(`${API}/api/ws/${session}/run-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, verbose, device: runDevice() }),
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
    const { value, done } = await reader.read().catch((e) => {
      throw interrupted(WORKSPACE_STRINGS["ws.streamCut"], e);
    });
    if (done) break;
    buffer += value;
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const event = JSON.parse(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      if (event.type === "line") onLine({ text: event.text, replace: !!event.replace });
      else if (event.type === "queued") onQueued(event.position);
      else if (event.type === "progress") onProgress(event.total ? event.done / event.total : 0);
      else if (event.type === "result") return { ...event, output: API + event.output };
      else if (event.type === "error") throw new RunFailed(event.message, event.console);
    }
  }
  throw new RunFailed("the run ended without a result");
}
