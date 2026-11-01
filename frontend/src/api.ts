import type { Manifest, Params } from "./types";

// Same origin by default (Vite proxies /api in dev; uvicorn serves both in `pixi run serve`).
const API = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");
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

/** Run a problem, calling onLine for every printed line as it happens (newline-delimited JSON stream). */
export async function runProblemStream(
  id: string,
  params: Params,
  files: Record<string, string> | undefined,
  verbose: boolean,
  onLine: (line: LineEvent) => void,
): Promise<RunResult> {
  let res: Response;
  try {
    res = await fetch(`${API}/api/run-stream/${id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params, files, verbose }),
    });
  } catch {
    throw new RunFailed("can't reach the backend — is `pixi run dev` still running?");
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
    throw new RunFailed("can't reach the backend — is `pixi run dev` still running?");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = body.detail;
    if (detail && typeof detail === "object") throw new RunFailed(detail.message, detail.console);
    throw new RunFailed(typeof detail === "string" ? detail : `request failed (${res.status})`);
  }
  return { ...body, output: API + body.output };
}
