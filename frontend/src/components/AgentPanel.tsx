import { useEffect, useRef, useState } from "react";
import { applyFiles, onAsk, onContext, onSelection, type CodeSelection, type ViewContext } from "../agentBus";
import { agentChat, agentStatus, RunFailed, type AgentEvent, type AgentId, type AgentStatus } from "../api";
import { useLang, type Key } from "../i18n";
import type { Problem } from "../types";
import AgentMascot from "./AgentMascot";
import { RichText } from "./Tex";

type Block =
  | { kind: "text"; text: string }
  | { kind: "tool"; id: string; name: string; input: Record<string, unknown>; ok?: boolean; result?: string }
  | { kind: "run"; output: string | null; console: string; ok: boolean }
  | { kind: "files"; problem: string; files: Record<string, string> }
  | { kind: "error"; text: string };
interface Message {
  role: "user" | "assistant";
  blocks: Block[];
}

const KEY_STORE = "tensorizer.agent.key";
const AGENT_STORE = "tensorizer.agent.mode";
const EXAMPLES: Key[] = ["agent.ex1", "agent.ex2", "agent.ex3", "agent.ex4"];
// "Create a problem" ideas: a different three from the pool each time the panel opens.
const NEW_IDEAS = Array.from({ length: 9 }, (_, i) => `agent.newEx${i + 1}` as Key);
const pickIdeas = () => [...NEW_IDEAS].sort(() => Math.random() - 0.5).slice(0, 3);

function load(key: string): string | null {
  try {
    return sessionStorage.getItem(key);
  } catch {
    return null;
  }
}
function save(key: string, value: string | null) {
  try {
    if (value === null) sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, value);
  } catch {
    // storage blocked: the value lasts until the page closes
  }
}

type Mode = "current" | "new";
interface Chat {
  messages: Message[];
  session: string | null;
}
const EMPTY_CHAT: Chat = { messages: [], session: null };

/** Floating "Ask AI" button (bottom right) and the conversation drawer it opens. One conversation per problem
 *  ("This problem" follows the problem on screen) plus one for creating a new problem. */
export default function AgentPanel({
  problem,
  onWorkspaceProblem,
  backendKey,
}: {
  problem: Problem;
  /** A problem the agent created: shown in the left panel's Workspace section. */
  onWorkspaceProblem: (entry: Problem) => void;
  /** Changes whenever the backend changes (picker or reconnect), so the panel re-detects what it can use. */
  backendKey: string;
}) {
  const { t } = useLang();
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<AgentStatus | null | "checking">("checking");
  const [apiKey, setApiKey] = useState<string | null>(() => load(KEY_STORE));
  const [model, setModel] = useState<string>("");
  const [agent, setAgent] = useState<AgentId>(() => (load(AGENT_STORE) as AgentId) || "claude-code");
  const [provider, setProvider] = useState("anthropic");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [chats, setChats] = useState<Record<string, Chat>>({});
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState<string | null>(null); // the chat that is answering
  const [mode, setMode] = useState<Mode>("current");
  const [selection, setSelection] = useState<CodeSelection | null>(null);
  const [view, setView] = useState<ViewContext | null>(null);
  // A Workspace problem continues the conversation that created it.
  const chatKey = mode === "new" || problem.category === "workspace" ? "new" : problem.id;
  const messages = (chats[chatKey] ?? EMPTY_CHAT).messages;
  const [ideas, setIdeas] = useState(pickIdeas);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  // Re-detect what this backend offers whenever it changes; pick Claude Code (local) automatically when it is
  // installed and logged in on that machine, otherwise the API.
  useEffect(() => {
    setStatus("checking");
  }, [backendKey]);
  useEffect(() => {
    if (!open || status !== "checking") return;
    agentStatus().then((s) => {
      setStatus(s);
      if (s?.models.length) setModel((m) => m || s.models[0]);
      const local = s?.agents.find((a) => a.id === "claude-code");
      if (s && (!local?.available || local.logged_in === false) && agent === "claude-code") setAgent("claude-api");
      else if (s && load(AGENT_STORE) === null) setAgent(s.default);
    });
  }, [open, status, agent]);
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  const workProblem = chatKey === "new" ? "new" : problem.id;
  const localAgent = status !== "checking" ? status?.agents.find((a) => a.id === "claude-code") : undefined;
  const localReady = !!localAgent?.available && localAgent.logged_in !== false;
  const connected = status !== "checking" && !!status?.enabled && (agent === "claude-code" ? localReady : !!apiKey);
  // Nothing usable yet: open the Agent form so the choice (or the key field) is right there.
  useEffect(() => {
    if (open && status !== "checking" && status?.enabled && !connected) setSettingsOpen(true);
  }, [open, status, connected]);
  const shownSelection = mode === "current" && selection?.problem === problem.id ? selection : null;

  // What is on screen (problem, edits, parameters) and highlighted in the editor; right-click → Explain / Fix.
  useEffect(() => {
    return onContext(setView);
  }, []);
  useEffect(() => {
    return onSelection(setSelection);
  }, []);
  const askRef = useRef<(action: "explain" | "fix", sel: CodeSelection) => void>(() => {});
  useEffect(() => {
    return onAsk(({ action, selection: sel }) => askRef.current(action, sel));
  }, []);
  askRef.current = (action, sel) => {
    setOpen(true);
    setMode("current");
    const lines = sel.fromLine === sel.toLine ? `${sel.fromLine}` : `${sel.fromLine}–${sel.toLine}`;
    const prompt = t(action === "explain" ? "agent.explainPrompt" : "agent.fixPrompt")
      .replace("{file}", sel.file)
      .replace("{lines}", lines);
    send(prompt, sel, sel.problem);
  };

  function edit(key: string, fn: (c: Chat) => Chat) {
    setChats((cs) => ({ ...cs, [key]: fn(cs[key] ?? EMPTY_CHAT) }));
  }

  function onEvent(m: string, e: AgentEvent) {
    const update = (fn: (blocks: Block[]) => Block[]) =>
      edit(m, (c) => {
        const last = c.messages[c.messages.length - 1];
        return { ...c, messages: [...c.messages.slice(0, -1), { ...last, blocks: fn(last.blocks) }] };
      });
    if (e.type === "session") edit(m, (c) => ({ ...c, session: e.id }));
    else if (e.type === "problem") onWorkspaceProblem(e.entry);
    else if (e.type === "text")
      update((bs) => {
        const last = bs[bs.length - 1];
        return last?.kind === "text"
          ? [...bs.slice(0, -1), { ...last, text: last.text + e.delta }]
          : [...bs, { kind: "text", text: e.delta }];
      });
    else if (e.type === "tool") update((bs) => [...bs, { kind: "tool", id: e.id, name: e.name, input: e.input }]);
    else if (e.type === "tool_result")
      update((bs) => bs.map((b) => (b.kind === "tool" && b.id === e.id ? { ...b, ok: e.ok, result: e.text } : b)));
    else if (e.type === "run") update((bs) => [...bs, { kind: "run", output: e.output, console: e.console, ok: e.ok }]);
    else if (e.type === "files") {
      // Claude's edits go straight into that problem's code panel, so the editor always shows the latest version.
      if (!e.problem.startsWith("new/")) applyFiles(e.problem, e.files);
      update((bs) => [...bs.filter((b) => b.kind !== "files"), { kind: "files", problem: e.problem, files: e.files }]);
    } else if (e.type === "error") update((bs) => [...bs, { kind: "error", text: e.message }]);
  }

  async function send(text: string, sel: CodeSelection | null = shownSelection, target = workProblem) {
    const message = text.trim();
    if (!message || busy || !connected) return;
    const m = target;
    setDraft("");
    setBusy(m);
    const where = sel ? `${sel.file}:${sel.fromLine}${sel.toLine > sel.fromLine ? `–${sel.toLine}` : ""}` : "";
    edit(m, (c) => ({
      ...c,
      messages: [
        ...c.messages,
        { role: "user", blocks: [{ kind: "text", text: where ? `${message}\n\n› ${where}` : message }] },
        { role: "assistant", blocks: [] },
      ],
    }));
    abort.current = new AbortController();
    // What the user is looking at: their current code, parameters and highlighted lines.
    const onScreen = view?.problem === target ? view : null;
    const context = {
      title: onScreen?.title,
      files: onScreen?.edits,
      params: onScreen?.params,
      selection: sel ? { file: sel.file, from_line: sel.fromLine, to_line: sel.toLine, text: sel.text } : undefined,
    };
    const body = { session: (chats[m] ?? EMPTY_CHAT).session, problem: target, message, model, context, agent };
    try {
      await agentChat(body, agent === "claude-api" ? apiKey : null, (e) => onEvent(m, e), abort.current.signal);
    } catch (e) {
      onEvent(m, { type: "error", message: e instanceof RunFailed ? e.message : String(e) });
    } finally {
      setBusy(null);
      abort.current = null;
    }
  }

  function chooseAgent(a: AgentId) {
    setAgent(a);
    save(AGENT_STORE, a);
  }

  function saveKey(key: string) {
    const k = key.trim() || null;
    setApiKey(k);
    save(KEY_STORE, k);
  }

  return (
    <>
      <button
        onClick={() => (setOpen((o) => !o), setIdeas(pickIdeas()))}
        aria-label={t("agent.open")}
        aria-expanded={open}
        className="group fixed right-4 bottom-10 z-40 h-16 w-16 rounded-full border border-blue-100 bg-white p-1.5 shadow-lg transition hover:scale-105 hover:shadow-xl"
      >
        <AgentMascot thinking={busy !== null} className="h-full w-full" />
        {/* "Ask AI" bubble beside the mascot on hover / keyboard focus */}
        <span className="pointer-events-none absolute top-1/2 right-full mr-2 -translate-y-1/2 rounded-full bg-blue-700 px-3 py-1 text-xs font-medium whitespace-nowrap text-white opacity-0 shadow transition group-hover:opacity-100 group-focus-visible:opacity-100">
          {t("agent.open")}
        </span>
      </button>

      {open && (
        <section
          aria-label={t("agent.title")}
          className="fixed right-4 bottom-28 z-40 flex h-[min(640px,calc(100vh-9rem))] w-[min(440px,calc(100vw-2rem))] flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-2xl"
        >
          <header className="flex items-center gap-2 border-b border-gray-200 px-3 py-2">
            <AgentMascot thinking={busy !== null} className="h-7 w-7" />
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold text-gray-900">{t("agent.title")}</div>
              <div className="truncate text-[11px] text-gray-500">
                {status === "checking"
                  ? t("backend.checking")
                  : !status?.enabled
                    ? t("agent.unavailable")
                    : connected
                      ? `${t("agent.agent")}: ${agent === "claude-code" ? t("agent.local") : `${t("agent.api")} · Anthropic`}`
                      : t("agent.chooseAgent")}
              </div>
            </div>
            {status !== "checking" && status?.enabled && (
              <button
                onClick={() => setSettingsOpen((o) => !o)}
                aria-expanded={settingsOpen}
                title={t("agent.agent")}
                className={`rounded px-2 py-1 text-[11px] hover:bg-gray-100 ${settingsOpen ? "bg-gray-100 text-gray-800" : "text-gray-500"}`}
              >
                ⚙ {t("agent.agent")}
              </button>
            )}
            <button
              onClick={() => edit(chatKey, () => EMPTY_CHAT)}
              disabled={busy !== null}
              className="rounded px-2 py-1 text-[11px] text-gray-500 hover:bg-gray-100 disabled:opacity-40"
            >
              {t("agent.newChat")}
            </button>
            <button
              onClick={() => setOpen(false)}
              aria-label="close"
              className="rounded px-2 py-1 text-gray-400 hover:bg-gray-100"
            >
              ✕
            </button>
          </header>

          {settingsOpen && status !== "checking" && status?.enabled && (
            <div className="space-y-2 border-b border-gray-200 bg-gray-50 px-3 py-2.5 text-xs">
              <div className="flex items-center gap-2">
                <span className="w-16 shrink-0 font-medium text-gray-600">{t("agent.agent")}</span>
                <div className="grid flex-1 grid-cols-2 gap-1 rounded-md bg-white p-0.5 ring-1 ring-gray-200">
                  {(["claude-code", "claude-api"] as const).map((a) => (
                    <button
                      key={a}
                      onClick={() => chooseAgent(a)}
                      disabled={a === "claude-code" && !localAgent?.available}
                      className={`rounded px-2 py-1 ${agent === a ? "bg-blue-600 text-white" : "text-gray-700 hover:bg-gray-100"} disabled:cursor-not-allowed disabled:opacity-40`}
                    >
                      {a === "claude-code" ? t("agent.local") : t("agent.api")}
                    </button>
                  ))}
                </div>
              </div>
              {agent === "claude-code" ? (
                <p className={localReady ? "text-green-700" : "text-amber-700"}>
                  {localReady
                    ? `✓ ${t("agent.localReady")}`
                    : localAgent?.available
                      ? t("agent.localNoLogin")
                      : t("agent.localUnavailable")}
                </p>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <span className="w-16 shrink-0 font-medium text-gray-600">{t("agent.provider")}</span>
                    <select
                      value={provider}
                      onChange={(e) => setProvider(e.target.value)}
                      className="flex-1 rounded border border-gray-300 bg-white px-1.5 py-1"
                    >
                      {(status.agents.find((a) => a.id === "claude-api")?.providers ?? []).map((p) => (
                        <option key={p.id} value={p.id} disabled={!p.available}>
                          {p.label}
                          {p.available ? "" : ` (${t("agent.soon")})`}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-16 shrink-0 font-medium text-gray-600">{t("agent.apiKey")}</span>
                    <input
                      type="password"
                      defaultValue={apiKey ?? ""}
                      onChange={(e) => saveKey(e.target.value)}
                      placeholder="sk-ant-…"
                      autoComplete="off"
                      className="min-w-0 flex-1 rounded border border-gray-300 bg-white px-2 py-1 font-mono"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-16 shrink-0 font-medium text-gray-600">{t("agent.model")}</span>
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="flex-1 rounded border border-gray-300 bg-white px-1.5 py-1"
                    >
                      {status.models.map((m) => (
                        <option key={m}>{m}</option>
                      ))}
                    </select>
                  </div>
                  <p className="text-[11px] text-gray-500">
                    {t("agent.keyHelp")}{" "}
                    <a
                      href="https://console.anthropic.com/settings/keys"
                      target="_blank"
                      rel="noreferrer"
                      className="text-blue-700 hover:underline"
                    >
                      {t("agent.getKey")} ↗
                    </a>
                  </p>
                </>
              )}
            </div>
          )}

          {/* Two conversations side by side: the current problem, and a brand-new one */}
          <div role="tablist" className="grid grid-cols-2 gap-1 border-b border-gray-200 bg-gray-50 p-1 text-xs">
            {(["current", "new"] as const).map((m) => (
              <button
                key={m}
                role="tab"
                aria-selected={mode === m}
                onClick={() => setMode(m)}
                className={`rounded-md px-2 py-1.5 ${mode === m ? "bg-white font-medium text-blue-700 shadow-sm" : "text-gray-600 hover:text-gray-900"}`}
              >
                {m === "current" ? t("agent.modeCurrent") : `✨ ${t("agent.modeNew")}`}
                {busy !== null && (busy === "new") === (m === "new") && (
                  <span className="ml-1 animate-pulse text-amber-500">●</span>
                )}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 border-b border-gray-100 px-3 py-1.5 text-[11px] text-gray-600">
            <span className="shrink-0 text-gray-400">{t("agent.currentProblem")}:</span>
            <span className="truncate font-medium text-gray-800">
              {chatKey === "new" && problem.category !== "workspace" ? (
                `✨ ${t("agent.modeNew")}`
              ) : (
                <RichText text={problem.title} />
              )}
            </span>
            <span className="ml-auto shrink-0 rounded-full bg-amber-50 px-2 py-0.5 text-amber-700">
              {t("agent.trial")}
            </span>
          </div>

          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 py-3 text-sm">
            {status !== "checking" && !status?.enabled && (
              <p className="rounded-lg bg-gray-50 p-3 text-xs text-gray-600">{t("agent.unavailableHelp")}</p>
            )}

            {messages.length === 0 && connected && (
              <div className="space-y-2">
                <p className="text-xs text-gray-500">{t("agent.tryOne")}</p>
                {(workProblem === "new" ? ideas : EXAMPLES).map((k) => (
                  <button
                    key={k}
                    onClick={() => send(t(k))}
                    className="block w-full rounded-lg border border-gray-200 px-3 py-2 text-left text-xs text-gray-700 hover:border-blue-300 hover:bg-blue-50"
                  >
                    {t(k)}
                  </button>
                ))}
              </div>
            )}

            {messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="ml-8 rounded-lg bg-gray-100 px-3 py-2 whitespace-pre-wrap text-gray-900">
                  {(m.blocks[0] as { text: string }).text}
                </div>
              ) : (
                <div key={i} className="space-y-2">
                  {m.blocks.map((b, j) => (
                    <AgentBlock key={j} block={b} />
                  ))}
                  {busy === chatKey && i === messages.length - 1 && (
                    <div className="text-xs text-gray-400">
                      <span className="animate-pulse">●</span> {t("agent.working")}
                    </div>
                  )}
                </div>
              ),
            )}
            <div ref={bottom} />
          </div>

          <footer className="border-t border-gray-200 p-2">
            {shownSelection && (
              <div className="mb-1.5 flex items-start gap-2 rounded-md border border-blue-100 bg-blue-50 px-2 py-1 text-[11px] text-blue-800">
                <span className="shrink-0">⌖</span>
                <span className="min-w-0 flex-1">
                  <span className="font-mono font-medium">
                    {shownSelection.file}:{shownSelection.fromLine}
                    {shownSelection.toLine > shownSelection.fromLine && `–${shownSelection.toLine}`}
                  </span>
                  <span className="block truncate font-mono text-blue-700/70">
                    {shownSelection.text.split("\n")[0]}
                  </span>
                </span>
                <button
                  onClick={() => setSelection(null)}
                  title={t("agent.clearSel")}
                  className="shrink-0 text-blue-400 hover:text-blue-700"
                >
                  ✕
                </button>
              </div>
            )}
            <div className="flex items-end gap-2 rounded-lg border border-gray-300 px-2 py-1.5 focus-within:border-blue-400">
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    send(draft);
                  }
                }}
                rows={Math.min(5, Math.max(1, draft.split("\n").length))}
                disabled={!connected}
                placeholder={t("agent.placeholder")}
                className="min-w-0 flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-gray-400 disabled:opacity-50"
              />
              {busy ? (
                <button
                  onClick={() => abort.current?.abort()}
                  title={t("agent.stop")}
                  className="rounded-md bg-gray-800 px-2 py-1 text-xs text-white hover:bg-gray-700"
                >
                  ■
                </button>
              ) : (
                <button
                  onClick={() => send(draft)}
                  disabled={!connected || !draft.trim()}
                  title={t("agent.send")}
                  className="rounded-md bg-blue-600 px-2 py-1 text-xs text-white hover:bg-blue-700 disabled:opacity-40"
                >
                  ↑
                </button>
              )}
            </div>
            <div className="mt-1.5 flex items-center justify-between gap-2 text-[10px] text-gray-400">
              <span>{t("agent.hint")}</span>
              <button
                disabled
                title={t("agent.proSoon")}
                className="cursor-not-allowed rounded border border-gray-200 px-1.5 py-0.5"
              >
                ☁ {t("agent.pro")}
              </button>
            </div>
          </footer>
        </section>
      )}
    </>
  );
}

const TOOL_LABEL: Record<string, string> = {
  Read: "Read",
  Write: "Write",
  Edit: "Edit",
  Glob: "Glob",
  Grep: "Grep",
  mcp__tensorizer__run_problem: "Run",
};

function AgentBlock({ block }: { block: Block }) {
  const { t } = useLang();
  if (block.kind === "text") return <AgentText text={block.text} />;
  if (block.kind === "error")
    return <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{block.text}</p>;
  if (block.kind === "tool") {
    const target = String(block.input.file_path ?? block.input.pattern ?? "")
      .split(/[\\/]/)
      .pop();
    const mark = block.ok === undefined ? "text-gray-400 animate-pulse" : block.ok ? "text-green-600" : "text-red-600";
    return (
      <details className="group rounded-md text-xs">
        <summary className="flex cursor-pointer list-none items-center gap-2 text-gray-700">
          <span className={mark}>●</span>
          <span className="font-semibold">{TOOL_LABEL[block.name] ?? block.name}</span>
          <span className="truncate font-mono text-gray-500">{target}</span>
        </summary>
        {block.result && (
          <pre className="mt-1 max-h-48 overflow-auto rounded bg-gray-50 p-2 text-[11px] whitespace-pre-wrap text-gray-600">
            {block.result}
          </pre>
        )}
      </details>
    );
  }
  if (block.kind === "run")
    return (
      <div className="space-y-1">
        {block.output && <img src={block.output} alt="" className="w-full rounded border border-gray-200" />}
        <details className="text-xs">
          <summary className="cursor-pointer text-gray-500">{t("agent.console")}</summary>
          <pre className="mt-1 max-h-48 overflow-auto rounded bg-gray-900 p-2 text-[11px] text-gray-100">
            {block.console}
          </pre>
        </details>
      </div>
    );
  if (block.problem.startsWith("new/"))
    return (
      <button
        onClick={() => downloadFiles(block.files)}
        className="rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 hover:bg-blue-100"
      >
        ⬇ {t("agent.download")} ({Object.keys(block.files).length})
      </button>
    );
  return (
    <button
      onClick={() => window.dispatchEvent(new CustomEvent("tensorizer:agent-files", { detail: block }))}
      className="rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 hover:bg-blue-100"
    >
      ⇩ {t("agent.applyFiles")} ({Object.keys(block.files).join(", ")})
    </button>
  );
}

/** Assistant text: paragraphs, ``` code fences and `inline code`. */
function AgentText({ text }: { text: string }) {
  // Markdown the model writes: ``` fences, # headings, - / 1. lists, --- rules, | tables |, **bold**, `code`, $math$.
  const parts = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => `$${m.trim()}$`).split(/```[a-z]*\n?/);
  return (
    <div className="space-y-2 text-gray-800">
      {parts.map((part, i) =>
        i % 2 ? (
          <pre key={i} className="overflow-x-auto rounded-md bg-gray-50 p-2 text-[12px]">
            {part.replace(/\n$/, "")}
          </pre>
        ) : (
          <MarkdownBlocks key={i} text={part} />
        ),
      )}
    </div>
  );
}

function MarkdownBlocks({ text }: { text: string }) {
  const out: React.ReactNode[] = [];
  const lines = text.split("\n");
  for (let k = 0; k < lines.length; k++) {
    const line = lines[k];
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (!line.trim()) continue;
    if (h) {
      out.push(
        <p key={k} className={`font-semibold text-gray-900 ${h[1].length <= 2 ? "text-[15px]" : "text-sm"}`}>
          <RichText text={h[2]} />
        </p>,
      );
    } else if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      out.push(<hr key={k} className="border-gray-200" />);
    } else if (/^\s*\|/.test(line)) {
      const rows: string[][] = [];
      for (; k < lines.length && /^\s*\|/.test(lines[k]); k++) {
        if (/^\s*\|[\s:|-]+\|\s*$/.test(lines[k])) continue; // the |---|---| separator
        rows.push(
          lines[k]
            .trim()
            .replace(/^\||\|$/g, "")
            .split("|")
            .map((c) => c.trim()),
        );
      }
      k--;
      out.push(
        <div key={k} className="overflow-x-auto">
          <table className="text-xs">
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri} className={ri === 0 ? "font-semibold" : "border-t border-gray-100"}>
                  {r.map((c, ci) => (
                    <td key={ci} className="px-2 py-1 align-top">
                      <RichText text={c} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
    } else if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const items: string[] = [];
      for (; k < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[k]); k++) items.push(lines[k]);
      k--;
      out.push(
        <ul key={k} className="list-disc space-y-1 pl-5">
          {items.map((it, ii) => (
            <li key={ii}>
              <RichText text={it.replace(/^\s*([-*]|\d+\.)\s+/, "")} />
            </li>
          ))}
        </ul>,
      );
    } else {
      out.push(
        <p key={k} className="whitespace-pre-wrap">
          <RichText text={line} />
        </p>,
      );
    }
  }
  return <>{out}</>;
}

/** Save a brand-new problem's files as my_problem.zip (drop the folder into problems/<topic>/ to keep it). */
async function downloadFiles(files: Record<string, string>) {
  const { zipSync, strToU8 } = await import("fflate");
  const zip = zipSync(Object.fromEntries(Object.entries(files).map(([n, c]) => [`my_problem/${n}`, strToU8(c)])));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([zip], { type: "application/zip" }));
  a.download = "my_problem.zip";
  a.click();
  URL.revokeObjectURL(a.href);
}
