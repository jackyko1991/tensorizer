import WhereChip, { backendWhere } from "./WhereChip";
import { useEffect, useMemo, useRef, useState } from "react";
import { pythonLanguage } from "@codemirror/lang-python";
import { classHighlighter, highlightCode } from "@lezer/highlight";
import { copyText, isPython, parseAnswer } from "../agentMarkdown";
import {
  applyFiles,
  applyNotebook,
  onAsk,
  onContext,
  onOpenAgent,
  onSelection,
  type CodeSelection,
  type ViewContext,
} from "../agentBus";
import {
  agentChat,
  agentStatus,
  backendChoice,
  backendScope,
  cancelTurn,
  connectorEvents,
  deleteConversation,
  followTurn,
  listConversations,
  openRouterCallback,
  RunFailed,
  turnInfo,
  type AgentEvent,
  type Conversation,
  type AgentId,
  type AgentStatus,
  type Connector,
  type FoundAgent,
} from "../api";
import { useLang, type Key } from "../i18n";
import { suggestPrompts } from "../suggest";
import { activeServer } from "../servers";
import type { Problem } from "../types";
import AgentMascot from "./AgentMascot";
import { captureTab, CaptureError, MAX_SCREENSHOT_BYTES, pngBytes, ScreenshotPreview } from "./AgentScreenshot";
import AgentSettings, { COMING_MODES, DEFAULT_CHOICE, type AgentChoice } from "./AgentSettings";
import GearIcon from "./GearIcon";
import Tex, { RichText } from "./Tex";
import DownloadIcon from "./DownloadIcon";

type Block =
  | { kind: "text"; text: string }
  | {
      kind: "tool";
      id: string;
      name: string;
      input: Record<string, unknown>;
      ok?: boolean;
      result?: string;
    }
  | { kind: "run"; output: string | null; console: string; ok: boolean }
  | { kind: "files"; problem: string; files: Record<string, string> }
  | { kind: "error"; text: string; resumable?: boolean }
  /** The agent's ask_user tool: a multiple-choice question; the click is sent as the next message. */
  | { kind: "question"; id: string; question: string; options: string[]; multi: boolean };
interface Message {
  role: "user" | "assistant";
  blocks: Block[];
  /** Which agent answered (shown above the answer). */
  agent?: string;
  /** Mirrored from a cloud agent (CLOUD mode), not answered here. */
  cloud?: boolean;
}

// The API key stays in this tab (sessionStorage), one per provider; the choice of agent is remembered.
const KEY_STORE = "tensorizer.agent.key";
const CHOICE_STORE = "tensorizer.agent.choice";
const FOUND_STORE = "tensorizer.agent.found";
const CLOUD_STORE = "tensorizer.agent.connector";
// The header's buttons: one height, one grey, one hover (text buttons add padding, icon buttons are square)
const HEAD_BTN =
  "flex h-8 shrink-0 items-center justify-center rounded-md text-xs text-gray-500 hover:bg-gray-100 hover:text-gray-800 disabled:opacity-40";
/** A 16 px stroke icon for the header (full screen, close), the same size as the gear. */
function HeadIcon({ d, small = false }: { d: string; small?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" className={small ? "h-3.5 w-3.5" : "h-4 w-4"} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}
const SIZE_STORE = "tensorizer.agent.rect";
const MIN_W = 360;
const MIN_H = 420;
const GAP = 8; // the panel never touches the window edge
interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
type Edge = "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";
const EDGES: Edge[] = ["n", "s", "e", "w", "ne", "nw", "se", "sw"];
// thin strips on the edges, small squares on the corners (double-click one: back to the default size and place)
const EDGE_CLASS: Record<Edge, string> = {
  n: "top-0 left-3 right-3 h-1.5 cursor-ns-resize",
  s: "bottom-0 left-3 right-3 h-1.5 cursor-ns-resize",
  e: "right-0 top-3 bottom-3 w-1.5 cursor-ew-resize",
  w: "left-0 top-3 bottom-3 w-1.5 cursor-ew-resize",
  ne: "top-0 right-0 h-3 w-3 cursor-nesw-resize",
  nw: "top-0 left-0 h-3 w-3 cursor-nwse-resize",
  se: "bottom-0 right-0 h-3 w-3 cursor-nwse-resize",
  sw: "bottom-0 left-0 h-3 w-3 cursor-nesw-resize",
};
/** Keep a panel rectangle inside the window, at least MIN_W × MIN_H. */
function fitRect(r: Rect): Rect {
  const W = typeof window === "undefined" ? 1280 : window.innerWidth;
  const H = typeof window === "undefined" ? 800 : window.innerHeight;
  const w = Math.max(MIN_W, Math.min(r.w, W - 2 * GAP));
  const h = Math.max(MIN_H, Math.min(r.h, H - 2 * GAP));
  return { w, h, x: Math.min(Math.max(GAP, r.x), W - w - GAP), y: Math.min(Math.max(GAP, r.y), H - h - GAP) };
}
/** The rectangle after dragging `edge` by (dx, dy): the opposite edge stays where it is. */
function resizeRect(r: Rect, edge: Edge, dx: number, dy: number): Rect {
  let { x, y, w, h } = r;
  if (edge.includes("e")) w = r.w + dx;
  if (edge.includes("s")) h = r.h + dy;
  if (edge.includes("w")) {
    w = Math.max(MIN_W, r.w - dx);
    x = r.x + r.w - w;
  }
  if (edge.includes("n")) {
    h = Math.max(MIN_H, r.h - dy);
    y = r.y + r.h - h;
  }
  return fitRect({ x, y, w, h });
}
// "Create a problem" ideas: a different three from the pool each time the panel opens.
const NEW_IDEAS = Array.from({ length: 9 }, (_, i) => `agent.newEx${i + 1}` as Key);
const pickIdeas = () => [...NEW_IDEAS].sort(() => Math.random() - 0.5).slice(0, 3);
const REGIONS = ["summary", "equations", "translation", "notes", "terminal", "params"] as const;
type Region = (typeof REGIONS)[number] | "page";
interface PageSelection {
  region: Region;
  /** Where in the region, untranslated: "(editing)", "Vector form", "row 2 · Meaning", … */
  detail?: string;
  text: string;
}
/** A selection inside a field being edited: which field, from the nearest [data-ai-region] and the field's label. */
function fieldDetail(el: HTMLInputElement | HTMLTextAreaElement, region: Region): string {
  const label = el.getAttribute("aria-label") ?? "";
  if (region === "translation") {
    const row = el.closest("tr");
    const index = row ? [...(row.parentElement?.children ?? [])].indexOf(row) + 1 : 0;
    return `row ${index}${label ? ` · ${label}` : ""}`;
  }
  if (region === "params") {
    const name = el.closest("tr")?.querySelector("input")?.value;
    return [name, label].filter(Boolean).join(" · ");
  }
  if (region === "equations" && label) return label;
  return "(editing)";
}
const AGENT_NAMES: Record<AgentId, string> = {
  "claude-code": "Claude Code",
  codex: "Codex",
  opencode: "OpenCode",
  gemini: "Gemini CLI",
  "claude-api": "Claude Code",
};

function storage(kind: "session" | "local"): Storage | null {
  try {
    return kind === "session" ? sessionStorage : localStorage;
  } catch {
    return null;
  }
}
function load(key: string, kind: "session" | "local" = "session"): string | null {
  try {
    return storage(kind)?.getItem(key) ?? null;
  } catch {
    return null;
  }
}
function save(key: string, value: string | null, kind: "session" | "local" = "session") {
  try {
    if (value === null) storage(kind)?.removeItem(key);
    else storage(kind)?.setItem(key, value);
  } catch {
    // storage blocked: the value lasts until the page closes
  }
}
function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = load(key, "local");
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
  } catch {
    return fallback;
  }
}
const keyStore = (provider: string) => (provider === "anthropic" ? KEY_STORE : `${KEY_STORE}.${provider}`);

type Mode = "current" | "new";
interface Chat {
  messages: Message[];
  session: string | null;
  /** the server-side turns (turns.py) of this conversation, oldest first: their logs rebuild the chat */
  turns: string[];
  /** its History entry (GET /api/agents/conversations), sent with each message so a continued chat stays one */
  conversation?: string | null;
}
const EMPTY_CHAT: Chat = { messages: [], session: null, turns: [] };

// Each chat's conversation (agent session + turn ids) per backend, so a reload or a closed tab can rebuild it from
// the server's turn logs; and, per turn, the last event whose edits (files, notebook, new problem) this browser
// already applied, so a rebuilt chat never re-applies old edits but does apply ones made while the page was closed.
const CHATS_STORE = "tensorizer.agent.chats";
const APPLIED_STORE = "tensorizer.agent.applied";
interface SavedChat {
  session: string | null;
  turns: string[];
  conversation?: string | null;
}
const chatsKey = () => `${CHATS_STORE}.${backendScope()}`;
function savedChats(): Record<string, SavedChat> {
  return loadJson<Record<string, SavedChat>>(chatsKey(), {});
}
function saveChat(key: string, chat: SavedChat | null) {
  const all = savedChats();
  if (chat && chat.turns.length)
    all[key] = { session: chat.session, turns: chat.turns.slice(-40), conversation: chat.conversation ?? null };
  else delete all[key];
  save(chatsKey(), JSON.stringify(all), "local");
}
function appliedSeq(turn: string): number | undefined {
  return loadJson<Record<string, number>>(APPLIED_STORE, {})[turn];
}
function setApplied(turn: string, seq: number) {
  const all = loadJson<Record<string, number>>(APPLIED_STORE, {});
  delete all[turn];
  all[turn] = seq;
  const keys = Object.keys(all);
  for (const k of keys.slice(0, Math.max(0, keys.length - 200))) delete all[k];
  save(APPLIED_STORE, JSON.stringify(all), "local");
}
/** live: this page sent the message; replay: rebuilt from the server's log; cloud: mirrored from a connector */
type Source = "live" | "replay" | "cloud";

/** Floating "Ask AI" button (bottom right) and the conversation drawer it opens. One conversation per problem
 *  ("This problem" follows the problem on screen) plus one for creating a new problem. */
export default function AgentPanel({
  problem,
  onWorkspaceProblem,
  onSyncWorkspace,
  backendKey,
  workspaceFull = false,
}: {
  problem: Problem;
  /** Free tier (demo accounts, account.ts): the Workspace already has its one problem, so no new one. */
  workspaceFull?: boolean;
  /** A Workspace problem (by its id): push the browser's files to its sandbox; returns the live sandbox id. */
  onSyncWorkspace: (problem: string) => Promise<string>;
  /** A problem the agent created: shown in the left panel's Workspace section. */
  onWorkspaceProblem: (entry: Problem) => void;
  /** Changes whenever the backend changes (picker or reconnect), so the panel re-detects what it can use. */
  backendKey: string;
}) {
  const { lang, t } = useLang();
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<AgentStatus | null | "checking">("checking");
  const [choice, setChoiceState] = useState<AgentChoice>(() => {
    const saved = loadJson(CHOICE_STORE, DEFAULT_CHOICE);
    return COMING_MODES.has(saved.mode) ? { ...saved, mode: "local" } : saved; // Cloud and API: coming
  });
  const [apiKey, setApiKey] = useState<string | null>(() => load(keyStore(choice.provider)));
  const [found, setFoundState] = useState<FoundAgent[] | null>(() => {
    const raw = load(FOUND_STORE, "local");
    try {
      return raw ? (JSON.parse(raw) as FoundAgent[]) : null;
    } catch {
      return null;
    }
  });
  const [connector, setConnector] = useState<Connector | null>(null); // just created: URL + token shown once
  const [cloudToken, setCloudToken] = useState<string | null>(() => load(CLOUD_STORE, "local"));
  const [cloudExpired, setCloudExpired] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [chats, setChats] = useState<Record<string, Chat>>({});
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState<string | null>(null); // the chat that is answering
  const [progress, setProgress] = useState<string | null>(null); // "waiting for a slot", "render queued", ...
  const [historyOpen, setHistoryOpen] = useState(false);
  const [full, setFull] = useState(false); // full screen, with a margin so it still reads as the top layer
  // Where the panel is once the user has resized it (any edge or corner); null: the default spot, bottom right.
  const [rect, setRect] = useState<Rect | null>(() => {
    const r = loadJson<Partial<Rect>>(SIZE_STORE, {});
    return r.x != null && r.y != null && r.w && r.h ? fitRect(r as Rect) : null;
  });
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [history, setHistory] = useState<Conversation[] | null | "loading">(null);
  const turnOf = useRef<Record<string, string | null>>({}); // the turn each chat is answering (or last answered)
  const restoring = useRef<Set<string>>(new Set());
  const [scope, setScope] = useState(backendScope);
  const [mode, setMode] = useState<Mode>("current");
  const [selection, setSelection] = useState<CodeSelection | null>(null);
  const [pageSel, setPageSel] = useState<PageSelection | null>(null);
  const [shot, setShot] = useState<{ image: string; original: string } | null>(null);
  const [shotOpen, setShotOpen] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [view, setView] = useState<ViewContext | null>(null);
  // Each problem has its own chat; a Workspace problem the AI created continues the conversation that created it
  // (the "new" chat, whose agent session holds the problem).
  const inWs = problem.category === "workspace";
  const aiOwned = inWs && !!problem.session && chats.new?.session === problem.session;
  const chatKey = mode === "new" || aiOwned ? "new" : problem.id;
  const messages = (chats[chatKey] ?? EMPTY_CHAT).messages;
  const [ideas, setIdeas] = useState(pickIdeas);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  // The input grows with its text (up to max-h-36, then scrolls) instead of showing a one-line scrollbar.
  const input = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = input.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
    el.style.overflowY = el.scrollHeight > el.clientHeight + 1 ? "auto" : "hidden";
  }, [draft]);
  const panelRef = useRef<HTMLElement>(null);

  function setChoice(c: AgentChoice) {
    if (c.provider !== choice.provider) setApiKey(load(keyStore(c.provider)));
    setChoiceState(c);
    save(CHOICE_STORE, JSON.stringify(c), "local");
  }
  function setFound(f: FoundAgent[]) {
    setFoundState(f);
    save(FOUND_STORE, JSON.stringify(f), "local");
    // Nothing chosen yet (or the chosen one is gone): take the first agent that is installed.
    if (!f.some((a) => a.id === choice.local && a.installed)) {
      const first = f.find((a) => a.installed && a.logged_in !== false) ?? f.find((a) => a.installed);
      if (first) setChoice({ ...choice, local: first.id });
    }
  }
  function saveKey(key: string | null) {
    setApiKey(key);
    save(keyStore(choice.provider), key);
  }

  // Back from "Sign in with OpenRouter": exchange the ?code= for a key, kept in this tab only.
  useEffect(() => {
    openRouterCallback()
      .then((key) => {
        if (!key) return;
        save(keyStore("openrouter"), key);
        const c = {
          ...loadJson(CHOICE_STORE, DEFAULT_CHOICE),
          mode: COMING_MODES.has("api") ? ("local" as const) : ("api" as const), // API mode: coming
          provider: "openrouter",
        };
        setChoiceState(c);
        save(CHOICE_STORE, JSON.stringify(c), "local");
        setApiKey(key);
        setOpen(true);
        setSettingsOpen(true);
        setNotice(`✓ ${t("ag.orSignedIn")}`);
      })
      .catch((e) => (setOpen(true), setNotice(`${t("ag.orFailed")}: ${e instanceof Error ? e.message : e}`)));
  }, []);

  // Re-detect what this backend offers whenever it changes (nothing on the machine is searched until asked).
  useEffect(() => {
    setStatus("checking");
  }, [backendKey]);
  useEffect(() => {
    if (!open || status !== "checking") return;
    agentStatus().then((s) => {
      setStatus(s);
      // First visit: LOCAL when Claude Code is ready on this machine, otherwise API.
      if (s && load(CHOICE_STORE, "local") === null) {
        const localReady = s.modes?.local && s.default === "claude-code";
        setChoiceState((c) => ({ ...c, mode: localReady || COMING_MODES.has("api") ? "local" : "api" }));
      }
    });
  }, [open, status]);
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // The problem a message works on: the one on screen, except in a "new" chat that hasn't made it.
  const workProblem = chatKey !== "new" || aiOwned ? problem.id : "new";
  const st = status !== "checking" ? status : null;
  const localClaude = st?.agents.find((a) => a.id === "claude-code");
  const foundAgent = found?.find((a) => a.id === choice.local);
  // Claude Code is known from the status without a search; the other CLIs need "Search this computer".
  const localReady =
    !!st?.modes?.local &&
    (choice.local === "claude-code"
      ? !!localClaude?.available && localClaude.logged_in !== false
      : !!foundAgent?.installed);
  const agentId: AgentId = choice.mode === "api" ? "claude-api" : choice.local;
  const connected =
    !!st?.enabled &&
    (choice.mode === "local" ? localReady : choice.mode === "api" ? !!apiKey : !!cloudToken && !cloudExpired);
  // in CLOUD mode the user's own agent does the talking; the free tier's Workspace holds one problem only
  const tierBlocked = workProblem === "new" && workspaceFull;
  const canType = connected && choice.mode !== "cloud" && !tierBlocked;
  const agentLabel =
    choice.mode === "cloud"
      ? `${t("ag.cloudAgent")} (${t("ag.modeCloud")})`
      : choice.mode === "api"
        ? `${AGENT_NAMES[agentId]} · ${choice.provider === "openrouter" ? "OpenRouter" : "Anthropic"} (${t("ag.modeApi")})`
        : `${AGENT_NAMES[agentId]} (${t("ag.modeLocal")})`;
  // Nothing usable yet: open the Agent form so the choice (or the key field) is right there.
  useEffect(() => {
    if (open && st?.enabled && !connected) setSettingsOpen(true);
  }, [open, st, connected]);
  const shownSelection = mode === "current" && selection?.problem === problem.id ? selection : null;
  // What part of a Workspace problem is open, for the "Current problem" line: "Notebook 1" or "Source (Markdown)".
  const onScreen = view?.problem === problem.id ? view : null;
  const wsView = inWs ? onScreen?.workspace : undefined;
  const wsLabel = !wsView
    ? ""
    : wsView.kind === "notebook"
      ? wsView.notebookTitle || t("ws.notebook")
      : wsView.kind === "data"
        ? t("data.title")
        : `${t("ws.source")} (${wsView.mode === "yaml" ? "meta.yaml" : t(wsView.mode === "markdown" ? "edit.markdown" : "edit.inline")})`;
  // Suggested prompts that follow the problem's own state (a Workspace problem) or the fixed library ones.
  const suggestions = suggestPrompts(problem, onScreen, lang).map(({ key, vars }) =>
    Object.entries(vars).reduce((text, [k, v]) => text.split(`{${k}}`).join(v), t(key as Key)),
  );

  // What is on screen (problem, edits, parameters) and highlighted in the editor; right-click → Explain / Fix.
  useEffect(() => {
    return onContext(setView);
  }, []);
  useEffect(() => {
    return onOpenAgent((tab) => (setOpen(true), setMode(tab), setIdeas(pickIdeas())));
  }, []);
  // One current selection, last writer wins: an editor's (through the bus) or the page's (below).
  useEffect(() => {
    return onSelection((s) => {
      setSelection(s);
      if (s) setPageSel(null);
    });
  }, []);
  // Text highlighted anywhere else on the page (summary, equations, translation table, terminal, …), labelled by
  // the nearest [data-ai-region]. The editor reports its own precise selection; the panel itself is ignored.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let field: Element | null = null; // the edited field the current page selection came from
    const regionOf = (el: Element): Region => {
      const name = el.closest<HTMLElement>("[data-ai-region]")?.dataset.aiRegion ?? "page";
      return (REGIONS as readonly string[]).includes(name) ? (name as Region) : "page";
    };
    const onChange = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        // Text fields keep their selection out of window.getSelection(): read it from the focused field itself.
        const active = document.activeElement;
        const editable =
          active instanceof HTMLTextAreaElement ||
          (active instanceof HTMLInputElement && ["text", "search", ""].includes(active.type));
        if (editable && !panelRef.current?.contains(active) && !active.closest("[data-cell], [role='dialog']")) {
          const { selectionStart: a, selectionEnd: b, value } = active;
          if (a === null || b === null) return;
          if (a === b) {
            if (field === active) setPageSel(null); // collapsed in the same field: clear its chip
            return;
          }
          const text = value.slice(a, b).trim();
          if (!text) return;
          const region = regionOf(active);
          field = active;
          setPageSel({ region, detail: fieldDetail(active, region), text: text.slice(0, 4000) });
          setSelection(null);
          return;
        }
        const sel = document.getSelection();
        const text = sel?.toString().trim() ?? "";
        if (!sel || sel.isCollapsed || !text || !sel.anchorNode) return; // clicking elsewhere keeps the last one
        const node = sel.anchorNode;
        const el = node instanceof Element ? node : node.parentElement;
        if (!el || panelRef.current?.contains(el) || el.closest(".cm-editor, [role='dialog'], input, textarea")) return;
        field = null;
        setPageSel({ region: regionOf(el), text: text.slice(0, 4000) });
        setSelection(null);
      }, 300);
    };
    const events = ["selectionchange", "select", "keyup", "mouseup"] as const;
    events.forEach((e) => document.addEventListener(e, onChange, true));
    return () => (clearTimeout(timer), events.forEach((e) => document.removeEventListener(e, onChange, true)));
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
    send(prompt, sel, sel.problem, sel.problem === problem.id ? chatKey : sel.problem);
  };

  const regionLabel = (p: PageSelection) => {
    const name = t(`ag.region.${p.region}` as Key);
    if (!p.detail) return name;
    return p.detail === "(editing)" ? `${name} (${t("ag.editing")})` : `${name}: ${p.detail}`;
  };

  function edit(key: string, fn: (c: Chat) => Chat) {
    setChats((cs) => ({ ...cs, [key]: fn(cs[key] ?? EMPTY_CHAT) }));
  }

  function onEvent(m: string, e: AgentEvent, source: Source = "live") {
    if (e.type === "turn") {
      turnOf.current[m] = e.id;
      if (source === "live") setApplied(e.id, e.seq ?? 0); // this page will see (and apply) every event of it
      edit(m, (c) => {
        const turns = [...c.turns.filter((x) => x !== e.id), e.id];
        saveChat(m, { session: c.session, turns, conversation: c.conversation });
        return { ...c, turns };
      });
      return;
    }
    // Edits reach the editor once: live, or when a rebuilt chat shows events this browser never applied.
    const turn = turnOf.current[m];
    const fresh =
      source === "cloud" ||
      (turn != null && typeof e.seq === "number" && e.seq > (appliedSeq(turn) ?? Number.POSITIVE_INFINITY));
    const applied = () => turn && typeof e.seq === "number" && setApplied(turn, e.seq);
    if (e.type === "waiting") return setProgress(t("ag.waitingAgents"));
    if (e.type === "question") {
      const q = { kind: "question" as const, id: e.id, question: e.question, options: e.options, multi: !!e.multi };
      return edit(m, (c) => {
        const last = c.messages[c.messages.length - 1];
        if (!last || last.blocks.some((b) => b.kind === "question" && b.id === e.id)) return c;
        return { ...c, messages: [...c.messages.slice(0, -1), { ...last, blocks: [...last.blocks, q] }] };
      });
    }
    if (e.type === "queued") return setProgress(t("ag.queued").replace("{n}", String(e.position)));
    if (e.type !== "session") setProgress(null);
    const update = (fn: (blocks: Block[]) => Block[]) =>
      edit(m, (c) => {
        const last = c.messages[c.messages.length - 1];
        return {
          ...c,
          messages: [...c.messages.slice(0, -1), { ...last, blocks: fn(last.blocks) }],
        };
      });
    if (e.type === "session")
      edit(m, (c) => {
        const conversation = c.conversation ?? e.id; // a new chat's History entry is its first session
        saveChat(m, { session: e.id, turns: c.turns, conversation });
        return { ...c, session: e.id, conversation };
      });
    else if (e.type === "problem") {
      if (fresh) (onWorkspaceProblem(e.entry), applied());
    }
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
    else if (e.type === "notebook") {
      if (fresh) (applyNotebook(e), applied());
    } else if (e.type === "files") {
      // The agent's edits go straight into that problem's code panel, so the editor always shows the latest version
      // (a Workspace problem: also its Inline / Markdown view and meta.yaml, from the re-parsed entry).
      if (fresh && !e.problem.startsWith("new/")) applyFiles(e.problem, e.files, e.entry);
      if (fresh) applied();
      update((bs) => [...bs.filter((b) => b.kind !== "files"), { kind: "files", problem: e.problem, files: e.files }]);
    } else if (e.type === "error") {
      // A turn stopped by the user or cut by a restart can be picked up again ("Continue").
      const stopped = e.message === "stopped";
      const text = e.interrupted ? t("ag.interrupted") : stopped ? t("ag.stopped") : e.message;
      update((bs) => [...bs, { kind: "error", text, resumable: !!e.interrupted || stopped }]);
    }
  }

  /** Rebuild a chat from its turns' logs on the server; a turn still running is followed live. */
  async function restoreChat(m: string, saved: SavedChat) {
    if (restoring.current.has(m)) return;
    restoring.current.add(m);
    const controller = new AbortController();
    edit(m, () => ({ messages: [], session: saved.session, turns: saved.turns, conversation: saved.conversation }));
    const kept: string[] = [];
    try {
      for (const id of saved.turns) {
        let info;
        try {
          info = await turnInfo(id);
        } catch {
          return; // backend unreachable: try again when the panel next opens
        }
        if (info === null) continue; // expired on the server
        kept.push(id);
        turnOf.current[m] = id;
        edit(m, (c) => ({
          ...c,
          messages: [
            ...c.messages,
            { role: "user", blocks: [{ kind: "text", text: info.message }] },
            { role: "assistant", blocks: [], agent: AGENT_NAMES[info.agent as AgentId] ?? info.agent },
          ],
        }));
        const live = info.status === "running";
        if (live) {
          setBusy(m);
          abort.current = controller;
          setNotice(t("ag.reconnected"));
        }
        try {
          await followTurn(id, 1, (e) => onEvent(m, e, "replay"), controller.signal); // event 0 is the message
        } catch (e) {
          onEvent(m, { type: "error", message: e instanceof RunFailed ? e.message : String(e) }, "replay");
        } finally {
          if (live) {
            setBusy(null);
            abort.current = null;
            setNotice(null);
            setProgress(null);
          }
        }
      }
    } finally {
      restoring.current.delete(m);
      if (kept.length !== saved.turns.length) {
        edit(m, (c) => ({ ...c, turns: kept }));
        saveChat(m, { session: saved.session, turns: kept, conversation: saved.conversation });
      }
    }
  }

  // A different backend has its own conversations.
  useEffect(() => {
    const next = backendScope();
    if (next !== scope) {
      setScope(next);
      setChats({});
      setHistory(null);
    }
  }, [backendKey]);

  // Opening the panel (or switching problem): rebuild that chat from the server if this page doesn't have it yet.
  useEffect(() => {
    if (!open || !st?.enabled) return;
    const m = chatKey;
    if ((chats[m]?.messages.length ?? 0) > 0 || busy === m) return;
    const saved = savedChats()[m];
    if (saved?.turns.length) void restoreChat(m, saved);
  }, [open, st?.enabled, chatKey, scope]);

  async function loadHistory() {
    setHistory("loading");
    setHistory(await listConversations(workProblem));
  }
  useEffect(() => {
    if (historyOpen) void loadHistory();
  }, [historyOpen, chatKey, scope]);
  function openConversation(c: Conversation) {
    if (busy) return;
    const saved = { session: c.session, turns: c.turns, conversation: c.id };
    saveChat(chatKey, saved);
    setHistoryOpen(false);
    void restoreChat(chatKey, saved);
  }
  async function removeConversation(c: Conversation) {
    if (!(await deleteConversation(c.id))) return;
    if (c.turns.some((id) => (chats[chatKey]?.turns ?? []).includes(id))) {
      edit(chatKey, () => EMPTY_CHAT);
      saveChat(chatKey, null);
    }
    void loadHistory();
  }
  function newChat() {
    edit(chatKey, () => EMPTY_CHAT);
    saveChat(chatKey, null);
    turnOf.current[chatKey] = null;
  }
  async function stop() {
    const id = busy ? turnOf.current[busy] : null;
    // Stop the agent on the server; its stream then ends by itself with "stopped". Unreachable: stop following.
    if (!id || !(await cancelTurn(id))) abort.current?.abort();
  }
  const runsOn = backendChoice() === "remote" ? (activeServer()?.name ?? "server") : t("ag.thisComputer");

  // While an answer is coming: a ticking clock, so the wait is visibly alive.
  useEffect(() => {
    if (busy === null) return setStartedAt(null);
    setStartedAt((s) => s ?? Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [busy]);

  /** Drag an edge or a corner to resize; the panel then keeps that place and size (remembered). */
  function startResize(e: React.PointerEvent, edge: Edge) {
    e.preventDefault();
    const el = panelRef.current;
    if (!el) return;
    const b = el.getBoundingClientRect();
    const start: Rect = { x: b.left, y: b.top, w: b.width, h: b.height };
    const x0 = e.clientX;
    const y0 = e.clientY;
    let last = start;
    const move = (ev: PointerEvent) => {
      last = resizeRect(start, edge, ev.clientX - x0, ev.clientY - y0);
      setRect(last);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      save(SIZE_STORE, JSON.stringify(last), "local");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }
  // a smaller window: keep the panel on screen
  useEffect(() => {
    const onResize = () => setRect((r) => (r ? fitRect(r) : r));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // CLOUD: mirror what the user's own agent does through the connector into the chat on screen.
  const cloudSeq = useRef(0);
  const cloudChat = useRef(chatKey);
  cloudChat.current = chatKey;
  useEffect(() => {
    if (choice.mode !== "cloud" || !cloudToken || cloudExpired || !st?.enabled) return;
    let stopped = false;
    const poll = async () => {
      try {
        const res = await connectorEvents(cloudToken, cloudSeq.current);
        if (stopped) return;
        if (res === null) return setCloudExpired(true);
        if (!res.events.length) return;
        cloudSeq.current = res.seq;
        const m = cloudChat.current;
        edit(m, (c) => {
          const last = c.messages[c.messages.length - 1];
          return last?.cloud
            ? c
            : {
                ...c,
                messages: [...c.messages, { role: "assistant", blocks: [], cloud: true }],
              };
        });
        res.events.forEach((e) => onEvent(m, e, "cloud"));
      } catch {
        // backend restarting: try again on the next tick
      }
    };
    poll();
    const timer = setInterval(poll, 2500);
    return () => ((stopped = true), clearInterval(timer));
  }, [choice.mode, cloudToken, cloudExpired, st?.enabled]);

  function onConnector(c: Connector) {
    setConnector(c);
    setCloudToken(c.token);
    setCloudExpired(false);
    cloudSeq.current = 0;
    save(CLOUD_STORE, c.token, "local");
  }

  async function capture() {
    setNotice(null);
    try {
      const image = await captureTab();
      setShot({ image, original: image });
      if (pngBytes(image) > MAX_SCREENSHOT_BYTES) setNotice(t("ag.captureTooBig"));
    } catch (e) {
      setNotice(
        t(e instanceof CaptureError && e.kind === "unsupported" ? "ag.captureUnsupported" : "ag.captureDenied"),
      );
    }
  }

  async function send(
    text: string,
    sel: CodeSelection | null = shownSelection,
    target = workProblem,
    m: string = chatKey,
  ) {
    const message = text.trim();
    if (!message || busy || !canType) return;
    if (shot && pngBytes(shot.image) > MAX_SCREENSHOT_BYTES) return setNotice(t("ag.captureTooBig"));
    const page = sel ? null : pageSel;
    setDraft("");
    setBusy(m);
    setNotice(null);
    const where = sel
      ? `${sel.region ?? sel.file}:${sel.fromLine}${sel.toLine > sel.fromLine ? `–${sel.toLine}` : ""}`
      : page
        ? `${regionLabel(page)}: “${page.text.slice(0, 40)}${page.text.length > 40 ? "…" : ""}”`
        : "";
    const attached = shot ? `${where ? " · " : ""}📷 ${t("ag.screenshot")}` : "";
    edit(m, (c) => ({
      ...c,
      messages: [
        ...c.messages,
        {
          role: "user",
          blocks: [
            {
              kind: "text",
              text: where || attached ? `${message}\n\n› ${where}${attached}` : message,
            },
          ],
        },
        { role: "assistant", blocks: [], agent: AGENT_NAMES[agentId] },
      ],
    }));
    abort.current = new AbortController();
    // What the user is looking at: their current code, parameters, highlighted text and an optional screenshot.
    const onScreen = view?.problem === target ? view : null;
    const ws = onScreen?.workspace;
    const context = {
      title: onScreen?.title,
      files: ws ? undefined : onScreen?.edits, // a Workspace problem's files are synced to its sandbox instead
      params: onScreen?.params,
      workspace: ws && {
        kind: ws.kind,
        mode: ws.mode,
        markdown: ws.mode === "markdown" ? ws.markdown : undefined,
        notebook_id: ws.notebookId,
        notebook_title: ws.notebookTitle,
        cells: ws.cells,
        focused_cell: ws.focusedCell,
      },
      selection: sel
        ? {
            file: sel.file,
            from_line: sel.fromLine,
            to_line: sel.toLine,
            text: sel.text,
            region: sel.region,
            lang: sel.lang,
          }
        : page
          ? { region: page.detail ? `${page.region} ${page.detail}` : page.region, text: page.text }
          : undefined,
      screenshot: shot?.image,
    };
    setPageSel(null);
    setShot(null);
    const model =
      choice.mode === "api"
        ? choice.apiModel
        : choice.local === "claude-code"
          ? choice.localModel || st?.models[0]
          : choice.cliModel;
    const body = {
      session: (chats[m] ?? EMPTY_CHAT).session,
      problem: target,
      message,
      model: model || undefined,
      context,
      agent: agentId,
      provider: choice.provider,
      sandbox: undefined as string | undefined,
      conversation: (chats[m] ?? EMPTY_CHAT).conversation ?? undefined,
    };
    try {
      // A Workspace problem: the agent works in its sandbox, synced with the browser's files just before.
      if (target.startsWith("workspace/")) body.sandbox = await onSyncWorkspace(target);
      await agentChat(body, choice.mode === "api" ? apiKey : null, (e) => onEvent(m, e, "live"), abort.current.signal);
    } catch (e) {
      onEvent(m, {
        type: "error",
        message: e instanceof RunFailed ? e.message : String(e),
      });
    } finally {
      setBusy(null);
      setProgress(null);
      abort.current = null;
    }
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
        <>
        {full && <div className="fixed inset-0 z-40 bg-gray-900/20" onClick={() => setFull(false)} aria-hidden />}
        <section
          ref={panelRef}
          aria-label={t("agent.title")}
          data-testid="agent-panel"
          style={full || !rect ? undefined : { left: rect.x, top: rect.y, width: rect.w, height: rect.h }}
          className={`fixed z-40 flex flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-2xl ${
            full
              ? "inset-3 sm:inset-6"
              : rect
                ? ""
                : "right-4 bottom-28 h-[min(640px,calc(100dvh-9rem))] w-[min(440px,calc(100vw-2rem))]"
          }`}
        >
          {!full &&
            EDGES.map((edge) => (
              <div
                key={edge}
                onPointerDown={(e) => startResize(e, edge)}
                onDoubleClick={() => (setRect(null), save(SIZE_STORE, null, "local"))}
                title={t("ag.resize")}
                aria-hidden
                data-testid={`agent-resize-${edge}`}
                className={`absolute z-20 ${EDGE_CLASS[edge]}`}
              />
            ))}
          <header className="flex items-center gap-2 border-b border-gray-200 px-3 py-2">
            <AgentMascot thinking={busy !== null} className="h-7 w-7" />
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold text-gray-900">{t("agent.title")}</div>
              <div className="truncate text-[11px] text-gray-500" data-testid="agent-label">
                {status === "checking"
                  ? t("backend.checking")
                  : !st?.enabled
                    ? t("agent.unavailable")
                    : connected
                      ? (
                          <span className="flex min-w-0 items-center gap-1.5">
                            <WhereChip where={choice.mode === "cloud" ? "cloud" : backendWhere()} />
                            <span className="truncate" title={t("ag.runsOn").replace("{server}", runsOn)}>
                              {t("agent.agent")}: {agentLabel}
                            </span>
                          </span>
                        )
                      : t("agent.chooseAgent")}
              </div>
            </div>
            {st?.enabled && (
              <button
                onClick={() => setSettingsOpen((o) => !o)}
                aria-expanded={settingsOpen}
                title={t("agent.agent")}
                className={`${HEAD_BTN} gap-1.5 px-2.5 ${settingsOpen ? "bg-gray-100 text-gray-800" : ""}`}
              >
                <GearIcon className="h-4 w-4" /> {t("agent.agent")}
              </button>
            )}
            <button
              onClick={newChat}
              disabled={busy !== null}
              className={`${HEAD_BTN} px-2.5`}
            >
              {t("agent.newChat")}
            </button>
            <button
              onClick={() => setFull((f) => !f)}
              title={t(full ? "ag.exitFull" : "ag.full")}
              aria-label={t(full ? "ag.exitFull" : "ag.full")}
              aria-pressed={full}
              data-testid="agent-full"
              className={`${HEAD_BTN} w-8`}
            >
              <HeadIcon d={full ? "M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" : "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"} />
            </button>
            <button
              onClick={() => (setOpen(false), setFull(false))}
              aria-label="close"
              title={t("ag.close")}
              className={`${HEAD_BTN} w-8`}
            >
              <HeadIcon d="M6 6l12 12M18 6L6 18" />
            </button>
          </header>

          {settingsOpen && st?.enabled && (
            <div className="max-h-[55%] shrink-0 overflow-y-auto">
              <div className="flex items-center border-b border-gray-200 bg-gray-50 px-3 pt-2 pb-1 text-[11px] font-semibold tracking-wide text-gray-500 uppercase">
                {t("agent.agent")}
                <button
                  onClick={() => setSettingsOpen(false)}
                  aria-label={t("ag.closeSettings")}
                  title={t("ag.closeSettings")}
                  data-testid="agent-settings-close"
                  className="ml-auto rounded px-1.5 text-sm font-normal text-gray-400 hover:bg-gray-200 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
              <AgentSettings
                status={st}
                choice={choice}
                onChoice={setChoice}
                found={found}
                onFound={setFound}
                apiKey={apiKey}
                onKey={saveKey}
                connector={connector}
                onConnector={onConnector}
                cloudToken={cloudExpired ? null : cloudToken}
              />
              {choice.mode === "cloud" && cloudExpired && (
                <p className="bg-gray-50 px-3 pb-2 text-xs text-amber-700">{t("ag.cloudExpired")}</p>
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
                onClick={() => (mode === m ? setHistoryOpen((o) => !o) : (setMode(m), setHistoryOpen(false)))}
                aria-expanded={mode === m ? historyOpen : undefined}
                title={mode === m ? t("ag.historyTitle") : undefined}
                data-testid={`agent-tab-${m}`}
                className={`flex items-center justify-center gap-1 rounded-md px-2 py-1.5 ${mode === m ? "bg-white font-medium text-blue-700 shadow-sm" : "text-gray-600 hover:text-gray-900"}`}
              >
                {m === "current" ? t("agent.modeCurrent") : `✨ ${t("agent.modeNew")}`}
                {mode === m && st?.enabled && (
                  <span aria-hidden className={`text-[10px] text-gray-400 transition ${historyOpen ? "rotate-180" : ""}`}>
                    ▾
                  </span>
                )}
                {busy !== null && (busy === "new") === (m === "new") && (
                  <span className="ml-1 animate-pulse text-amber-500">●</span>
                )}
              </button>
            ))}
          </div>

          {historyOpen && st?.enabled && (
            <div className="max-h-[45%] shrink-0 overflow-y-auto border-b border-gray-200 bg-gray-50 px-3 py-2 text-xs" data-testid="agent-history-list">
              <p className="mb-1.5 font-medium text-gray-700">{t("ag.historyTitle")}</p>
              {history === "loading" ? (
                <p className="text-gray-400">{t("backend.checking")}</p>
              ) : !history?.length ? (
                <p className="text-gray-500">{t("ag.historyEmpty")}</p>
              ) : (
                <ul className="space-y-1">
                  {history.map((c) => {
                    const current = c.turns.some((id) => (chats[chatKey]?.turns ?? []).includes(id));
                    return (
                      <li key={c.id} className="flex items-center gap-2 rounded-md bg-white px-2 py-1.5 ring-1 ring-gray-200">
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-gray-800">{c.title || "…"}</div>
                          <div className="text-[10px] text-gray-400">
                            {new Date(c.updated * 1000).toLocaleString()} · {t("ag.turns").replace("{n}", String(c.turns.length))}
                            {c.status === "running" && <span className="ml-1 text-amber-600">● {t("agent.working")}</span>}
                            {current && <span className="ml-1 text-blue-600">· {t("ag.historyCurrent")}</span>}
                          </div>
                        </div>
                        <button
                          onClick={() => openConversation(c)}
                          disabled={busy !== null || current}
                          className="rounded border border-gray-300 px-2 py-0.5 text-gray-700 hover:bg-gray-50 disabled:opacity-40"
                        >
                          {t("ag.historyOpen")}
                        </button>
                        <button
                          onClick={() => void removeConversation(c)}
                          disabled={c.status === "running"}
                          title={t("ag.historyDelete")}
                          aria-label={t("ag.historyDelete")}
                          className="rounded px-1.5 py-0.5 text-gray-400 hover:bg-red-50 hover:text-red-600 disabled:opacity-40"
                        >
                          ✕
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          )}

          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 py-3 text-sm">
            {status !== "checking" && !status?.enabled && (
              <p className="rounded-lg bg-gray-50 p-3 text-xs text-gray-600">{t("agent.unavailableHelp")}</p>
            )}

            {messages.length === 0 && connected && choice.mode === "cloud" && (
              <p className="rounded-lg bg-gray-50 p-3 text-xs text-gray-600">☁ {t("ag.cloudChatHint")}</p>
            )}
            {tierBlocked && connected && (
              <p className="rounded-lg bg-amber-50 p-3 text-xs text-amber-800" data-testid="agent-tier-limit">
                {t("ag.freeLimit")}
              </p>
            )}
            {messages.length === 0 && canType && (
              <div className="space-y-2">
                <p className="text-xs text-gray-500">{t("agent.tryOne")}</p>
                {(workProblem === "new" ? ideas.map((k) => t(k)) : suggestions).map((text) => (
                  <button
                    key={text}
                    onClick={() => send(text)}
                    data-testid="agent-suggestion"
                    className="block w-full rounded-lg border border-gray-200 px-3 py-2 text-left text-xs text-gray-700 hover:border-blue-300 hover:bg-blue-50"
                  >
                    <RichText text={text} />
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
                <div key={i} className="group/answer space-y-2">
                  {(m.cloud || m.agent) && (
                    <div className="text-[11px] font-medium text-gray-600" data-testid="agent-answer-name">
                      {m.cloud ? `☁ ${t("ag.cloudAgent")}` : m.agent}
                    </div>
                  )}
                  {groupSteps(m.blocks).map((g, j) =>
                    g.kind === "steps" ? (
                      <Steps
                        key={j}
                        steps={g.steps}
                        live={busy === chatKey && i === messages.length - 1 && j === groupSteps(m.blocks).length - 1}
                      />
                    ) : (
                      g.block.kind === "question" ? (
                        <QuestionCard
                          key={j}
                          block={g.block}
                          // only the latest answer's question can be answered, and only when nothing is running
                          active={i === messages.length - 1 && busy === null && canType}
                          onAnswer={(answer) => send(answer, null)}
                        />
                      ) : (
                        <AgentBlock key={j} block={g.block} />
                      )
                    ),
                  )}
                  {busy === chatKey && i === messages.length - 1 && (
                    <div className="text-xs text-gray-400" data-testid="agent-working">
                      <span className="animate-pulse">●</span> {progress ?? currentStep(m.blocks) ?? t("agent.working")}
                      {startedAt && <span className="ml-1 tabular-nums">· {Math.max(0, Math.round((now - startedAt) / 1000))} s</span>}
                    </div>
                  )}
                  {/* below a finished answer: copy its text (as Markdown) */}
                  {!(busy === chatKey && i === messages.length - 1) && copyText(m.blocks) && (
                    <div className="flex justify-start">
                      <CopyButton text={copyText(m.blocks)} label={t("ag.copyAnswer")} />
                    </div>
                  )}
                  {busy === null &&
                    canType &&
                    i === messages.length - 1 &&
                    m.blocks[m.blocks.length - 1]?.kind === "error" &&
                    (m.blocks[m.blocks.length - 1] as { resumable?: boolean }).resumable && (
                      <button
                        onClick={() => send(t("ag.continuePrompt"), null)}
                        data-testid="agent-continue"
                        className="rounded-md border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-medium text-blue-700 hover:bg-blue-100"
                      >
                        ↻ {t("ag.continue")}
                      </button>
                    )}
                </div>
              ),
            )}
            <div ref={bottom} />
          </div>

          <footer className="border-t border-gray-200 p-2">
            {/* Context sent with the message: the problem (always), then what the user focuses on in it */}
            <div className="mb-1.5 flex flex-wrap items-center gap-1.5 text-[11px]" data-testid="agent-context">
                <span className="shrink-0 text-gray-400">{t("ag.context")}:</span>
                <span
                  className="inline-flex max-w-[60%] items-center gap-1 truncate rounded-full bg-gray-100 px-2 py-0.5 font-medium text-gray-800"
                  title={t("agent.currentProblem")}
                  data-testid="agent-context-problem"
                >
                  {workProblem === "new" ? (
                    `✨ ${t("agent.modeNew")}`
                  ) : (
                    <span className="truncate">
                      <RichText text={problem.title} />
                      {wsLabel && <span className="font-normal text-gray-500"> · {wsLabel}</span>}
                    </span>
                  )}
                </span>
                {(shownSelection || pageSel || shot) && <span className="shrink-0 text-gray-400">{t("ag.focus")}:</span>}
                {shownSelection && (
                  <Pill
                    title={shownSelection.text}
                    onRemove={() => setSelection(null)}
                    removeLabel={t("agent.clearSel")}
                  >
                    <span className="font-mono">
                      {shownSelection.region ?? shownSelection.file}:{shownSelection.fromLine}
                      {shownSelection.toLine > shownSelection.fromLine && `–${shownSelection.toLine}`}
                    </span>
                  </Pill>
                )}
                {pageSel && (
                  <Pill title={pageSel.text} onRemove={() => setPageSel(null)} removeLabel={t("agent.clearSel")}>
                    <span data-testid="agent-page-selection">
                      {regionLabel(pageSel)}: “{pageSel.text.slice(0, 18)}
                      {pageSel.text.length > 18 ? "…" : ""}”
                    </span>
                  </Pill>
                )}
                {shot && (
                  <Pill
                    icon={
                      <img src={shot.image} alt="" className="h-4 w-6 rounded-sm object-cover ring-1 ring-blue-200" />
                    }
                    onClick={() => setShotOpen(true)}
                    onRemove={() => setShot(null)}
                    removeLabel={t("ag.removeScreenshot")}
                  >
                    {t("ag.screenshot")}
                  </Pill>
                )}
                {notice && <span className="text-amber-700">{notice}</span>}
                <span className="ml-auto shrink-0 rounded-full bg-amber-50 px-2 py-0.5 text-amber-700">
                  {t("agent.trial")}
                </span>
            </div>
            <div className="flex items-end gap-1.5 rounded-xl border border-gray-300 py-1.5 pr-1.5 pl-3 focus-within:border-blue-400">
              <textarea
                ref={input}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    send(draft);
                  }
                }}
                rows={1}
                disabled={!canType}
                placeholder={t("agent.placeholder")}
                className="max-h-36 min-w-0 flex-1 resize-none overflow-y-hidden bg-transparent py-1.5 text-sm leading-relaxed outline-none placeholder:text-gray-400 disabled:opacity-50"
              />
              <button
                onClick={capture}
                disabled={!canType || busy !== null}
                title={t("ag.capture")}
                aria-label={t("ag.capture")}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-800 disabled:opacity-40"
              >
                <svg
                  viewBox="0 0 24 24"
                  className="h-[18px] w-[18px]"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  aria-hidden="true"
                >
                  <path d="M4 8h3l2-2.5h6L17 8h3v11H4z" strokeLinejoin="round" />
                  <circle cx="12" cy="13.5" r="3.5" />
                </svg>
              </button>
              {busy ? (
                <button
                  onClick={() => void stop()}
                  title={t("agent.stop")}
                  aria-label={t("agent.stop")}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gray-800 text-xs text-white hover:bg-gray-700"
                >
                  ■
                </button>
              ) : (
                <button
                  onClick={() => send(draft)}
                  disabled={!canType || !draft.trim()}
                  title={t("agent.send")}
                  aria-label={t("agent.send")}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-sm text-white hover:bg-blue-700 disabled:opacity-40"
                >
                  ↑
                </button>
              )}
            </div>
            <div className="mt-1.5 flex items-center justify-between gap-2 text-[10px] text-gray-400">
              <span data-testid="agent-footnote">
                {busy ? t("ag.keepsRunning").replace("{server}", runsOn) : t("agent.hint")}
              </span>
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
        </>
      )}
      {shot && shotOpen && (
        <ScreenshotPreview
          image={shot.image}
          original={shot.original}
          onChange={(image) => {
            setShot({ ...shot, image });
            setNotice(pngBytes(image) > MAX_SCREENSHOT_BYTES ? t("ag.captureTooBig") : null);
          }}
          onClose={() => setShotOpen(false)}
        />
      )}
    </>
  );
}

/** A compact context chip above the chat input: "⌖ main.py:7–24 ✕". */
function Pill({
  children,
  icon,
  title,
  onClick,
  onRemove,
  removeLabel,
}: {
  children: React.ReactNode;
  icon?: React.ReactNode;
  title?: string;
  onClick?: () => void;
  onRemove: () => void;
  removeLabel: string;
}) {
  return (
    <span
      title={title}
      className="inline-flex max-w-full items-center gap-1 rounded-md border border-blue-200 bg-blue-50 py-0.5 pr-1 pl-1.5 text-blue-800"
    >
      {icon ?? <span className="shrink-0">⌖</span>}
      {onClick ? (
        <button onClick={onClick} className="truncate hover:underline">
          {children}
        </button>
      ) : (
        <span className="truncate">{children}</span>
      )}
      <button
        onClick={onRemove}
        title={removeLabel}
        aria-label={removeLabel}
        className="shrink-0 text-blue-400 hover:text-blue-700"
      >
        ✕
      </button>
    </span>
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

/** The agent asked a question (ask_user): one click answers it (several, then Send, when it allows more than one),
 *  or the user types their own. The answer goes out as the next message of the same conversation. */
function QuestionCard({
  block,
  active,
  onAnswer,
}: {
  block: Extract<Block, { kind: "question" }>;
  active: boolean;
  onAnswer: (answer: string) => void;
}) {
  const { t } = useLang();
  const [picked, setPicked] = useState<string[]>([]);
  const [own, setOwn] = useState("");
  const toggle = (o: string) => setPicked((p) => (p.includes(o) ? p.filter((x) => x !== o) : [...p, o]));
  const choose = (o: string) => (block.multi ? toggle(o) : onAnswer(o));
  return (
    <div className="rounded-lg border border-blue-200 bg-blue-50/60 p-3 text-sm" data-testid="agent-question">
      <p className="mb-2 font-medium text-gray-900">
        <RichText text={block.question} />
      </p>
      <div className="flex flex-col gap-1.5">
        {block.options.map((o) => {
          const on = picked.includes(o);
          return (
            <button
              key={o}
              type="button"
              disabled={!active}
              onClick={() => choose(o)}
              aria-pressed={block.multi ? on : undefined}
              className={`flex items-center gap-2 rounded-md border px-3 py-1.5 text-left text-xs transition disabled:cursor-default disabled:opacity-60 ${
                on ? "border-blue-500 bg-white text-blue-800" : "border-gray-200 bg-white text-gray-800 hover:border-blue-300"
              }`}
            >
              {block.multi && <span className={on ? "text-blue-600" : "text-gray-300"}>{on ? "☑" : "☐"}</span>}
              <RichText text={o} />
            </button>
          );
        })}
      </div>
      {active && (
        <div className="mt-2 flex items-center gap-1.5">
          <input
            value={own}
            onChange={(e) => setOwn(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && own.trim() && !e.nativeEvent.isComposing) onAnswer(own.trim());
            }}
            placeholder={t("ag.ownAnswer")}
            className="min-w-0 flex-1 rounded-md border border-gray-200 bg-white px-2 py-1 text-xs outline-none focus:border-blue-400"
          />
          {(block.multi || own.trim()) && (
            <button
              type="button"
              disabled={!own.trim() && !picked.length}
              onClick={() => onAnswer([...picked, ...(own.trim() ? [own.trim()] : [])].join("; "))}
              className="rounded-md bg-blue-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-40"
            >
              {t("agent.send")}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function AgentBlock({ block }: { block: Block }) {
  const { t } = useLang();
  if (block.kind === "question") return null; // drawn by QuestionCard, which can send the answer
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
        className="inline-flex items-center gap-1.5 rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 hover:bg-blue-100"
      >
        <DownloadIcon /> {t("agent.download")} ({Object.keys(block.files).length})
      </button>
    );
  return (
    <button
      onClick={() => window.dispatchEvent(new CustomEvent("tensorizer:agent-files", { detail: block }))}
      className="inline-flex items-center gap-1.5 rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 hover:bg-blue-100"
    >
      ⇩ {t("agent.applyFiles")} ({Object.keys(block.files).join(", ")})
    </button>
  );
}

/** Assistant text: Markdown, display maths ($$…$$ or \[…\]), inline maths ($…$ or \(…\)) and highlighted code. */
function AgentText({ text }: { text: string }) {
  const segments = useMemo(() => parseAnswer(text), [text]);
  return (
    <div className="space-y-2 text-gray-800">
      {segments.map((seg, i) =>
        seg.kind === "code" ? (
          <CodeView key={i} code={seg.code} lang={seg.lang} />
        ) : seg.kind === "math" ? (
          <div key={i} className="overflow-x-auto py-0.5">
            <Tex tex={seg.tex} display />
          </div>
        ) : (
          <MarkdownBlocks key={i} text={seg.text} />
        ),
      )}
    </div>
  );
}

/** A code block in an answer: Python highlighted (lezer, the parser CodeMirror uses), other languages plain. */
function CodeView({ code, lang }: { code: string; lang: string }) {
  const { t } = useLang();
  const lines = useMemo(() => {
    if (!isPython(lang)) return null;
    const out: { text: string; cls: string }[][] = [[]];
    try {
      highlightCode(
        code,
        pythonLanguage.parser.parse(code),
        classHighlighter,
        (text, cls) => out[out.length - 1].push({ text, cls }),
        () => out.push([]),
      );
    } catch {
      return null;
    }
    return out;
  }, [code, lang]);
  return (
    <div className="agent-code overflow-hidden rounded-md ring-1 ring-gray-200">
      <div className="flex items-center border-b border-gray-200 bg-gray-100/70 px-2 py-0.5 font-mono text-[10px] text-gray-500">
        <span>{lang || "code"}</span>
        <span className="ml-auto">
          <CopyButton text={code} label={t("ag.copyCode")} compact />
        </span>
      </div>
      <pre className="overflow-x-auto bg-gray-50 p-2 font-mono text-[12px] leading-relaxed">
        {lines
          ? lines.map((line, k) => (
              <div key={k}>
                {line.length ? line.map((tok, j) => (tok.cls ? <span key={j} className={tok.cls}>{tok.text}</span> : tok.text)) : "\u200b"}
              </div>
            ))
          : code}
      </pre>
    </div>
  );
}

function CopyButton({ text, label, compact = false }: { text: string; label: string; compact?: boolean }) {
  const [done, setDone] = useState(false);
  return (
    <button
      onClick={() =>
        void navigator.clipboard?.writeText(text).then(
          () => (setDone(true), setTimeout(() => setDone(false), 1200)),
          () => {},
        )
      }
      title={label}
      aria-label={label}
      data-testid="agent-copy"
      className={`${HEAD_BTN} ${compact ? "h-6 w-6" : "w-8"} ${done ? "text-green-600" : ""}`}
    >
      <HeadIcon
        d={done ? "M5 12.5l4.5 4.5L19 7.5" : "M9 9h10v10H9zM5 15V5h10"}
        small={compact}
      />
    </button>
  );
}

type Grouped = { kind: "steps"; steps: Extract<Block, { kind: "tool" }>[] } | { kind: "block"; block: Block };

/** Consecutive tool calls fold into one "steps" line (closed by default; open it to watch them live). */
function groupSteps(blocks: Block[]): Grouped[] {
  const out: Grouped[] = [];
  for (const b of blocks) {
    const last = out[out.length - 1];
    if (b.kind === "tool" && last?.kind === "steps") last.steps.push(b);
    else if (b.kind === "tool") out.push({ kind: "steps", steps: [b] });
    else out.push({ kind: "block", block: b });
  }
  return out;
}

const stepTarget = (b: Extract<Block, { kind: "tool" }>) =>
  String(b.input.file_path ?? b.input.pattern ?? b.input.path ?? b.input.command ?? "")
    .split(/[\\/]/)
    .pop();

/** What the agent is doing right now: its last unfinished tool call, e.g. "Read solve.py". */
function currentStep(blocks: Block[]): string | null {
  const last = [...blocks].reverse().find((b) => b.kind === "tool");
  if (!last || last.kind !== "tool" || last.ok !== undefined) return null;
  return `${`${TOOL_LABEL[last.name] ?? last.name} ${stepTarget(last)}`.trim()}…`;
}

function Steps({ steps, live }: { steps: Extract<Block, { kind: "tool" }>[]; live: boolean }) {
  const { t } = useLang();
  const failed = steps.filter((s) => s.ok === false).length;
  const pending = steps.some((s) => s.ok === undefined);
  const last = steps[steps.length - 1];
  return (
    <details className="rounded-md bg-gray-50 px-2 py-1 text-xs ring-1 ring-gray-100" data-testid="agent-steps">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-gray-600">
        <span className={pending && live ? "animate-pulse text-blue-500" : failed ? "text-red-500" : "text-green-600"}>●</span>
        <span>{t("ag.stepCount").replace("{n}", String(steps.length))}</span>
        <span className="truncate font-mono text-gray-400">
          {TOOL_LABEL[last.name] ?? last.name} {stepTarget(last)}
        </span>
        <span className="ml-auto text-gray-400">▾</span>
      </summary>
      <div className="mt-1 space-y-1 border-t border-gray-200 pt-1">
        {steps.map((b) => (
          <AgentBlock key={b.id} block={b} />
        ))}
      </div>
    </details>
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
