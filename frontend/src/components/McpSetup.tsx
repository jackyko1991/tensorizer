import { useEffect, useState } from "react";
import { mcpBackendKey, createConnector, disconnectMcp, mcpConnection, type Connector } from "../api";
import { useLang } from "../i18n";
import type { SetupAgent } from "./AgentPicker";
import ShellCommand from "./ShellCommand";
import { RichText } from "./Tex";

/**
 * The Agent Skills page's MCP tab, laid out like its Skills tab: what the agent gets (tools and rules), then the
 * numbered steps to connect it. The connection (token) belongs to the chosen backend and stays in this browser tab.
 */
export default function McpSetup({ agent, onAgent }: { agent: SetupAgent; onAgent: (agent: SetupAgent) => void }) {
  const { lang, t } = useLang();
  const storageKey = `tensorizer.mcp.${mcpBackendKey()}`;
  const [connector, setConnector] = useState<Connector | null>(() => {
    try {
      const c = JSON.parse(sessionStorage.getItem(storageKey) ?? "null");
      return c && typeof c.token === "string" && typeof c.url_with_token === "string" ? c : null;
    } catch { return null; }
  });
  const [lastSeen, setLastSeen] = useState<number | null>(null);
  const [where, setWhere] = useState<"local" | "web">("local");
  const [local, setLocal] = useState<LocalMethod>(() =>
    LOCAL_METHODS.some((m) => m.id === agent) ? (agent as LocalMethod) : "claude-code",
  );
  const [web, setWeb] = useState<WebMethod>("claude-ai");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  function remember(c: Connector | null) {
    setConnector(c);
    setLastSeen(null);
    try {
      if (c) sessionStorage.setItem(storageKey, JSON.stringify(c));
      else sessionStorage.removeItem(storageKey);
    } catch { /* The connection remains usable until this page closes. */ }
  }

  useEffect(() => {
    if (!connector) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function check() {
      setChecking(true);
      try {
        const status = await mcpConnection(connector!.token);
        if (!alive) return;
        if (!status) { remember(null); return; }
        setLastSeen(status.last_seen);
        setError(null);
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (alive) { setChecking(false); timer = setTimeout(check, 5000); }
      }
    }
    void check();
    return () => { alive = false; clearTimeout(timer); };
  }, [connector?.token]);

  async function connect() {
    setBusy(true);
    setError(null);
    try { remember(await createConnector(connector?.token ?? null)); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  async function disconnect() {
    if (!connector) return;
    setBusy(true);
    setError(null);
    try { await disconnectMcp(connector.token); remember(null); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  // A URL is quoted as one literal argument in bash and PowerShell.
  const url = connector ? `'${connector.url_with_token.replaceAll("'", "%27")}'` : "";
  const status = !connector ? t("ya.none") : lastSeen ? t("ya.contacted") : checking ? t("ya.checking") : t("ya.waiting");

  return (
    <div className="space-y-8" data-testid="mcp-setup">
      <div>
        <h2 className="text-xl font-semibold">{t("ya.mcpTitle")}</h2>
        <p className="mt-2 text-sm text-gray-600">{t("ya.mcpIntro")}</p>
      </div>

      <section className="rounded-xl border border-gray-200 bg-white p-6">
        <h2 className="text-lg font-semibold">{t("ya.mcpGives")}</h2>
        <div className="mt-3 grid gap-6 text-sm text-gray-700 md:grid-cols-2">
          <div>
            <h3 className="mb-2 text-xs font-semibold tracking-wide text-gray-500 uppercase">{t("ya.mcpTools")}</h3>
            <ul className="space-y-2">
              {(["ya.tool1", "ya.tool2", "ya.tool3", "ya.tool4"] as const).map((k) => (
                <li key={k} className="flex gap-2">
                  <span className="text-blue-600">›</span>
                  <span>
                    <RichText text={t(k)} />
                  </span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="mb-2 text-xs font-semibold tracking-wide text-gray-500 uppercase">{t("ag.rules")}</h3>
            <ul className="space-y-2">
              {(["ya.rule1", "ya.rule2", "ya.rule3", "ya.rule4"] as const).map((k) => (
                <li key={k} className="flex gap-2">
                  <span className="text-blue-600">✓</span>
                  {t(k)}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-gray-200 bg-white p-6" data-testid="mcp-connect">
        <h2 className="text-lg font-semibold">{t("ya.mcpConnect")}</h2>
        <ol className="mt-4 space-y-6 text-sm text-gray-700">
          <Step n={1} title={t("ya.step1")}>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={connect}
                disabled={busy}
                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
              >
                {busy ? t("ya.checking") : connector ? t("ag.cloudNew") : t("ag.cloudConnect")}
              </button>
              {connector && (
                <button
                  type="button"
                  onClick={disconnect}
                  disabled={busy}
                  className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium hover:bg-gray-50 disabled:opacity-50"
                >
                  {t("ya.disconnect")}
                </button>
              )}
              <span
                role="status"
                className={`ml-auto rounded-full px-3 py-1 text-xs font-medium ${connector ? (lastSeen ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-800") : "bg-gray-100 text-gray-600"}`}
              >
                {status}
                {lastSeen && <span className="ml-1 font-normal">· {new Date(lastSeen * 1000).toLocaleString(lang)}</span>}
              </span>
            </div>
            <p className="text-gray-600">{t("ya.scope")}</p>
            {error && (
              <p role="alert" className="text-red-700">
                {error}
              </p>
            )}
            {connector && (
              <>
                <p className="text-xs font-medium text-gray-500">{t("ag.cloudUrl")}</p>
                <ShellCommand text={connector.url_with_token} />
                <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-900">{t("ya.access")}</p>
                {connector.local && <p className="rounded-lg bg-blue-50 px-3 py-2 text-blue-900">{t("ag.cloudLocal")}</p>}
              </>
            )}
          </Step>

          <Step n={2} title={t("ya.step2")}>
            <p className="font-medium text-gray-800">{t("ya.where")}</p>
            <div className="grid gap-2 sm:grid-cols-2" role="radiogroup" aria-label={t("ya.where")} data-testid="mcp-where">
              {(["local", "web"] as const).map((w) => (
                <button
                  key={w}
                  type="button"
                  role="radio"
                  aria-checked={where === w}
                  onClick={() => setWhere(w)}
                  className={`rounded-lg border p-3 text-left ${where === w ? "border-blue-500 bg-blue-50/60 ring-1 ring-blue-500" : "border-gray-200 hover:border-blue-300"}`}
                >
                  <span className="block font-medium text-gray-900">{t(w === "local" ? "ya.whereLocal" : "ya.whereWeb")}</span>
                  <span className="block text-xs text-gray-500">{t(w === "local" ? "ya.whereLocalHint" : "ya.whereWebHint")}</span>
                </button>
              ))}
            </div>
            <MethodPicker
              methods={where === "local" ? LOCAL_METHODS : WEB_METHODS}
              value={where === "local" ? local : web}
              onChange={(m) => {
                if (where === "web") return setWeb(m as WebMethod);
                setLocal(m as LocalMethod);
                if (SHARED.includes(m as SetupAgent)) onAgent(m as SetupAgent); // the Skills tab shows the same agent
              }}
              label={t("ya.client")}
            />
            {where === "web" && (
              <>
                <p className="text-gray-600">{t("ya.webPublic")}</p>
                {connector?.local && (
                  <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-900" data-testid="mcp-web-local">
                    {t("ya.webLocalWarn")}
                  </p>
                )}
              </>
            )}
            {!connector ? (
              <p className="text-gray-500">{t("ya.createFirst")}</p>
            ) : where === "local" ? (
              <LocalSetup method={local} connector={connector} url={url} />
            ) : (
              <WebSetup method={web} connector={connector} />
            )}
          </Step>

          <Step n={3} title={t("ya.step3")}>
            <div className="rounded-lg bg-blue-50 p-4 text-blue-900">
              <p className="mb-2 font-semibold">{t("ag.try")}</p>
              <p>{t("ya.prompt")}</p>
            </div>
            <p className="text-xs text-gray-500">{t("ya.statusHint")}</p>
          </Step>
        </ol>
      </section>

    </div>
  );
}

/** One numbered step, with the same blue number as the skill's steps. */
function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-600 text-xs text-white">
        {n}
      </span>
      <div className="min-w-0 flex-1 space-y-3">
        <h3 className="font-semibold text-gray-900">{title}</h3>
        {children}
      </div>
    </li>
  );
}

type LocalMethod = "claude-code" | "codex" | "vscode" | "antigravity" | "opencode" | "muse-code" | "other";
type WebMethod = "claude-ai" | "chatgpt" | "grok" | "other-web";
// local methods that are also agents on the Skills tab
const SHARED: SetupAgent[] = ["claude-code", "codex", "opencode", "antigravity", "other"];
// Names of products are not translated; "other" is.
const LOCAL_METHODS: { id: LocalMethod; label: string }[] = [
  { id: "claude-code", label: "Claude Code" },
  { id: "codex", label: "Codex" },
  { id: "vscode", label: "VS Code Copilot" },
  { id: "antigravity", label: "Antigravity CLI" },
  { id: "opencode", label: "OpenCode" },
  { id: "muse-code", label: "Muse Code" },
  { id: "other", label: "" },
];
const WEB_METHODS: { id: WebMethod; label: string }[] = [
  { id: "claude-ai", label: "claude.ai" },
  { id: "chatgpt", label: "ChatGPT" },
  { id: "grok", label: "Grok" },
  { id: "other-web", label: "" },
];

function MethodPicker({
  methods,
  value,
  onChange,
  label,
}: {
  methods: { id: string; label: string }[];
  value: string;
  onChange: (id: string) => void;
  label: string;
}) {
  const { t } = useLang();
  return (
    <div className="flex flex-wrap gap-1 border-b border-gray-200" role="tablist" aria-label={label}>
      {methods.map((m) => (
        <button
          key={m.id}
          type="button"
          role="tab"
          aria-selected={value === m.id}
          onClick={() => onChange(m.id)}
          className={`-mb-px border-b-2 px-3 py-1.5 text-sm font-medium ${value === m.id ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500 hover:text-gray-800"}`}
        >
          {m.label || t(m.id === "other" ? "ya.other" : "ya.otherWeb")}
          {m.id === "claude-code" && <span className="ml-1 text-[10px] font-normal text-green-700">{t("ag.recommended")}</span>}
        </button>
      ))}
    </div>
  );
}

const json = (value: unknown) => JSON.stringify(value, null, 2);

/** An agent on this computer: a command, or a config file to paste (with the token in a header where possible). */
function LocalSetup({ method, connector, url }: { method: LocalMethod; connector: Connector; url: string }) {
  const { t } = useLang();
  const bearer = `Bearer ${connector.token}`;
  if (method === "claude-code")
    return (
      <>
        <p>{t("ya.ccHeader")}</p>
        <ShellCommand text={`claude mcp add --transport http tensorizer '${connector.url}' --header 'Authorization: ${bearer}'`} />
      </>
    );
  if (method === "codex")
    return (
      <>
        <p>{t("ag.cloudStep1")}</p>
        <ShellCommand text={`codex mcp add tensorizer --url ${url}`} />
      </>
    );
  if (method === "vscode")
    return (
      <>
        <p>{t("ya.vscode1")}</p>
        <ShellCommand
          text={json({
            inputs: [{ type: "promptString", id: "tensorizer-token", description: "Tensorizer token", password: true }],
            servers: {
              tensorizer: { type: "http", url: connector.url, headers: { Authorization: "Bearer ${input:tensorizer-token}" } },
            },
          })}
        />
        <p>{t("ya.vscode2")}</p>
        <ShellCommand text={connector.token} />
        <p className="text-gray-600">{t("ya.vscode3")}</p>
      </>
    );
  if (method === "antigravity")
    return (
      <>
        <p>{t("ya.agy1")}</p>
        <ShellCommand text={json({ mcpServers: { tensorizer: { serverUrl: connector.url, headers: { Authorization: bearer } } } })} />
        <p className="text-gray-600">{t("ya.agy2")}</p>
      </>
    );
  if (method === "muse-code")
    return (
      <>
        <p>{t("ya.muse1")}</p>
        <ShellCommand
          text={json({
            mcp_servers: { tensorizer: { transport: "streamable_http", url: connector.url, headers: { Authorization: bearer } } },
          })}
        />
        <p className="text-xs text-gray-500">{t("ya.muse2")}</p>
      </>
    );
  return (
    <>
      <p>{t("ag.cloudStep2")}</p>
      <p className="text-xs font-medium text-gray-500">{t("ya.serverUrl")}</p>
      <ShellCommand text={connector.url} />
      <p className="text-xs font-medium text-gray-500">{t("ag.cloudToken")}</p>
      <ShellCommand text={connector.token} />
    </>
  );
}

/** A web app: it reaches the backend from its own servers, so only the steps in that app are left. */
function WebSetup({ method, connector }: { method: WebMethod; connector: Connector }) {
  const { t } = useLang();
  const header = `Bearer ${connector.token}`;
  if (method === "claude-ai")
    return (
      <>
        <ol className="list-decimal space-y-1.5 pl-5">
          {(["ya.claude1", "ya.claude2", "ya.claude3", "ya.claude4"] as const).map((k) => (
            <li key={k}>{t(k)}</li>
          ))}
        </ol>
        <p className="text-xs font-medium text-gray-500">{t("ya.serverUrl")}</p>
        <ShellCommand text={connector.url} />
        <p className="text-xs font-medium text-gray-500">{t("ya.authHeader")}</p>
        <ShellCommand text={header} />
        <p className="text-xs text-gray-500">{t("ya.claudeTeam")}</p>
        <a href="https://claude.ai/customize/connectors" target="_blank" rel="noreferrer" className="inline-block font-medium text-blue-700 hover:underline">
          claude.ai ▸ Connectors ↗
        </a>
      </>
    );
  if (method === "chatgpt")
    return (
      <>
        <p>{t("ya.chatgpt")}</p>
        <ShellCommand text={connector.url_with_token} />
        <p className="text-gray-600">{t("ya.dot")}</p>
        <div className="flex flex-wrap gap-4">
          <a href="https://chatgpt.com/plugins" target="_blank" rel="noreferrer" className="font-medium text-blue-700 hover:underline">
            {t("ya.openChatGPT")} ↗
          </a>
          <a href="https://developers.openai.com/api/docs/guides/custom-mcp-server" target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
            {t("reference")} ↗
          </a>
        </div>
      </>
    );
  if (method === "grok")
    return (
      <>
        <p>{t("ya.grok1")}</p>
        <p>{t("ya.grok2")}</p>
        <ShellCommand text={connector.url_with_token} />
        <p className="text-xs text-gray-500">{t("ya.grokNote")}</p>
        <div className="flex flex-wrap gap-4">
          <a href="https://grok.com/connectors" target="_blank" rel="noreferrer" className="font-medium text-blue-700 hover:underline">
            grok.com ▸ Connectors ↗
          </a>
          <a href="https://docs.x.ai/grok/connectors" target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
            {t("reference")} ↗
          </a>
        </div>
      </>
    );
  return (
    <>
      <p>{t("ya.webOther")}</p>
      <p className="text-xs font-medium text-gray-500">{t("ya.serverUrl")}</p>
      <ShellCommand text={connector.url} />
      <p className="text-xs font-medium text-gray-500">{t("ya.authHeader")}</p>
      <ShellCommand text={header} />
      <p className="text-xs text-gray-500">{t("ya.museApp")}</p>
    </>
  );
}
