import { useState } from "react";
import {
  createConnector,
  openRouterSignIn,
  scanAgents,
  RunFailed,
  type AgentMode,
  type AgentStatus,
  type Connector,
  type FoundAgent,
  type LocalAgentId,
} from "../api";
import { useLang } from "../i18n";

/** The agent the panel talks to, remembered across visits (keys are not: they stay in the tab's sessionStorage). */
export interface AgentChoice {
  mode: AgentMode;
  local: LocalAgentId;
  localModel: string; // Claude Code (local): one of the status models
  cliModel: string; // Codex / OpenCode / Gemini: free text, "" = the CLI's default
  provider: string; // API: anthropic | openrouter
  apiModel: string;
}

/** Modes shown but not offered yet: only LOCAL (an agent on this computer) works for now. */
export const COMING_MODES: ReadonlySet<AgentChoice["mode"]> = new Set(["cloud", "api"]);

export const DEFAULT_CHOICE: AgentChoice = {
  mode: "local",
  local: "claude-code",
  localModel: "",
  cliModel: "",
  provider: "anthropic",
  apiModel: "",
};

const KEY_LINKS: Record<string, { href: string; placeholder: string }> = {
  anthropic: {
    href: "https://console.anthropic.com/settings/keys",
    placeholder: "sk-ant-…",
  },
  openrouter: {
    href: "https://openrouter.ai/settings/keys",
    placeholder: "sk-or-…",
  },
};

/** The ⚙ Agent form: Local | Cloud | API. */
export default function AgentSettings({
  status,
  choice,
  onChoice,
  found,
  onFound,
  apiKey,
  onKey,
  connector,
  onConnector,
  cloudToken,
}: {
  status: AgentStatus;
  choice: AgentChoice;
  onChoice: (c: AgentChoice) => void;
  found: FoundAgent[] | null;
  onFound: (f: FoundAgent[]) => void;
  apiKey: string | null;
  onKey: (key: string | null) => void;
  /** A connector just created: its URL and token are shown once. */
  connector: Connector | null;
  onConnector: (c: Connector) => void;
  cloudToken: string | null;
}) {
  const { t } = useLang();
  const [searching, setSearching] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const modes = status.modes ?? { local: true, cloud: true, api: true };
  const set = (patch: Partial<AgentChoice>) =>
    onChoice({ ...choice, ...patch });
  const providers =
    status.agents.find((a) => a.id === "claude-api")?.providers ?? [];
  const provider =
    providers.find((p) => p.id === choice.provider) ?? providers[0];
  const apiModels = provider?.models ?? status.models;

  async function search() {
    setSearching(true);
    setError(null);
    try {
      onFound(await scanAgents());
    } catch (e) {
      setError(e instanceof RunFailed ? e.message : String(e));
    } finally {
      setSearching(false);
    }
  }

  async function connect() {
    setConnecting(true);
    setError(null);
    try {
      onConnector(await createConnector(cloudToken));
    } catch (e) {
      setError(e instanceof RunFailed ? e.message : String(e));
    } finally {
      setConnecting(false);
    }
  }

  return (
    <div className="space-y-2 border-b border-gray-200 bg-gray-50 px-3 py-2.5 text-xs">
      <div
        role="tablist"
        className="grid grid-cols-3 gap-1 rounded-md bg-white p-0.5 ring-1 ring-gray-200"
      >
        {(["local", "cloud", "api"] as const).map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={choice.mode === m}
            aria-disabled={COMING_MODES.has(m) || undefined}
            disabled={COMING_MODES.has(m)}
            title={COMING_MODES.has(m) ? t("ag.coming") : undefined}
            onClick={() => set({ mode: m })}
            className={`rounded px-2 py-1 ${choice.mode === m ? "bg-blue-600 text-white" : COMING_MODES.has(m) ? "cursor-not-allowed text-gray-400" : "text-gray-700 hover:bg-gray-100"}`}
          >
            {t(
              m === "local"
                ? "ag.tabLocal"
                : m === "cloud"
                  ? "ag.tabCloud"
                  : "ag.tabApi",
            )}
            {COMING_MODES.has(m) && <span className="ml-1 text-[10px]">({t("ag.coming")})</span>}
          </button>
        ))}
      </div>

      {choice.mode === "local" &&
        (!modes.local ? (
          <p className="text-amber-700">{t("ag.localOnly")}</p>
        ) : (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <button
                onClick={search}
                disabled={searching}
                className="rounded-md bg-blue-600 px-2.5 py-1 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
              >
                🔍 {searching ? t("ag.searching") : t("ag.search")}
              </button>
            </div>
            {!found && (
              <p className="text-[11px] text-gray-500">{t("ag.searchHelp")}</p>
            )}
            {found && !found.some((a) => a.installed) && (
              <p className="text-amber-700">{t("ag.noneFound")}</p>
            )}
            {found && (
              <ul className="space-y-1" role="radiogroup">
                {found.map((a) => (
                  <li key={a.id}>
                    <label
                      className={`flex items-center gap-2 rounded-md border bg-white px-2 py-1.5 ${choice.local === a.id && a.installed ? "border-blue-400" : "border-gray-200"} ${a.installed ? "cursor-pointer" : "opacity-50"}`}
                      title={
                        a.experimental ? t("ag.experimentalHelp") : undefined
                      }
                    >
                      <input
                        type="radio"
                        name="local-agent"
                        checked={choice.local === a.id}
                        disabled={!a.installed}
                        onChange={() => set({ local: a.id })}
                      />
                      <span className="font-medium text-gray-800">
                        {a.label}
                      </span>
                      {a.version && (
                        <span className="font-mono text-[10px] text-gray-400">
                          {a.version}
                        </span>
                      )}
                      <span className="ml-auto flex flex-wrap justify-end gap-1">
                        <Chip tone={a.installed ? "green" : "gray"}>
                          {t(a.installed ? "ag.installed" : "ag.notInstalled")}
                        </Chip>
                        {a.installed && (
                          <Chip
                            tone={
                              a.logged_in
                                ? "green"
                                : a.logged_in === false
                                  ? "red"
                                  : "gray"
                            }
                          >
                            {t(
                              a.logged_in
                                ? "ag.loggedIn"
                                : a.logged_in === false
                                  ? "ag.notLoggedIn"
                                  : "ag.loginUnknown",
                            )}
                          </Chip>
                        )}
                        {a.experimental && (
                          <Chip tone="amber">{t("ag.experimental")}</Chip>
                        )}
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
            )}
            {found?.some((a) => a.id === choice.local && a.installed) && (
              <div className="flex items-center gap-2">
                <span className="w-16 shrink-0 font-medium text-gray-600">
                  {t("agent.model")}
                </span>
                {choice.local === "claude-code" ? (
                  <select
                    value={choice.localModel || status.models[0]}
                    onChange={(e) => set({ localModel: e.target.value })}
                    className="flex-1 rounded border border-gray-300 bg-white px-1.5 py-1"
                  >
                    {status.models.map((m) => (
                      <option key={m}>{m}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    value={choice.cliModel}
                    onChange={(e) => set({ cliModel: e.target.value.trim() })}
                    placeholder={t("ag.modelDefault")}
                    className="min-w-0 flex-1 rounded border border-gray-300 bg-white px-2 py-1 font-mono"
                  />
                )}
              </div>
            )}
          </div>
        ))}

      {choice.mode === "cloud" && (
        <div className="space-y-2">
          <p className="text-[11px] text-gray-600">{t("ag.cloudIntro")}</p>
          {connector ? (
            <div className="space-y-1.5">
              <CopyRow
                label={t("ag.cloudUrl")}
                value={connector.url_with_token}
              />
              <CopyRow label={t("ag.cloudToken")} value={connector.token} />
              <p className="text-[11px] font-medium text-amber-700">
                {t("ag.cloudOnce")}
              </p>
              <ol className="list-decimal space-y-0.5 pl-4 text-[11px] text-gray-600">
                <li>{t("ag.cloudStep1")}</li>
                <li>
                  {t("ag.cloudStep2")}{" "}
                  <span className="font-mono text-gray-500">
                    {connector.url}
                  </span>
                </li>
                <li>{t("ag.cloudStep3")}</li>
              </ol>
              {connector.local && (
                <p className="text-[11px] text-gray-500">
                  {t("ag.cloudLocal")}
                </p>
              )}
            </div>
          ) : (
            cloudToken && (
              <p className="text-green-700">✓ {t("ag.cloudWaiting")}</p>
            )
          )}
          <button
            onClick={connect}
            disabled={connecting}
            className={`rounded-md px-2.5 py-1 font-medium disabled:opacity-50 ${cloudToken ? "border border-gray-300 bg-white text-gray-700 hover:bg-gray-100" : "bg-blue-600 text-white hover:bg-blue-700"}`}
          >
            ☁ {cloudToken ? t("ag.cloudNew") : t("ag.cloudConnect")}
          </button>
        </div>
      )}

      {choice.mode === "api" && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="w-16 shrink-0 font-medium text-gray-600">
              {t("agent.provider")}
            </span>
            <select
              value={provider?.id}
              onChange={(e) => set({ provider: e.target.value, apiModel: "" })}
              className="flex-1 rounded border border-gray-300 bg-white px-1.5 py-1"
            >
              {providers.map((p) => (
                <option key={p.id} value={p.id} disabled={!p.available}>
                  {p.label}
                  {p.available ? "" : ` (${t("agent.soon")})`}
                </option>
              ))}
            </select>
          </div>
          {provider?.id === "openrouter" && (
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={() => openRouterSignIn()}
                className="rounded-md border border-gray-300 bg-white px-2.5 py-1 font-medium text-gray-800 hover:bg-gray-100"
                title={t("ag.orHelp")}
              >
                {t("ag.signInOR")} ↗
              </button>
              <span className="text-[11px] text-gray-400">{t("ag.or")}</span>
            </div>
          )}
          <div className="flex items-center gap-2">
            <span className="w-16 shrink-0 font-medium text-gray-600">
              {t("agent.apiKey")}
            </span>
            <input
              key={provider?.id}
              type="password"
              defaultValue={apiKey ?? ""}
              onChange={(e) => onKey(e.target.value.trim() || null)}
              placeholder={KEY_LINKS[provider?.id ?? "anthropic"]?.placeholder}
              autoComplete="off"
              className="min-w-0 flex-1 rounded border border-gray-300 bg-white px-2 py-1 font-mono"
            />
          </div>
          <div className="flex items-center gap-2">
            <span className="w-16 shrink-0 font-medium text-gray-600">
              {t("agent.model")}
            </span>
            <select
              value={
                apiModels.includes(choice.apiModel)
                  ? choice.apiModel
                  : apiModels[0]
              }
              onChange={(e) => set({ apiModel: e.target.value })}
              className="flex-1 rounded border border-gray-300 bg-white px-1.5 py-1"
            >
              {apiModels.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </div>
          <p className="text-[11px] text-gray-500">
            {t("agent.keyHelp")}{" "}
            <a
              href={KEY_LINKS[provider?.id ?? "anthropic"]?.href}
              target="_blank"
              rel="noreferrer"
              className="text-blue-700 hover:underline"
            >
              {t("agent.getKey")} ↗
            </a>
          </p>
        </div>
      )}
      {error && <p className="text-red-700">{error}</p>}
    </div>
  );
}

function Chip({
  tone,
  children,
}: {
  tone: "green" | "red" | "gray" | "amber";
  children: React.ReactNode;
}) {
  const tones = {
    green: "bg-green-50 text-green-700 ring-green-200",
    red: "bg-red-50 text-red-700 ring-red-200",
    gray: "bg-gray-50 text-gray-500 ring-gray-200",
    amber: "bg-amber-50 text-amber-700 ring-amber-200",
  };
  return (
    <span
      className={`rounded-full px-1.5 py-px text-[10px] ring-1 ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

function CopyRow({ label, value }: { label: string; value: string }) {
  const { t } = useLang();
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex items-center gap-2">
      <span className="w-16 shrink-0 font-medium text-gray-600">{label}</span>
      <input
        readOnly
        value={value}
        onFocus={(e) => e.target.select()}
        className="min-w-0 flex-1 rounded border border-gray-300 bg-white px-2 py-1 font-mono text-[11px]"
      />
      <button
        onClick={() =>
          navigator.clipboard
            ?.writeText(value)
            .then(
              () => (setCopied(true), setTimeout(() => setCopied(false), 1500)),
            )
        }
        className="shrink-0 rounded border border-gray-300 bg-white px-2 py-1 hover:bg-gray-100"
      >
        {copied ? `✓ ${t("ag.copied")}` : t("ag.copy")}
      </button>
    </div>
  );
}
