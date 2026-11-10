import { useLang } from "../i18n";

export type SetupAgent = "claude-code" | "claude-desktop" | "codex" | "opencode" | "antigravity" | "other";

const AGENTS: { id: SetupAgent; label: string }[] = [
  { id: "claude-code", label: "Claude Code" },
  { id: "claude-desktop", label: "Claude Desktop / claude.ai" },
  { id: "codex", label: "Codex" },
  { id: "opencode", label: "OpenCode" },
  { id: "antigravity", label: "Antigravity CLI" }, // replaced Gemini CLI (June 2026)
  { id: "other", label: "Other" },
];

/** One agent picker for both setup pages, with the same options, order and appearance. */
export default function AgentPicker({ agent, onAgent }: {
  agent: SetupAgent;
  onAgent: (agent: SetupAgent) => void;
}) {
  const { t } = useLang();
  return <div className="mt-3 flex flex-wrap gap-1 border-b border-gray-200" role="tablist" aria-label={t("ya.client")}>
    {AGENTS.map((a) => <button
      key={a.id}
      type="button"
      role="tab"
      aria-selected={agent === a.id}
      onClick={() => onAgent(a.id)}
      className={`-mb-px border-b-2 px-3 py-1.5 text-sm font-medium ${agent === a.id
        ? "border-blue-600 text-blue-700"
        : "border-transparent text-gray-500 hover:text-gray-800"}`}
    >
      {a.label}
      {a.id === "claude-code" && <span className="ml-1 text-[10px] font-normal text-green-700">{t("ag.recommended")}</span>}
    </button>)}
  </div>;
}
