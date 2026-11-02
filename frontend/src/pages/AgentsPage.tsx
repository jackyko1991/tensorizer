import { markdown } from "@codemirror/lang-markdown";
import { python } from "@codemirror/lang-python";
import { yaml } from "@codemirror/lang-yaml";
import { LanguageDescription } from "@codemirror/language";
import { EditorView } from "@codemirror/view";
import CodeMirror from "@uiw/react-codemirror";
import { useEffect, useState } from "react";
import { generatedUrl } from "../api";
import { useLang, type Key } from "../i18n";
import DownloadIcon from "../components/DownloadIcon";
import ShellCommand from "../components/ShellCommand";

/** Where the site serves the generated skill files (scripts/agent_skill.py), for commands run on any machine. */
const PUBLIC = "https://jackyko1991.github.io/tensorizer/generated/agent/";
const REPO = "jackyko1991/tensorizer";

// SKILL.md shown read-only: Markdown, with the python / yaml / bash code blocks highlighted too
const PREVIEW_EXTENSIONS = [
  markdown({
    codeLanguages: [
      LanguageDescription.of({ name: "python", alias: ["py"], support: python() }),
      LanguageDescription.of({ name: "yaml", alias: ["yml"], support: yaml() }),
    ],
  }),
  EditorView.lineWrapping,
];

type Agent = "claude-code" | "claude-desktop" | "codex" | "opencode" | "gemini" | "other";
type Os = "unix" | "windows";

const AGENTS: { id: Agent; label: string }[] = [
  { id: "claude-code", label: "Claude Code" },
  { id: "claude-desktop", label: "Claude Desktop / claude.ai" },
  { id: "codex", label: "Codex" },
  { id: "opencode", label: "OpenCode" },
  { id: "gemini", label: "Gemini CLI" },
  { id: "other", label: "Other" },
];

/** Download `file` from the site into `dest` (a path relative to the current folder or the home folder). */
function fetchTo(os: Os, file: string, dest: string): string {
  const url = PUBLIC + file;
  if (os === "windows") {
    const win = dest.replace(/^~\//, "$HOME\\").replace(/\//g, "\\");
    const dir = win.includes("\\") ? win.slice(0, win.lastIndexOf("\\")) : "";
    return `${dir ? `New-Item -ItemType Directory -Force "${dir}" | Out-Null; ` : ""}irm ${url} -OutFile "${win}"`;
  }
  const dir = dest.includes("/") ? dest.slice(0, dest.lastIndexOf("/")) : "";
  return `${dir ? `mkdir -p ${dir} && ` : ""}curl -fsSL ${url} -o ${dest}`;
}

/**
 * Agent Skills: the Concept and Coding Skills pages as one skill for the visitor's own AI coding agent, with install
 * steps per agent (Claude Code plugin in one line, a zip for Claude Desktop / claude.ai, AGENTS.md / GEMINI.md).
 */
export default function AgentsPage() {
  const { t } = useLang();
  const [agent, setAgent] = useState<Agent>("claude-code");
  const [os, setOs] = useState<Os>(() => (navigator.userAgent.toLowerCase().includes("win") ? "windows" : "unix"));
  const [skill, setSkill] = useState("");
  useEffect(() => {
    fetch(generatedUrl("agent/SKILL.md"))
      .then((r) => (r.ok ? r.text() : ""))
      .then(setSkill, () => setSkill(""));
  }, []);

  const osToggle = (
    <div className="flex gap-1 text-xs">
      {(["unix", "windows"] as const).map((o) => (
        <button
          key={o}
          onClick={() => setOs(o)}
          className={`rounded px-2 py-0.5 ${os === o ? "bg-blue-50 font-medium text-blue-700" : "text-gray-500 hover:text-gray-800"}`}
        >
          {o === "unix" ? "macOS / Linux" : "Windows (PowerShell)"}
        </button>
      ))}
    </div>
  );

  return (
    <div className="h-full overflow-y-auto bg-gray-50/60">
      <article className="mx-auto max-w-4xl space-y-8 px-5 py-10">
        <header>
          <p className="text-sm font-medium text-blue-700">{t("ag.kicker")}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{t("ag.title")}</h1>
          <p className="mt-3 text-gray-600">{t("ag.intro")}</p>
        </header>

        <section className="rounded-xl border border-gray-200 bg-white p-6">
          <h2 className="text-lg font-semibold">{t("ag.teaches")}</h2>
          <div className="mt-3 grid gap-6 text-sm text-gray-700 md:grid-cols-2">
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("ag.steps")}</h3>
              <ol className="space-y-2">
                {(["ag.s1", "ag.s2", "ag.s3", "ag.s4", "ag.s5"] as Key[]).map((k, i) => (
                  <li key={k} className="flex gap-2.5">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-600 text-[11px] text-white">
                      {i}
                    </span>
                    {t(k)}
                  </li>
                ))}
              </ol>
            </div>
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("ag.rules")}</h3>
              <ul className="space-y-2">
                {(["ag.r1", "ag.r2", "ag.r3", "ag.r4"] as Key[]).map((k) => (
                  <li key={k} className="flex gap-2">
                    <span className="text-blue-600">✓</span>
                    {t(k)}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section className="rounded-xl border border-gray-200 bg-white p-6" data-testid="agent-install">
          <h2 className="text-lg font-semibold">{t("ag.install")}</h2>
          <div className="mt-3 flex flex-wrap gap-1 border-b border-gray-200" role="tablist">
            {AGENTS.map((a) => (
              <button
                key={a.id}
                role="tab"
                aria-selected={agent === a.id}
                onClick={() => setAgent(a.id)}
                className={`-mb-px border-b-2 px-3 py-1.5 text-sm font-medium ${
                  agent === a.id
                    ? "border-blue-600 text-blue-700"
                    : "border-transparent text-gray-500 hover:text-gray-800"
                }`}
              >
                {a.label}
                {a.id === "claude-code" && (
                  <span className="ml-1 text-[10px] font-normal text-green-700">{t("ag.recommended")}</span>
                )}
              </button>
            ))}
          </div>

          <div className="mt-4 space-y-3 text-sm text-gray-700">
            {agent === "claude-code" && (
              <>
                <p>{t("ag.cc.one")}</p>
                <ShellCommand
                  text={`claude plugin marketplace add ${REPO} && claude plugin install tensorizer@tensorizer`}
                />
                <p>{t("ag.cc.inside")}</p>
                <ShellCommand text={`/plugin marketplace add ${REPO}\n/plugin install tensorizer@tensorizer`} />
                <div className="flex items-center justify-between gap-3 pt-1">
                  <p>{t("ag.cc.file")}</p>
                  {osToggle}
                </div>
                <ShellCommand text={fetchTo(os, "SKILL.md", "~/.claude/skills/tensorizer/SKILL.md")} />
              </>
            )}
            {agent === "claude-desktop" && (
              <>
                <ol className="list-decimal space-y-1 pl-5">
                  {(["ag.d1", "ag.d2", "ag.d3", "ag.d4"] as Key[]).map((k) => (
                    <li key={k}>{t(k)}</li>
                  ))}
                </ol>
                <p className="text-xs text-gray-500">{t("ag.dnote")}</p>
                <a
                  href={generatedUrl("agent/tensorizer-skill.zip")}
                  download="tensorizer-skill.zip"
                  className="inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
                  data-testid="download-zip"
                >
                  <DownloadIcon /> {t("ag.download")} tensorizer-skill.zip
                </a>
              </>
            )}
            {(agent === "codex" || agent === "opencode" || agent === "gemini") && (
              <>
                <div className="flex items-center justify-between gap-3">
                  <p>{t("ag.project")}</p>
                  {osToggle}
                </div>
                <ShellCommand
                  text={fetchTo(
                    os,
                    agent === "gemini" ? "GEMINI.md" : "AGENTS.md",
                    agent === "gemini" ? "GEMINI.md" : "AGENTS.md",
                  )}
                />
                <p>{t("ag.global")}</p>
                <ShellCommand
                  text={fetchTo(
                    os,
                    agent === "gemini" ? "GEMINI.md" : "AGENTS.md",
                    agent === "codex"
                      ? "~/.codex/AGENTS.md"
                      : agent === "opencode"
                        ? "~/.config/opencode/AGENTS.md"
                        : "~/.gemini/GEMINI.md",
                  )}
                />
              </>
            )}
            {agent === "other" && (
              <>
                <p>{t("ag.other")}</p>
                <div className="flex flex-wrap gap-2">
                  {["AGENTS.md", "SKILL.md"].map((f) => (
                    <a
                      key={f}
                      href={generatedUrl(`agent/${f}`)}
                      download={f}
                      className="inline-flex items-center gap-1.5 rounded-md border border-gray-300 px-3 py-1.5 text-xs hover:bg-gray-50"
                    >
                      <DownloadIcon /> {f}
                    </a>
                  ))}
                </div>
              </>
            )}
          </div>

          <div className="mt-5 rounded-lg bg-blue-50/60 px-4 py-3 text-sm">
            <span className="text-gray-500">{t("ag.try")}: </span>
            <q className="text-blue-900">{t("ag.prompt")}</q>
          </div>
        </section>

        <details className="rounded-xl border border-gray-200 bg-white">
          <summary className="cursor-pointer px-6 py-3 font-medium text-gray-700">{t("ag.preview")}</summary>
          <div className="max-h-[32rem] overflow-auto border-t border-gray-100 text-xs" data-testid="skill-preview">
            <CodeMirror
              value={skill}
              extensions={PREVIEW_EXTENSIONS}
              editable={false}
              theme="light"
              basicSetup={{ foldGutter: false, highlightActiveLine: false, highlightActiveLineGutter: false }}
            />
          </div>
          <p className="border-t border-gray-100 px-6 py-2 text-xs text-gray-500">
            <a
              href={`https://github.com/${REPO}/tree/main/plugins/tensorizer`}
              target="_blank"
              rel="noreferrer"
              className="text-blue-700 hover:underline"
            >
              {t("ag.source")} ↗
            </a>
          </p>
        </details>
      </article>
    </div>
  );
}
