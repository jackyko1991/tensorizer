/** The home of "My workspace" (#/workspace): its problems as cards, the ways to start a new one, recently viewed
 *  library problems, and export / import. */
import { useMemo, useRef, useState } from "react";
import { assetUrl } from "../api";
import { useLang, type Key } from "../i18n";
import type { Manifest, Problem, Workspace } from "../types";
import { plainText, RichText } from "./Tex";

interface Props {
  manifest: Manifest;
  workspace: Workspace | null;
  recent: Problem[];
  onOpen: (route: string) => void;
  onRenameWorkspace: () => void;
  onNewBlank: () => Promise<void>;
  onCopy: (libraryId: string) => Promise<void>;
  onAskAI: () => void;
  onRename: (id: string) => void;
  onDelete: (id: string) => void;
  onExport: () => void;
  onImport: (file: File) => void;
  onGitHub: () => void;
}

function ago(time: number | undefined, lang: string): string {
  if (!time) return "";
  const s = (time - Date.now()) / 1000;
  const fmt = new Intl.RelativeTimeFormat(lang, { numeric: "auto" });
  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ["day", 86400],
    ["hour", 3600],
    ["minute", 60],
  ];
  for (const [unit, size] of units) if (Math.abs(s) >= size) return fmt.format(Math.round(s / size), unit);
  return fmt.format(0, "minute");
}

export default function WorkspaceHome(props: Props) {
  const { manifest, workspace, recent } = props;
  const { lang, t } = useLang();
  const [picker, setPicker] = useState(false);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const problems = [...(workspace?.problems ?? [])].sort((a, b) => (b.updated ?? 0) - (a.updated ?? 0));
  const title = (p: Problem) => plainText(p.i18n?.[lang]?.title ?? p.title);
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    return manifest.problems
      .filter((p) => !q || title(p).toLowerCase().includes(q) || p.category.includes(q))
      .slice(0, 40);
  }, [query, manifest, lang]);

  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    try {
      await action();
    } finally {
      setBusy(false);
      setPicker(false);
    }
  };

  const actions = (
    <div className="grid gap-3 sm:grid-cols-3">
      <BigAction icon="📄" label={t("ws.blank")} hint={t("ws.blankHint")} onClick={() => run(props.onNewBlank)} disabled={busy} />
      <BigAction icon="📚" label={t("ws.fromLibrary")} hint={t("ws.fromLibraryHint")} onClick={() => setPicker((o) => !o)} disabled={busy} />
      <BigAction icon="✨" label={t("home.askAI")} hint={t("home.askAIHint")} onClick={props.onAskAI} />
    </div>
  );

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-6 sm:px-6" data-testid="ws-home">
      <header className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h2 className="truncate text-2xl font-semibold tracking-tight text-gray-900" data-testid="ws-home-name">
              {workspace?.name || t("ws.default")}
            </h2>
            <button
              onClick={props.onRenameWorkspace}
              title={t("home.renameWs")}
              aria-label={t("home.renameWs")}
              className="rounded px-1.5 text-sm text-gray-400 hover:bg-gray-100 hover:text-gray-700"
            >
              ✎
            </button>
          </div>
          <p className="text-sm text-gray-500">{t("home.subtitle")}</p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <button onClick={props.onExport} disabled={!problems.length} className="rounded-md border border-gray-300 px-2.5 py-1.5 hover:bg-gray-50 disabled:opacity-40">
            ⬇ {t("ws.export")}
          </button>
          <button onClick={() => file.current?.click()} className="rounded-md border border-gray-300 px-2.5 py-1.5 hover:bg-gray-50">
            ⬆ {t("ws.import")}
          </button>
          <button onClick={props.onGitHub} disabled={!problems.length} className="rounded-md border border-gray-300 px-2.5 py-1.5 hover:bg-gray-50 disabled:opacity-40">
            {t("gh.menu")}
          </button>
          <button disabled title={t("ws.cloudHint")} className="cursor-not-allowed rounded-md border border-gray-200 px-2.5 py-1.5 text-gray-400">
            ☁ {t("ws.cloud")} <span className="ml-1 rounded-full bg-amber-100 px-1.5 text-[10px] font-semibold text-amber-700">Pro</span>
          </button>
          <input
            ref={file}
            type="file"
            accept=".zip,application/zip"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) props.onImport(f);
              e.target.value = "";
            }}
          />
        </div>
      </header>

      {problems.length === 0 ? (
        <section className="space-y-4 rounded-2xl border border-dashed border-gray-300 bg-gray-50/60 p-6 text-center">
          <p className="text-sm text-gray-600">{t("home.empty")}</p>
          {actions}
        </section>
      ) : (
        <>
          {actions}
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("home.problems")}</h3>
            <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3" data-testid="ws-home-cards">
              {problems.map((p) => (
                <li key={p.id} className="group relative">
                  <button
                    onClick={() => props.onOpen(p.entry.id)}
                    className="flex w-full flex-col overflow-hidden rounded-xl border border-gray-200 bg-white text-left shadow-sm transition hover:border-blue-300 hover:shadow"
                    data-testid="ws-home-card"
                  >
                    <div className="flex h-28 w-full items-center justify-center bg-gray-50">
                      {p.entry.output ? (
                        <img
                          src={assetUrl(p.entry.output)}
                          alt=""
                          className="h-full w-full object-contain"
                          onError={(e) => (e.currentTarget.style.display = "none")}
                        />
                      ) : (
                        <span className="text-xs text-gray-400">{t("home.notRun")}</span>
                      )}
                    </div>
                    <div className="space-y-0.5 p-3">
                      <div className="truncate text-sm font-semibold text-gray-900">
                        <RichText text={p.entry.title || t("ws.untitled")} />
                      </div>
                      <div className="flex flex-wrap gap-x-2 text-[11px] text-gray-500">
                        {p.updated && <span>{t("home.edited").replace("{time}", ago(p.updated, lang))}</span>}
                        <span>{t("home.notebooks").replace("{n}", String(p.notebooks.length))}</span>
                        {!!p.entry.data?.length && <span>{t("home.dataFiles").replace("{n}", String(p.entry.data.length))}</span>}
                      </div>
                    </div>
                  </button>
                  <span className="absolute top-2 right-2 flex gap-1 opacity-100 [@media(hover:hover)]:opacity-0 group-hover:opacity-100 focus-within:opacity-100">
                    <button
                      onClick={() => props.onRename(p.id)}
                      title={t("ws.rename")}
                      aria-label={t("ws.rename")}
                      className="rounded bg-white/90 px-1.5 text-xs text-gray-600 shadow hover:bg-gray-100"
                    >
                      ✎
                    </button>
                    <button
                      onClick={() => props.onDelete(p.id)}
                      title={t("ws.delete")}
                      aria-label={t("ws.delete")}
                      className="rounded bg-white/90 px-1.5 text-xs text-gray-600 shadow hover:bg-red-50 hover:text-red-600"
                    >
                      🗑
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}

      {picker && (
        <section className="rounded-xl border border-blue-100 bg-blue-50/40 p-3" data-testid="ws-home-picker">
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("home.search")}
            aria-label={t("home.search")}
            className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:border-blue-400 focus:outline-none"
          />
          <ul className="mt-2 max-h-64 overflow-y-auto">
            {matches.length === 0 && <li className="px-2 py-1 text-xs text-gray-500">{t("home.noResults")}</li>}
            {matches.map((p) => (
              <li key={p.id}>
                <button
                  disabled={busy}
                  onClick={() => run(() => props.onCopy(p.id))}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-white disabled:opacity-50"
                >
                  <span className="min-w-0 flex-1 truncate">{title(p)}</span>
                  <span className="shrink-0 text-[11px] text-gray-400">{t(`cat.${p.category}` as Key) ?? p.category}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {recent.length > 0 && (
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("home.recent")}</h3>
          <ul className="flex gap-2 overflow-x-auto pb-1" data-testid="ws-home-recent">
            {recent.map((p) => (
              <li key={p.id} className="shrink-0">
                <button
                  onClick={() => props.onOpen(p.id)}
                  className="w-44 rounded-lg border border-gray-200 bg-white px-3 py-2 text-left hover:border-blue-300"
                >
                  <span className="block truncate text-sm font-medium text-gray-800">{title(p)}</span>
                  <span className="block truncate text-[11px] text-gray-500">{t(`cat.${p.category}` as Key) ?? p.category}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function BigAction(props: { icon: string; label: string; hint: string; onClick: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={props.onClick}
      disabled={props.disabled}
      className="flex items-start gap-3 rounded-xl border border-gray-200 bg-white p-4 text-left shadow-sm transition hover:border-blue-300 hover:shadow disabled:opacity-50"
    >
      <span className="text-2xl">{props.icon}</span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold text-gray-900">{props.label}</span>
        <span className="block text-xs leading-snug text-gray-500">{props.hint}</span>
      </span>
    </button>
  );
}
