/** "Export to GitHub…": token + repository name, an explicit confirm step, then one commit pushed from the browser.
 *  The token is kept in sessionStorage only and sent only to api.github.com (see github.ts). */
import { useState } from "react";
import { GitHubError, pushToGitHub, TOKEN_URL, validRepoName, type Progress } from "../github";
import { useLang, type Key } from "../i18n";

const TOKEN_STORE = "tensorizer.github.token";
const load = () => {
  try {
    return sessionStorage.getItem(TOKEN_STORE) ?? "";
  } catch {
    return "";
  }
};
const save = (token: string) => {
  try {
    sessionStorage.setItem(TOKEN_STORE, token);
  } catch {
    // blocked storage: the token lasts while the dialog is open
  }
};

export default function GitHubDialog({
  defaultName,
  files,
  onClose,
}: {
  defaultName: string;
  /** Builds the files to push (the same content as the .zip export). */
  files: () => Promise<Record<string, Uint8Array>>;
  onClose: () => void;
}) {
  const { t } = useLang();
  const [token, setToken] = useState(load);
  const [repo, setRepo] = useState(defaultName);
  const [priv, setPriv] = useState(true);
  const [stage, setStage] = useState<"form" | "confirm" | "pushing" | "done">("form");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const visibility = t(priv ? "gh.privateWord" : "gh.public");

  async function push() {
    setStage("pushing");
    setError(null);
    try {
      save(token.trim());
      const link = await pushToGitHub({
        token: token.trim(),
        repo,
        private: priv,
        files: await files(),
        message: "Export from Tensorizer",
        onProgress: setProgress,
      });
      setUrl(link);
      setStage("done");
    } catch (e) {
      const code = e instanceof GitHubError ? e.code : "other";
      const message = e instanceof Error ? e.message : String(e);
      setError(t(`gh.err.${code}` as Key).replace("{message}", message));
      setStage("form");
    }
  }

  const step =
    progress?.step === "blobs"
      ? t("gh.stepBlobs").replace("{n}", String(progress.done)).replace("{total}", String(progress.total))
      : progress
        ? t(progress.step === "user" ? "gh.stepUser" : progress.step === "repo" ? "gh.stepRepo" : "gh.stepCommit")
        : "";
  const input = "w-full rounded-md border border-gray-300 px-2.5 py-1.5 text-sm focus:border-blue-400 focus:outline-none";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-label={t("gh.title")}
        data-testid="gh-dialog"
        className="w-full max-w-md space-y-3 rounded-xl bg-white p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2">
          <h2 className="text-base font-semibold text-gray-900">{t("gh.title")}</h2>
          <button onClick={onClose} aria-label="close" className="ml-auto rounded px-2 text-gray-400 hover:bg-gray-100">
            ✕
          </button>
        </div>

        {stage === "form" && (
          <>
            <p className="text-sm text-gray-600">{t("gh.intro")}</p>
            <label className="block space-y-1 text-xs font-medium text-gray-700">
              <span>{t("gh.token")}</span>
              <input
                type="password"
                autoComplete="off"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="github_pat_…"
                className={`${input} font-mono`}
                data-testid="gh-token"
              />
            </label>
            <p className="text-[11px] leading-snug text-gray-500">
              {t("gh.tokenHelp")}{" "}
              <a href={TOKEN_URL} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
                {t("gh.createToken")}
              </a>
            </p>
            <label className="block space-y-1 text-xs font-medium text-gray-700">
              <span>{t("gh.repo")}</span>
              <input value={repo} onChange={(e) => setRepo(e.target.value.trim())} className={input} data-testid="gh-repo" />
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={priv} onChange={(e) => setPriv(e.target.checked)} />
              {t("gh.private")}
            </label>
            {error && <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>}
            <div className="flex justify-end gap-2 pt-1">
              <button onClick={onClose} className="rounded-md px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100">
                {t("ws.cancel")}
              </button>
              <button
                onClick={() => setStage("confirm")}
                disabled={!token.trim() || !validRepoName(repo)}
                className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
              >
                {t("gh.next")}
              </button>
            </div>
          </>
        )}

        {stage === "confirm" && (
          <>
            <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900" data-testid="gh-confirm">
              {t("gh.confirm").replace("{repo}", repo).replace("{visibility}", visibility)}
            </p>
            <div className="flex justify-end gap-2">
              <button onClick={() => setStage("form")} className="rounded-md px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100">
                {t("gh.back")}
              </button>
              <button onClick={push} className="rounded-md bg-gray-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-gray-700">
                {t("gh.push")}
              </button>
            </div>
          </>
        )}

        {stage === "pushing" && <p className="animate-pulse text-sm text-blue-700">{step}</p>}

        {stage === "done" && url && (
          <div className="space-y-2">
            <p className="text-sm text-green-700">✓ {t("gh.done")}</p>
            <a href={url} target="_blank" rel="noreferrer" className="text-sm font-medium text-blue-700 hover:underline">
              {t("gh.openRepo")}
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
