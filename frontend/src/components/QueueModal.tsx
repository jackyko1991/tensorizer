import { useEffect, useState } from "react";
import { abortMyRun, backendChoice, cancelMyRuns, queueDetails, type QueueDetails } from "../api";
import { useLang } from "../i18n";

/** The run queue: what is running and waiting on the current backend, refreshed every 2 s; cancel your own runs. */
export default function QueueModal({ onClose }: { onClose: () => void }) {
  const { t } = useLang();
  const [q, setQ] = useState<QueueDetails | null | "loading">("loading");
  const [busy, setBusy] = useState(false);
  const local = backendChoice() === "local";

  useEffect(() => {
    let alive = true;
    const load = () => queueDetails().then((d) => alive && setQ(d));
    void load();
    const timer = window.setInterval(load, 2000);
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => {
      alive = false;
      window.clearInterval(timer);
      window.removeEventListener("keydown", key);
    };
  }, [onClose]);

  const d = q === "loading" ? null : q;
  const mine = d?.runs.some((r) => r.mine) ?? false;
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-gray-900/40 p-4 text-left text-gray-900"
      onMouseDown={onClose}
    >
      <div
        role="dialog"
        aria-label={t("q.title")}
        data-testid="queue-modal"
        onMouseDown={(e) => e.stopPropagation()}
        className="mt-16 w-full max-w-md overflow-hidden rounded-xl bg-white shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-gray-200 px-4 py-2.5">
          <h2 className="text-sm font-semibold">
            {t("q.title")}{" "}
            <span className="font-normal text-gray-500">/ {t(local ? "be.tab.local" : "be.shared")}</span>
          </h2>
          <button
            onClick={onClose}
            aria-label={t("plot.close")}
            className="text-xl leading-none text-gray-400 hover:text-gray-700"
          >
            ×
          </button>
        </div>
        <div className="space-y-3 px-4 py-3 text-sm">
          {q === "loading" ? (
            <p className="text-gray-500">…</p>
          ) : !d ? (
            <p className="text-amber-700">{t("q.offline")}</p>
          ) : (
            <>
              <p className="text-xs text-gray-500" data-testid="queue-counts">
                {t("q.counts")
                  .replace("{running}", String(d.running))
                  .replace("{concurrent}", String(d.concurrent))
                  .replace("{waiting}", String(d.waiting))}
              </p>
              {d.runs.length === 0 ? (
                <p className="text-gray-500">{t("q.empty")}</p>
              ) : (
                <ul className="divide-y divide-gray-100 rounded-lg border border-gray-200">
                  {d.runs.map((r, i) => (
                    <li key={i} className="flex items-center gap-2 px-3 py-1.5 text-xs">
                      <span
                        className={`rounded-full px-2 py-0.5 font-medium ${
                          r.state === "running" ? "bg-green-50 text-green-700" : "bg-amber-50 text-amber-700"
                        }`}
                      >
                        {r.state === "running" ? t("q.running") : t("q.position").replace("{n}", String(r.position))}
                      </span>
                      <span className="min-w-0 flex-1 truncate font-mono text-gray-700">{r.problem}</span>
                      {r.mine && <span className="rounded bg-blue-50 px-1.5 text-blue-700">{t("q.you")}</span>}
                      <span className="w-12 text-right text-gray-500 tabular-nums">{r.seconds} s</span>
                    </li>
                  ))}
                </ul>
              )}
              <p className="text-[11px] text-gray-400">{t("q.note").replace("{s}", String(Math.round(d.max_run_s)))}</p>
              {mine && (
                <div className="flex justify-end">
                  <button
                    disabled={busy}
                    onClick={async () => {
                      setBusy(true);
                      await cancelMyRuns();
                      abortMyRun(); // the page stops showing it as running straight away
                      setQ(await queueDetails());
                      setBusy(false);
                    }}
                    className="rounded-md border border-red-200 px-3 py-1.5 text-xs text-red-700 hover:bg-red-50 disabled:opacity-50"
                    data-testid="queue-cancel"
                  >
                    {t("q.cancel")}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
