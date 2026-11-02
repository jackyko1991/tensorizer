import { useEffect } from "react";
import type { TestTally } from "../api";
import { useLang } from "../i18n";

/** The TDD light, opened: every test in test.py with its outcome, and the failure message for red ones. */
export default function TestsModal({ tests, onClose }: { tests: TestTally; onClose: () => void }) {
  const { t } = useLang();
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);

  const items = tests.items ?? [];
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-gray-900/40 p-4 text-left text-gray-900"
      onMouseDown={onClose}
    >
      <div
        role="dialog"
        aria-label={t("tdd.title")}
        data-testid="tests-modal"
        onMouseDown={(e) => e.stopPropagation()}
        className="mt-16 w-full max-w-lg overflow-hidden rounded-xl bg-white shadow-2xl"
      >
        <div className="flex items-center justify-between border-b border-gray-200 px-4 py-2.5">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <span className={`h-2.5 w-2.5 rounded-full ${tests.ok ? "bg-emerald-500" : "bg-red-500"}`} />
            {t("tdd.title")}
            <span className="font-normal text-gray-500">
              {tests.passed}/{tests.passed + tests.failed}
            </span>
          </h2>
          <button
            onClick={onClose}
            aria-label={t("plot.close")}
            className="text-xl leading-none text-gray-400 hover:text-gray-700"
          >
            ×
          </button>
        </div>
        <div className="max-h-[60vh] overflow-y-auto px-4 py-3 text-sm">
          {items.length === 0 ? (
            <p className="text-gray-500">{t("tdd.noItems")}</p>
          ) : (
            <ul className="divide-y divide-gray-100 rounded-lg border border-gray-200">
              {items.map((item) => {
                const good = item.outcome === "passed" || item.outcome === "xfail";
                const skipped = item.outcome === "skipped";
                return (
                  <li key={item.name} className="px-3 py-2 text-xs" data-outcome={item.outcome}>
                    <div className="flex items-center gap-2">
                      <span
                        className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] text-white ${
                          skipped ? "bg-gray-300" : good ? "bg-emerald-500" : "bg-red-500"
                        }`}
                      >
                        {skipped ? "–" : good ? "✓" : "✗"}
                      </span>
                      <span className="min-w-0 flex-1 truncate font-mono text-gray-800">{item.name}</span>
                      <span
                        className={`shrink-0 ${skipped ? "text-gray-400" : good ? "text-emerald-700" : "text-red-700"}`}
                      >
                        {item.outcome}
                      </span>
                    </div>
                    {item.message && (
                      <p className="mt-1 ml-6 font-mono text-[11px] break-words whitespace-pre-wrap text-red-700">
                        {item.message}
                      </p>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
          <p className="mt-2 text-[11px] text-gray-400">{t("tdd.hint")}</p>
        </div>
      </div>
    </div>
  );
}
