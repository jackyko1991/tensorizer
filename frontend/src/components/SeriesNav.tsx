/** "🧭 Series · step k of n · role" with ◀ ▶ under a library problem's title, for each series it belongs to. */
import { useMemo } from "react";
import { buildCollections, ROLE_STYLE, seriesOf } from "../collections";
import { useLang } from "../i18n";
import type { Manifest } from "../types";

export default function SeriesNav({
  manifest,
  problemId,
  onSelect,
}: {
  manifest: Manifest;
  problemId: string;
  onSelect: (id: string) => void;
}) {
  const { lang, t } = useLang();
  const found = useMemo(() => seriesOf(buildCollections(manifest, lang, t), problemId), [manifest, lang, t, problemId]);
  if (!found.length) return null;
  return (
    <div className="mt-2 space-y-1">
      {found.map(({ collection: c, index }) => {
        const here = c.items[index];
        const prev = c.items[index - 1];
        const next = c.items[index + 1];
        return (
          <div
            key={c.id}
            data-testid="series-nav"
            className="flex flex-wrap items-center gap-2 rounded-md border border-gray-200 bg-gray-50 px-2 py-1 text-xs"
          >
            <span className="font-medium text-gray-800">🧭 {c.title}</span>
            <span className="text-gray-500">
              {t("coll.step")
                .replace("{k}", String(index + 1))
                .replace("{n}", String(c.items.length))}
            </span>
            {here.role && (
              <span className={`rounded px-1.5 py-0.5 text-[10px] ${ROLE_STYLE[here.role]}`}>
                {t(`srole.${here.role}` as Parameters<typeof t>[0])}
              </span>
            )}
            <span className="ml-auto flex items-center gap-1">
              <button
                disabled={!prev}
                onClick={() => prev && onSelect(prev.problem.id)}
                title={prev ? (prev.problem.i18n?.[lang]?.title ?? prev.problem.title) : undefined}
                data-testid="series-prev"
                className="rounded px-1.5 text-gray-600 hover:bg-white disabled:opacity-30"
              >
                ◀
              </button>
              <button
                disabled={!next}
                onClick={() => next && onSelect(next.problem.id)}
                title={next ? (next.problem.i18n?.[lang]?.title ?? next.problem.title) : undefined}
                data-testid="series-next"
                className="rounded px-1.5 text-gray-600 hover:bg-white disabled:opacity-30"
              >
                ▶
              </button>
            </span>
          </div>
        );
      })}
    </div>
  );
}
