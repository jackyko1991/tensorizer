import { useEffect, useState } from "react";
import { useLang } from "../i18n";
import { RichText } from "./Tex";

export interface TocItem {
  /** id of the section element to scroll to */
  id: string;
  label: string;
  /** shown before the label, e.g. the step number */
  mark?: string | number;
}

/**
 * Sticky table of contents for the long Concept and Coding Skills pages (wide screens only). It highlights the
 * section currently in view, watched inside the page's own scroll container.
 */
export default function PageToc({ items, scrollRoot }: { items: TocItem[]; scrollRoot: string }) {
  const { t } = useLang();
  const [active, setActive] = useState(items[0]?.id);

  useEffect(() => {
    const root = document.getElementById(scrollRoot);
    const seen = new Map<string, boolean>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) seen.set(e.target.id, e.isIntersecting);
        const first = items.find((it) => seen.get(it.id));
        if (first) setActive(first.id);
      },
      { root, rootMargin: "0px 0px -60% 0px" },
    );
    for (const it of items) {
      const el = document.getElementById(it.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [items.map((it) => it.id).join(), scrollRoot]);

  return (
    <nav aria-label={t("toc")} className="sticky top-6 hidden w-56 shrink-0 self-start lg:block">
      <p className="mb-2 px-3 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("toc")}</p>
      <ol className="space-y-0.5 border-l border-gray-200 text-sm">
        {items.map((it) => (
          <li key={it.id}>
            <button
              onClick={() => document.getElementById(it.id)?.scrollIntoView({ behavior: "smooth", block: "start" })}
              className={`-ml-px flex w-full items-baseline gap-2 border-l-2 px-3 py-1 text-left ${
                active === it.id
                  ? "border-blue-600 font-medium text-blue-700"
                  : "border-transparent text-gray-600 hover:border-gray-300 hover:text-gray-900"
              }`}
            >
              {it.mark !== undefined && <span className="w-4 shrink-0 text-xs text-gray-400">{it.mark}</span>}
              <span>
                <RichText text={it.label} />
              </span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}
