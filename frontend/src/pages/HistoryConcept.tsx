import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { plainText, RichText } from "../components/Tex";
import { usePageText, type PageKey } from "../content/pages";
import { HISTORY_CARDS, HISTORY_PARTS, type Drawing, type HistoryCard, type Part } from "../historyCards";
import { formatPageRoute } from "../pageRoute";

/**
 * Concept tab "Tensor history": a trilogy (Maths, Physics, AI and silicon) told as a vertical timeline. Cards
 * alternate left and right of a central line on wide screens (≥ lg) and sit to the right of a line at the left
 * on phones. A sticky navigation (sidebar on wide screens, folding bar on phones) highlights the card in view.
 * Deep links: #/concept/history/<card id> (the scroll is done by ConceptPage, element id history-<card id>).
 * Some cards hide "💡 Did you know?" notes (card.funFacts): a 💡 button next to the year opens one card's note, and a
 * switch at the bottom of the navigation opens or closes them all.
 */

const SCROLL_ROOT = "concept-scroll";
const elId = (id: string) => `history-${id}`;
const PART_STYLE: Record<Part, { text: string; dot: string; badge: string; border: string; band: string }> = {
  1: { text: "text-amber-800", dot: "bg-amber-500", badge: "bg-amber-50 text-amber-800 ring-amber-200", border: "border-amber-200", band: "border-amber-200 bg-amber-50 text-amber-800" },
  2: { text: "text-emerald-800", dot: "bg-emerald-600", badge: "bg-emerald-50 text-emerald-800 ring-emerald-200", border: "border-emerald-200", band: "border-emerald-200 bg-emerald-50 text-emerald-800" },
  3: { text: "text-blue-800", dot: "bg-blue-600", badge: "bg-blue-50 text-blue-800 ring-blue-200", border: "border-blue-200", band: "border-blue-200 bg-blue-50 text-blue-800" },
};

const reducedMotion = () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

function goTo(id: string) {
  document.getElementById(elId(id))?.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  // keep the address shareable without a new history entry (no hashchange: the page does not re-render)
  history.replaceState(null, "", `#${formatPageRoute("concept", "history", id)}`);
}

/** The card whose top last passed a line near the top of the viewport (robust with the interlocked columns, where
 *  the previous card is often still on screen), and the set of cards that have been on screen (for the reveal). */
function useScrollSpy(ids: string[], initial: string | null) {
  const [active, setActive] = useState(initial && ids.includes(initial) ? initial : ids[0]);
  const [seen, setSeen] = useState<Set<string>>(() => new Set());
  const clickedAt = useRef(0);
  useEffect(() => {
    const el = document.getElementById(SCROLL_ROOT);
    // the Concept page scrolls itself on xl screens; below that the window scrolls
    const root = el && getComputedStyle(el).overflowY !== "visible" ? el : null;
    const target: HTMLElement | Window = root ?? window;
    let frame = 0;
    const update = () => {
      frame = 0;
      if (Date.now() - clickedAt.current < 1200) return;
      const sc = root ?? document.scrollingElement ?? document.documentElement;
      const line = (root ? root.getBoundingClientRect().top : 0) + 140;
      let pick = ids[0];
      for (const id of ids) {
        const top = document.getElementById(elId(id))?.getBoundingClientRect().top ?? Infinity;
        if (top <= line) pick = id;
      }
      if (sc.scrollTop > 0 && sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 4) pick = ids[ids.length - 1];
      setActive(pick);
    };
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    target.addEventListener("scroll", onScroll, { passive: true });
    const timer = setTimeout(update, 300);
    const reveal = new IntersectionObserver(
      (entries) => {
        const shown = entries.filter((e) => e.isIntersecting).map((e) => e.target.id.slice("history-".length));
        if (shown.length) setSeen((s) => new Set([...s, ...shown]));
      },
      { root, rootMargin: "0px 0px -8% 0px" },
    );
    for (const id of ids) {
      const card = document.getElementById(elId(id));
      if (card) reveal.observe(card);
    }
    return () => {
      target.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(frame);
      clearTimeout(timer);
      reveal.disconnect();
    };
  }, [ids.join()]);
  const jump = (id: string) => {
    clickedAt.current = Date.now();
    setActive(id);
    setSeen((s) => new Set([...s, id]));
    goTo(id);
  };
  return { active, seen, jump };
}

/** Wide screens (≥ lg): interlocked columns. Each card starts halfway down the previous one (on the other side),
 *  but never before the previous same-side card has ended, and a part's first card leaves room above for the part
 *  badge on the line. Measured with a ResizeObserver; returns null below lg (plain single column). */
const LG = "(min-width: 1024px)";
const GAP = 24; // px between consecutive cards on the same side
const BADGE = 140; // px reserved above a part's first card for its badge
function useInterlock(listRef: React.RefObject<HTMLOListElement | null>, ids: string[], partStarts: Set<string>) {
  const [layout, setLayout] = useState<{ tops: Record<string, number>; height: number } | null>(null);
  useLayoutEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const mq = window.matchMedia(LG);
    let frame = 0;
    const compute = () => {
      frame = 0;
      if (!mq.matches) return setLayout(null);
      const tops: Record<string, number> = {};
      const bottoms: number[] = [];
      let prevTop = 0;
      let prevH = 0;
      ids.forEach((id, k) => {
        const h = list.querySelector<HTMLElement>(`[data-history-card="${id}"]`)?.offsetHeight ?? 0;
        let top = k === 0 ? BADGE : prevTop + prevH / 2;
        if (k >= 2) top = Math.max(top, bottoms[k - 2] + GAP);
        // room for the part badge between the previous card's marker (top + 50 px) and this card's year label
        if (k > 0 && partStarts.has(id)) top = Math.max(top, prevTop + 50 + BADGE);
        tops[id] = top;
        bottoms[k] = top + h;
        prevTop = top;
        prevH = h;
      });
      setLayout({ tops, height: Math.max(0, ...bottoms) });
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(compute);
    };
    const ro = new ResizeObserver(schedule);
    ro.observe(list);
    list.querySelectorAll("[data-history-card]").forEach((el) => ro.observe(el));
    mq.addEventListener("change", schedule);
    compute();
    return () => {
      ro.disconnect();
      mq.removeEventListener("change", schedule);
      cancelAnimationFrame(frame);
    };
  }, [ids.join()]);
  return layout;
}

export default function HistoryConcept({ section }: { section: string | null }) {
  const T = usePageText();
  const ids = useMemo(() => HISTORY_CARDS.map((c) => c.id), []);
  const { active, seen, jump } = useScrollSpy(ids, section);
  const [open, setOpen] = useState(false);
  // cards whose fun-fact note is open (all closed at first)
  const factIds = useMemo(() => HISTORY_CARDS.filter((c) => c.funFacts?.length).map((c) => c.id), []);
  const [factsOpen, setFactsOpen] = useState<Set<string>>(() => new Set());
  const allFacts = factIds.every((id) => factsOpen.has(id));
  const toggleFact = (id: string) =>
    setFactsOpen((s) => {
      const next = new Set(s);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  const factSwitch = (
    <button
      type="button"
      role="switch"
      aria-checked={allFacts}
      onClick={() => setFactsOpen(allFacts ? new Set() : new Set(factIds))}
      data-history-fun-all
      className="mt-3 flex w-full items-center gap-2 rounded-lg border border-dashed border-amber-300 bg-amber-50/60 px-3 py-2 text-left text-sm text-amber-900 hover:bg-amber-50"
    >
      <span className="min-w-0 flex-1">{T("history.fun.showAll")}</span>
      <span
        aria-hidden
        className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${allFacts ? "bg-amber-500" : "bg-gray-300"}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${allFacts ? "translate-x-4" : ""}`}
        />
      </span>
    </button>
  );
  const listRef = useRef<HTMLOListElement>(null);
  const partStarts = useMemo(
    () => new Set(HISTORY_PARTS.map((part) => HISTORY_CARDS.find((c) => c.part === part)!.id)),
    [],
  );
  const layout = useInterlock(listRef, ids, partStarts);
  // The interlocked layout is measured after the first paint and may settle over a few passes (card widths change,
  // so heights and tops change): keep a deep-linked card in place while that happens, for a short while after the
  // address names it.
  const linkedAt = useRef(0);
  useEffect(() => {
    linkedAt.current = Date.now();
  }, [section]);
  useEffect(() => {
    if (layout && section && Date.now() - linkedAt.current < 2500)
      document.getElementById(elId(section))?.scrollIntoView({ block: "start", behavior: "instant" });
  }, [layout]);
  const year = (c: HistoryCard) => c.year ?? T("history.today");
  const title = (c: HistoryCard) => T(`history.${c.id}.title` as PageKey);
  const activeCard = HISTORY_CARDS.find((c) => c.id === active) ?? HISTORY_CARDS[0];
  // a deep-linked card counts as seen straight away
  const isSeen = (id: string) => seen.has(id) || id === section;

  // "how to measure a curved space from the inside": the part's one-line story, without its bold "Part I, Maths:" lead
  const tagline = (part: number) =>
    T(`history.introPart${part}` as PageKey)
      .replace(/^\*\*[^*]+\*\*\s*[:：]?\s*/, "") // the colon may sit inside the bold ("**Part I, Maths:** how…")
      .replace(/^\p{Ll}/u, (c) => c.toUpperCase()) // it now starts its own line: "How to measure…"
      .trim();
  const navList = (onPick: (id: string) => void) =>
    HISTORY_PARTS.map((part) => (
      <div key={part} className="mb-2 pt-3 first:pt-0">
        <p className={`px-3 pb-1.5 text-[15px] font-bold leading-snug ${PART_STYLE[part].text}`}>
          {T(`history.part${part}.label` as PageKey)} · {T(`history.part${part}.title` as PageKey).split(":")[0].split("：")[0]}
        </p>
        <p className="px-3 pb-2 text-xs leading-snug text-gray-500">{tagline(part)}</p>
        <ol className="space-y-0.5 border-l border-gray-200 text-sm">
          {HISTORY_CARDS.filter((c) => c.part === part).map((c) => (
            <li key={c.id}>
              <button
                onClick={() => onPick(c.id)}
                aria-current={active === c.id ? "true" : undefined}
                className={`-ml-px flex w-full items-baseline gap-2 border-l-2 px-3 py-1 text-left ${
                  active === c.id
                    ? "border-blue-600 font-medium text-blue-700"
                    : "border-transparent text-gray-600 hover:border-gray-300 hover:text-gray-900"
                }`}
              >
                <span className="w-16 shrink-0 text-[11px] tabular-nums text-gray-400">{year(c)}</span>
                <span className="min-w-0">{title(c)}</span>
              </button>
            </li>
          ))}
        </ol>
      </div>
    ));

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-4 pt-6 pb-16 sm:px-5" data-testid="history">
      <nav
        aria-label={T("history.nav")}
        className="sticky top-24 hidden max-h-[calc(100vh-12rem)] w-60 shrink-0 self-start overflow-y-auto lg:block"
      >
        <p className="mb-2 px-3 text-xs font-semibold uppercase tracking-wide text-gray-500">{T("history.nav")}</p>
        {navList(jump)}
        {factSwitch}
      </nav>

      <article className="min-w-0 flex-1">
        <header className="max-w-3xl">
          <p className="text-sm font-medium text-blue-700">{T("history.kicker")}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("history.title")}</h1>
          <p className="mt-3 text-gray-600">
            <RichText text={T("history.intro")} />
          </p>
        </header>

        {/* phones and tablets: a folding bar that stays at the top of the page */}
        <div className="sticky top-16 z-20 -mx-4 mt-5 border-b border-gray-200 bg-white/95 px-4 py-2 backdrop-blur sm:-mx-5 sm:px-5 lg:hidden">
          <button
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="flex w-full items-center gap-2 text-left text-sm"
          >
            <span className={`h-2 w-2 shrink-0 rounded-full ${PART_STYLE[activeCard.part].dot}`} />
            <span className="shrink-0 text-xs font-semibold uppercase tracking-wide text-gray-500">{T("history.nav")}</span>
            <span className="min-w-0 flex-1 truncate">
              <span className="tabular-nums text-gray-500">{year(activeCard)}</span> {title(activeCard)}
            </span>
            <span className="shrink-0 text-xs text-blue-700">
              {T("history.navOpen")} {open ? "▴" : "▾"}
            </span>
          </button>
          {open && (
            <div className="mt-2 max-h-[60vh] overflow-y-auto pb-1">
              {navList((id) => {
                setOpen(false);
                jump(id);
              })}
              {factSwitch}
            </div>
          )}
        </div>

        {/* One timeline for all three parts: the line is at the left on phones and centred on wide screens. On wide
            screens the cards alternate by global index (the alternation runs on across parts) and interlock: see
            useInterlock. Each part's badge sits on the line just above its first card. */}
        <div className="relative mt-8">
          <div
            aria-hidden
            data-history-line
            className="absolute top-0 bottom-0 left-4 w-0.5 -translate-x-1/2 bg-gray-200 lg:left-1/2"
          />
          <ol
            ref={listRef}
            className={`relative space-y-8 ${layout ? "lg:space-y-0" : ""}`}
            style={layout ? { height: layout.height } : undefined}
          >
            {HISTORY_PARTS.flatMap((part) => {
              const first = HISTORY_CARDS.findIndex((c) => c.part === part);
              const st = PART_STYLE[part];
              const label = T(`history.part${part}.label` as PageKey);
              const titleText = T(`history.part${part}.title` as PageKey);
              return [
                // phones: a full heading box in the flow
                <li key={`p${part}`} className="relative ml-10 lg:hidden">
                  <div className={`rounded-xl border bg-white px-5 py-4 ${st.border}`}>
                    <p className={`text-xs font-semibold uppercase tracking-wide ${st.text}`}>{label}</p>
                    <h2 className="mt-0.5 text-xl font-semibold">{titleText}</h2>
                    <p className="mt-1 text-sm text-gray-600">{T(`history.part${part}.sub` as PageKey)}</p>
                  </div>
                </li>,
                // wide screens: a badge on the line, in the row before the part's first card
                <li
                  key={`b${part}`}
                  aria-hidden
                  data-history-badge={part}
                  className={`z-10 w-36 -translate-x-1/2 ${layout ? "absolute left-1/2 hidden lg:block" : "hidden"}`}
                  style={layout ? { top: layout.tops[HISTORY_CARDS[first].id] - BADGE + 8 } : undefined}
                >
                  <div className={`rounded-xl border-2 bg-white px-2 py-3 text-center shadow-sm ${st.border}`}>
                    <p className={`text-sm font-semibold uppercase tracking-wide ${st.text}`}>{label}</p>
                    <p className="mt-0.5 text-xl font-bold leading-tight break-words">{titleText.split(/[:：]/)[0]}</p>
                  </div>
                </li>,
                ...HISTORY_CARDS.flatMap((c, g) =>
                  c.part === part
                    ? [
                        <TimelineCard
                          key={c.id}
                          card={c}
                          index={g}
                          partStart={g === first}
                          seen={isSeen(c.id)}
                          active={active === c.id}
                          top={layout?.tops[c.id]}
                          factOpen={factsOpen.has(c.id)}
                          onToggleFact={() => toggleFact(c.id)}
                        />,
                      ]
                    : [],
                ),
              ];
            })}
          </ol>
        </div>
      </article>
    </div>
  );
}

function TimelineCard({
  card,
  index,
  partStart,
  seen,
  active,
  top,
  factOpen,
  onToggleFact,
}: {
  card: HistoryCard;
  /** global index over all parts: even = left column, odd = right column on wide screens */
  index: number;
  partStart: boolean;
  seen: boolean;
  active: boolean;
  /** wide screens: offset from the top of the timeline (useInterlock); undefined in the single-column layout */
  top?: number;
  /** the card's "💡 Did you know?" note is expanded */
  factOpen: boolean;
  onToggleFact: () => void;
}) {
  const left = index % 2 === 0;
  const T = usePageText();
  const k = (f: string) => T(`history.${card.id}.${f}` as PageKey);
  const st = PART_STYLE[card.part];
  const year = card.year ?? T("history.today");
  // Marker geometry: the dot's centre is 35 px below the card's top, on the line (phones: 16 px from the timeline's
  // left edge, cards start at 40 px, below the year; wide screens: the middle of the 9rem centre column, 4.5rem
  // from either card, with the year above the dot).
  // The 2 px connector sits at the same y and runs from the line to the card's edge. The reveal animation moves
  // the whole item, so dot, connector and card always stay together.
  return (
    <li
      data-history-card={card.id}
      className={`pt-8 pl-10 lg:pt-0 ${top === undefined ? "relative" : "absolute lg:w-[calc(50%-4.5rem)] lg:pl-0"} ${
        seen ? "" : "motion-safe:translate-y-6 motion-safe:opacity-0"
      } motion-safe:transition motion-safe:duration-700`}
      style={top === undefined ? undefined : left ? { top, left: 0 } : { top, right: 0 }}
    >
      <span
        aria-hidden
        data-history-connector
        className={`absolute top-[66px] left-4 h-0.5 w-6 lg:top-[34px] opacity-50 ${st.dot} ${
          left ? "lg:left-auto lg:right-[-4.5rem] lg:w-[4.5rem]" : "lg:left-[-4.5rem] lg:w-[4.5rem]"
        }`}
      />
      <span
        aria-hidden
        data-history-dot
        className={`absolute top-[60px] left-4 z-10 h-3.5 lg:top-7 w-3.5 -translate-x-1/2 rounded-full ring-4 ring-gray-50 ${st.dot} ${
          left ? "lg:left-auto lg:right-[-4.5rem] lg:translate-x-1/2" : "lg:left-[-4.5rem]"
        } ${active ? "scale-125" : ""} transition-transform`}
      />
      <span
        aria-hidden
        className={`absolute top-0 left-10 whitespace-nowrap text-base leading-6 font-bold tabular-nums lg:-top-1 lg:bg-gray-50 lg:px-1 ${st.text} ${
          left ? "lg:left-auto lg:right-[-4.5rem] lg:translate-x-1/2" : "lg:left-[-4.5rem] lg:-translate-x-1/2"
        }`}
      >
        {year}
      </span>
      <article
        id={elId(card.id)}
        className={`scroll-mt-24 overflow-hidden rounded-xl border bg-white shadow-sm lg:scroll-mt-10 ${
          active ? "border-blue-300 ring-1 ring-blue-200" : "border-gray-200"
        }`}
      >
        {partStart && (
          <div className={`hidden border-b px-5 py-2.5 lg:block ${st.band}`}>
            <p className="text-[15px] font-semibold leading-snug">
              {T(`history.part${card.part}.label` as PageKey)} · {T(`history.part${card.part}.title` as PageKey)}
            </p>
            <p className="mt-0.5 text-[13px] text-gray-600">{T(`history.part${card.part}.sub` as PageKey)}</p>
          </div>
        )}
        <figure className="border-b border-gray-100 bg-gray-50">
          <div className="flex h-52 items-center justify-center p-2">
            {card.photo ? (
              <img
                src={`${import.meta.env.BASE_URL}history/${card.photo.file}`}
                alt={plainText(k("alt")).replace(/`/g, "")}
                loading="lazy"
                decoding="async"
                className="max-h-full max-w-full rounded object-contain"
              />
            ) : (
              <HistoryDrawing kind={card.drawing!} label={plainText(k("alt")).replace(/`/g, "")} />
            )}
          </div>
          <figcaption className="px-4 pb-2 text-[11px] leading-snug text-gray-500">
            <RichText text={k("alt")} />
            {card.photo ? (
              <>
                {" · "}
                {T("history.picture")}: {card.photo.author},{" "}
                {card.photo.licenceUrl ? (
                  <a href={card.photo.licenceUrl} target="_blank" rel="noreferrer" className="underline hover:text-gray-800">
                    {card.photo.licence}
                  </a>
                ) : (
                  card.photo.licence
                )}
                ,{" "}
                <a href={card.photo.source} target="_blank" rel="noreferrer" className="underline hover:text-gray-800">
                  Wikimedia Commons
                </a>
              </>
            ) : null}
          </figcaption>
        </figure>
        <div className="p-5">
          <p className="flex flex-wrap items-center gap-2">
            <span className={`rounded-full px-2 py-0.5 text-xs font-semibold tabular-nums ring-1 ${st.badge}`}>{year}</span>
            {card.funFacts?.length ? (
              <button
                type="button"
                onClick={onToggleFact}
                aria-expanded={factOpen}
                aria-controls={`history-fun-${card.id}`}
                aria-label={T("history.fun.title")}
                title={T("history.fun.title")}
                data-history-fun-badge={card.id}
                className={`rounded-full border border-dashed px-1.5 py-0.5 text-xs leading-none transition-colors ${
                  factOpen ? "border-amber-500 bg-amber-100" : "border-amber-300 bg-amber-50 hover:bg-amber-100"
                }`}
              >
                <span aria-hidden>💡</span>
              </button>
            ) : null}
          </p>
          {card.funFacts?.length && factOpen ? (
            <aside
              id={`history-fun-${card.id}`}
              data-history-fun={card.id}
              className="mt-3 rounded-lg border-2 border-dashed border-amber-300 bg-amber-50 px-4 py-3 text-sm leading-relaxed text-gray-800"
            >
              <p className="text-xs font-semibold text-amber-900">{T("history.fun.title")}</p>
              {card.funFacts.map((f) => (
                <div key={f.id} className="mt-1.5 break-words [&+&]:mt-2.5 [&+&]:border-t [&+&]:border-dashed [&+&]:border-amber-200 [&+&]:pt-2.5">
                  {f.embed && (
                    // only rendered while the note is open, so nothing loads from YouTube while it is collapsed
                    <iframe
                      src={f.embed}
                      title={T("history.fun.videoTitle")}
                      loading="lazy"
                      allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                      referrerPolicy="strict-origin-when-cross-origin"
                      className="mb-2 aspect-video w-full rounded"
                    />
                  )}
                  <p>
                    <RichText text={T(`history.fun.${f.id}` as PageKey)} />
                  </p>
                  <p className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
                    {f.sources.map((s) => (
                      <a key={s.url} href={s.url} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
                        {s.label} ↗
                      </a>
                    ))}
                  </p>
                </div>
              ))}
            </aside>
          ) : null}
          <h3 className="mt-2 text-xl font-bold leading-snug break-words">{k("title")}</h3>
          <div className="mt-3 space-y-3 text-sm leading-relaxed text-gray-700 [&_.katex]:whitespace-nowrap">
            <p>
              <RichText text={k("p1")} />
            </p>
            <p>
              <RichText text={k("p2")} />
            </p>
          </div>
          {card.id === "einsum" && (
            <div className="mt-4 flex flex-wrap gap-2">
              <a
                href="#/skills/techniques/einsum"
                className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
              >
                {T("history.finale.skills")}
              </a>
              <a
                href="#/concept/laws"
                className="rounded-md border border-blue-200 px-3 py-1.5 text-sm font-medium text-blue-800 hover:bg-blue-50"
              >
                {T("history.finale.laws")}
              </a>
            </div>
          )}
          <div className="mt-4 border-t border-gray-100 pt-3">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">{T("history.sources")}</p>
            <ul className="mt-1 space-y-0.5 text-xs">
              {card.sources.map((s) => (
                <li key={s.url} className="break-words">
                  <a href={s.url} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
                    {s.label} ↗
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </article>
    </li>
  );
}

/** Small explanatory drawings for cards without a freely licensed photo. */
function HistoryDrawing({ kind, label }: { kind: Drawing; label: string }) {
  const grid = (x: number, y: number, n: number, m: number, s: number, fill: (i: number, j: number) => string) =>
    Array.from({ length: n * m }, (_, q) => {
      const i = Math.floor(q / m);
      const j = q % m;
      return <rect key={q} x={x + j * s} y={y + i * s} width={s - 2} height={s - 2} rx={2} className={fill(i, j)} />;
    });
  const txt = "fill-gray-700 text-[13px] font-medium";
  const body: Record<Drawing, React.ReactNode> = {
    im2col: (
      <>
        {grid(10, 30, 5, 5, 22, (i, j) => (i < 3 && j < 3 ? "fill-blue-300" : "fill-gray-200"))}
        <text x={66} y={22} textAnchor="middle" className={txt}>image</text>
        <path d="M128 85h34" className="stroke-gray-400" strokeWidth={2} markerEnd="url(#hd-arrow)" />
        {grid(172, 20, 1, 9, 14, () => "fill-blue-300")}
        {grid(172, 34, 5, 9, 14, () => "fill-gray-200")}
        <text x={235} y={14} textAnchor="middle" className={txt}>patches × 9</text>
        <text x={312} y={70} className={txt}>×</text>
        {grid(330, 20, 9, 1, 14, () => "fill-amber-300")}
        <text x={337} y={160} textAnchor="middle" className={txt}>filter</text>
        <text x={355} y={95} className={txt}>=</text>
        {grid(372, 20, 6, 1, 14, (i) => (i === 0 ? "fill-blue-500" : "fill-gray-300"))}
        <text x={379} y={120} textAnchor="middle" className={txt}>out</text>
        <text x={220} y={160} textAnchor="middle" className="fill-gray-500 text-[12px]">GEMM</text>
      </>
    ),
    dataflow: (
      <>
        {[
          [40, 40, "x"],
          [40, 120, "W"],
          [160, 80, "matmul"],
          [270, 80, "+ b"],
          [370, 80, "ReLU"],
        ].map(([x, y, t]) => (
          <g key={t as string}>
            <rect x={(x as number) - 38} y={(y as number) - 16} width={76} height={32} rx={16} className="fill-white stroke-blue-400" strokeWidth={2} />
            <text x={x as number} y={(y as number) + 5} textAnchor="middle" className={txt}>{t}</text>
          </g>
        ))}
        {[
          "M78 46L122 72",
          "M78 114L122 88",
          "M198 80h34",
          "M308 80h24",
        ].map((d) => (
          <path key={d} d={d} className="stroke-amber-500" strokeWidth={3} markerEnd="url(#hd-arrow)" />
        ))}
        <text x={215} y={70} textAnchor="middle" className="fill-amber-700 text-[11px]">tensor</text>
        <text x={210} y={150} textAnchor="middle" className="fill-gray-500 text-[12px]">nodes = ops · edges = tensors</text>
      </>
    ),
    tape: (
      <>
        {["x", "W·x", "+b", "ReLU", "loss"].map((t, i) => (
          <g key={t}>
            <rect x={14 + i * 80} y={44} width={64} height={30} rx={6} className="fill-white stroke-blue-400" strokeWidth={2} />
            <text x={46 + i * 80} y={64} textAnchor="middle" className={txt}>{t}</text>
            {i < 4 && <path d={`M${80 + i * 80} 59h12`} className="stroke-blue-500" strokeWidth={2} markerEnd="url(#hd-arrow)" />}
            {i > 0 && (
              <path
                d={`M${40 + i * 80} 82 Q ${5 + i * 80} 118 ${i * 80 - 30} 82`}
                className="fill-none stroke-amber-500"
                strokeWidth={2}
                strokeDasharray="4 3"
                markerEnd="url(#hd-arrow)"
              />
            )}
          </g>
        ))}
        <text x={210} y={30} textAnchor="middle" className="fill-blue-700 text-[12px]">forward: run now, record on the tape</text>
        <text x={210} y={140} textAnchor="middle" className="fill-amber-700 text-[12px]">backward: ∂loss/∂x, ∂loss/∂W</text>
      </>
    ),
    tensorcore: (
      <>
        {grid(10, 40, 4, 4, 18, () => "fill-blue-300")}
        <text x={45} y={30} textAnchor="middle" className={txt}>A (FP16)</text>
        <text x={92} y={78} className={txt}>×</text>
        {grid(110, 40, 4, 4, 18, () => "fill-blue-300")}
        <text x={145} y={30} textAnchor="middle" className={txt}>B (FP16)</text>
        <text x={192} y={78} className={txt}>+</text>
        {grid(210, 40, 4, 4, 18, () => "fill-gray-300")}
        <text x={245} y={30} textAnchor="middle" className={txt}>C</text>
        <text x={292} y={78} className={txt}>=</text>
        {grid(310, 40, 4, 4, 18, () => "fill-emerald-400")}
        <text x={345} y={30} textAnchor="middle" className={txt}>D (FP32)</text>
        <text x={200} y={140} textAnchor="middle" className="fill-gray-500 text-[12px]">one 4 × 4 × 4 multiply–accumulate per clock</text>
      </>
    ),
    einsum: (
      <>
        <text x={210} y={30} textAnchor="middle" className="fill-gray-800 font-mono text-[15px]">einsum("ij,jk-&gt;ik", A, B)</text>
        {grid(30, 50, 3, 4, 16, (_, j) => (j === 1 ? "fill-amber-300" : "fill-blue-200"))}
        <text x={62} y={115} textAnchor="middle" className={txt}>A_ij</text>
        <text x={110} y={80} className={txt}>·</text>
        {grid(130, 42, 4, 3, 16, (i) => (i === 1 ? "fill-amber-300" : "fill-blue-200"))}
        <text x={154} y={120} textAnchor="middle" className={txt}>B_jk</text>
        <path d="M196 74h34" className="stroke-gray-400" strokeWidth={2} markerEnd="url(#hd-arrow)" />
        {grid(244, 50, 3, 3, 16, () => "fill-emerald-300")}
        <text x={268} y={115} textAnchor="middle" className={txt}>C_ik</text>
        <text x={350} y={70} textAnchor="middle" className="fill-amber-700 text-[12px]">j appears twice</text>
        <text x={350} y={88} textAnchor="middle" className="fill-amber-700 text-[12px]">→ summed (Einstein)</text>
        <text x={210} y={150} textAnchor="middle" className="fill-gray-500 text-[12px]">1916 notation · 2010s hardware</text>
      </>
    ),
  };
  return (
    <svg viewBox="0 0 420 170" role="img" aria-label={label} className="h-full max-h-48 w-full max-w-md">
      <defs>
        <marker id="hd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" className="fill-gray-500" />
        </marker>
      </defs>
      {body[kind]}
    </svg>
  );
}
