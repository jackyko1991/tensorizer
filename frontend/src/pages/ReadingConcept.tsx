import PageToc from "../components/PageToc";
import { RichText } from "../components/Tex";
import { usePageText, type PageKey } from "../content/pages";
import { AFFILIATE_REL, BOOK_GROUPS, BOOKS, amazonUrl, type Book } from "../books";

/**
 * Concept tab "Further reading": books in four groups, top picks (★) first. Each card has the title, authors,
 * publisher and year, a one-line reason, an affiliate link to Amazon UK, a link to a free official online edition
 * when one exists, and a link to the matching tab of this site. Affiliate disclosure at the bottom (Amazon Associates
 * Operating Agreement; ASA/CMA). No prices, ratings, covers or Amazon scripts. Book data: books.ts.
 * Deep links: #/concept/reading/<group> (the scroll is done by ConceptPage, element id reading-<group>).
 */
export default function ReadingConcept() {
  const T = usePageText();

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pb-10">
      <PageToc
        scrollRoot="concept-scroll"
        items={BOOK_GROUPS.map((g, i) => ({ id: `reading-${g}`, mark: i + 1, label: T(`reading.group.${g}` as PageKey) }))}
      />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10" data-testid="reading-concept">
        <header>
          <p className="text-sm font-medium text-blue-700">{T("reading.kicker")}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("reading.title")}</h1>
          <p className="mt-3 text-gray-600">
            <RichText text={T("reading.intro")} />
          </p>
        </header>

        {BOOK_GROUPS.map((g, i) => (
          <section key={g} id={`reading-${g}`} className="scroll-mt-6">
            <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
                {i + 1}
              </span>
              {T(`reading.group.${g}` as PageKey)}
            </h2>
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {BOOKS.filter((b) => b.group === g).map((b) => (
                <BookCard key={b.id} book={b} />
              ))}
            </div>
          </section>
        ))}

        {/* affiliate disclosure: one quiet grey line under the books */}
        <p className="border-t border-gray-200 pt-3 text-xs text-gray-400" data-testid="reading-disclosure">
          {T("reading.disclosure")} {T("reading.note")}
        </p>
      </article>
    </div>
  );
}

function BookCard({ book: b }: { book: Book }) {
  const T = usePageText();
  return (
    <div
      className={`flex flex-col rounded-xl border bg-white p-4 sm:p-5 ${b.top ? "border-amber-300" : "border-gray-200"}`}
      data-testid={`book-${b.id}`}
    >
      {b.top && (
        <span className="mb-2 self-start rounded-full bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-800 ring-1 ring-amber-200">
          ★ {T("reading.top")}
        </span>
      )}
      <h3 className="font-semibold leading-snug text-gray-900" lang="en">
        {b.title}
        {b.edition && <span className="font-normal text-gray-600"> ({b.edition})</span>}
      </h3>
      <p className="mt-1 text-sm text-gray-600" lang="en">
        {b.authors}
      </p>
      <p className="text-xs text-gray-500" lang="en">
        {b.publisher}, {b.year}
      </p>
      <p className="mt-3 flex-1 text-sm leading-relaxed text-gray-700">{T(`reading.book.${b.id}.why` as PageKey)}</p>
      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <a
          href={amazonUrl(b.isbn10)}
          target="_blank"
          rel={AFFILIATE_REL}
          className="rounded-md bg-blue-600 px-3 py-1.5 font-medium text-white hover:bg-blue-700"
        >
          {T("reading.amazon")}
        </a>
        {b.free && (
          <a href={b.free} target="_blank" rel="noopener noreferrer" className="font-medium text-blue-700 hover:underline">
            {T("reading.free")} ↗
          </a>
        )}
      </div>
      <p className="mt-3 border-t border-gray-100 pt-2 text-xs text-gray-500">
        {T("reading.site")}{" "}
        <a href={b.site.href} className="font-medium text-blue-700 hover:underline">
          {T(b.site.label)}
        </a>
      </p>
    </div>
  );
}
