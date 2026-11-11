/**
 * Further reading ▸ Code labs / Explorables: the sister sites of the learnkit series of one kind, as cards (logo,
 * name, what it is about, a one-line pitch), the current site marked "You are here", and a short line linking to the
 * other kind. Language comes in as a plain code so sites with their own i18n can use this file as it is.
 */
import { SERIES_GROUPS, SERIES_INTRO, SERIES_SITES, SERIES_TITLE, seriesText, seriesUrl, VISIT, YOU_ARE_HERE, type SeriesKind } from "../series";

export default function SeriesPage({
  kind,
  current,
  lang,
  onOther,
}: {
  kind: SeriesKind;
  /** this site's id in the series ("formalizer"), marked "You are here" */
  current: string;
  lang: string;
  /** open the other group's tab */
  onOther?: () => void;
}) {
  const group = SERIES_GROUPS.find((g) => g.kind === kind)!;
  const other = SERIES_GROUPS.find((g) => g.kind !== kind)!;
  const sites = SERIES_SITES.filter((s) => s.kind === kind);
  const T = (texts: Parameters<typeof seriesText>[0]) => seriesText(texts, lang);
  return (
    <div className="mx-auto max-w-4xl px-5 pt-8 pb-16" data-testid={`series-${kind}`}>
      <p className="text-xs font-semibold tracking-wide text-gray-500 uppercase">{T(SERIES_TITLE)}</p>
      <h1 className="mt-1 text-3xl font-semibold tracking-tight text-gray-900">{T(group.title)}</h1>
      <p className="mt-3 text-gray-600">{T(group.intro)}</p>
      <p className="mt-1 text-sm text-gray-500">{T(SERIES_INTRO)}</p>
      <ul className="mt-6 grid gap-4 sm:grid-cols-2">
        {sites.map((s) => {
          const here = s.id === current;
          return (
            <li key={s.id}>
              <a
                href={here ? undefined : seriesUrl(s, lang)}
                target={here ? undefined : "_blank"}
                rel="noreferrer"
                aria-current={here ? "page" : undefined}
                data-testid="series-site"
                className={`flex h-full flex-col rounded-xl border bg-white p-5 shadow-sm transition ${s.border} ${here ? "cursor-default" : ""}`}
              >
                <span className="flex items-center gap-3">
                  <img src={s.logo} alt="" className="h-9 w-9 rounded-md" loading="lazy" />
                  <span className="min-w-0">
                    <span className="block text-lg font-semibold text-gray-900">{s.name}</span>
                    <span className={`mt-0.5 inline-block rounded-full px-2 py-0.5 text-xs font-medium ${s.badge}`}>{T(s.verb)}</span>
                  </span>
                </span>
                <span className="mt-3 flex-1 text-sm leading-relaxed text-gray-700">{T(s.pitch)}</span>
                <span className="mt-4 text-sm font-medium text-gray-500">
                  {here ? <span className="rounded-full bg-gray-100 px-2.5 py-1 text-xs">{T(YOU_ARE_HERE)}</span> : `${T(VISIT)} ↗`}
                </span>
              </a>
            </li>
          );
        })}
      </ul>
      {onOther && (
        <p className="mt-8 text-sm text-gray-600">
          {T(other.intro)}{" "}
          <button onClick={onOther} className="font-medium text-blue-700 hover:underline">
            {T(other.title)} →
          </button>
        </p>
      )}
    </div>
  );
}
