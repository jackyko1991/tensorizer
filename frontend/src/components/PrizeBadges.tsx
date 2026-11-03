import { useLang, type Key } from "../i18n";

type T = (key: Key) => string;

/** A library tag as text: "★ Millennium Prize problem", "🏅 Nobel Prize in Physics 2016", or null for unknown tags.
 *  Tags are prize:field:year (nobel, wolf, breakthrough) or prize:year (fields, abel, turing, boltzmann, ...). */
export function prizeLabel(tag: string, t: T): string | null {
  const [prize, ...rest] = tag.split(":");
  if (prize === "millennium") return `★ ${t("tag.millennium")}`;
  const field = rest.length > 1 ? rest[0] : null;
  const year = rest.at(-1) ?? "";
  const key = (field ? `prize.${prize}.${field}` : `prize.${prize}`) as Key;
  const name = t(key);
  return name === key ? null : `🏅 ${name} ${year}`.trim();
}

/** Library tags as gold badges: Millennium Prize problems and prizes won by the work behind a problem. */
export default function PrizeBadges({ tags }: { tags: string[] }) {
  const { t } = useLang();
  return (
    <>
      {tags.map((tag) => {
        const text = prizeLabel(tag, t);
        return (
          text && (
            <span
              key={tag}
              title={t(tag === "millennium" ? "tag.millenniumHint" : "prize.hint")}
              data-testid={`tag-${tag.split(":")[0]}`}
              className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800"
            >
              {text}
            </span>
          )
        );
      })}
    </>
  );
}
