import { useLang, type Key } from "../i18n";

type T = (key: Key) => string;

// famous problem lists and years (not prizes): shown as violet badges
const LANDMARKS: Record<string, { label: Key; hint: Key }> = {
  "annus-mirabilis": { label: "tag.annus", hint: "tag.annusHint" },
  hilbert: { label: "tag.hilbert", hint: "tag.hilbertHint" },
  smale: { label: "tag.smale", hint: "tag.smaleHint" },
  landau: { label: "tag.landau", hint: "tag.landauHint" },
};

/** A library tag as text: "★ Millennium Prize problem", "🏅 Nobel Prize in Physics 2016", or null for unknown tags.
 *  Tags are prize:field:year (nobel, wolf, breakthrough) or prize:year (fields, abel, turing, boltzmann, ...). */
export function prizeLabel(tag: string, t: T): string | null {
  const [prize, ...rest] = tag.split(":");
  if (prize === "millennium") return `★ ${t("tag.millennium")}`;
  const landmark = LANDMARKS[prize];
  if (landmark) return `✦ ${t(landmark.label).replace("{n}", rest[0] ?? "")}`.trim();
  const field = rest.length > 1 ? rest[0] : null;
  const year = rest.at(-1) ?? "";
  const key = (field ? `prize.${prize}.${field}` : `prize.${prize}`) as Key;
  const name = t(key);
  return name === key ? null : `🏅 ${name} ${year}`.trim();
}

const kind = (tag: string) => tag.split(":")[0];

/** Library tags as badges (gold: prizes; violet: famous problem lists and years): Millennium Prize problems and prizes won by the work behind a problem. */
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
              title={t(LANDMARKS[kind(tag)]?.hint ?? (tag === "millennium" ? "tag.millenniumHint" : "prize.hint"))}
              data-testid={`tag-${kind(tag)}`}
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                LANDMARKS[kind(tag)] ? "bg-violet-100 text-violet-800" : "bg-amber-100 text-amber-800"
              }`}
            >
              {text}
            </span>
          )
        );
      })}
    </>
  );
}
