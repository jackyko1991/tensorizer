import { useLang, type Key } from "../i18n";

type T = (key: Key) => string;

// the status of a statement (mathematics, or a physics claim): open, proved, disproved, …
export const STATUS: Record<string, { icon: string; label: Key; hint: Key; style: string }> = {
  open: {
    icon: "❓",
    label: "status.open",
    hint: "status.openHint",
    style: "bg-sky-50 text-sky-800 ring-1 ring-sky-200",
  },
  proved: {
    icon: "✅",
    label: "status.proved",
    hint: "status.provedHint",
    style: "bg-emerald-50 text-emerald-800 ring-1 ring-emerald-200",
  },
  disproved: {
    icon: "❌",
    label: "status.disproved",
    hint: "status.disprovedHint",
    style: "bg-rose-50 text-rose-800 ring-1 ring-rose-200",
  },
  partial: {
    icon: "◐",
    label: "status.partial",
    hint: "status.partialHint",
    style: "bg-teal-50 text-teal-800 ring-1 ring-teal-200",
  },
  claimed: {
    icon: "⏳",
    label: "status.claimed",
    hint: "status.claimedHint",
    style: "bg-yellow-50 text-yellow-800 ring-1 ring-yellow-200",
  },
  undecidable: {
    icon: "∞",
    label: "status.undecidable",
    hint: "status.undecidableHint",
    style: "bg-slate-100 text-slate-700 ring-1 ring-slate-300",
  },
};

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
  if (prize === "speculative") return `🧪 ${t("tag.speculative")}`;
  if (prize === "status" && STATUS[rest[0]]) return `${STATUS[rest[0]].icon} ${t(STATUS[rest[0]].label)}`;
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
              title={t(
                LANDMARKS[kind(tag)]?.hint ??
                  (kind(tag) === "status" ? STATUS[tag.split(":")[1]]?.hint : undefined) ??
                  (tag === "millennium"
                    ? "tag.millenniumHint"
                    : tag === "speculative"
                      ? "tag.speculativeHint"
                      : "prize.hint"),
              )}
              data-testid={`tag-${kind(tag)}`}
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                kind(tag) === "status"
                  ? (STATUS[tag.split(":")[1]]?.style ?? "")
                  : tag === "speculative"
                    ? "border border-dashed border-orange-300 bg-orange-50 text-orange-800"
                    : LANDMARKS[kind(tag)]
                      ? "bg-violet-100 text-violet-800"
                      : "bg-amber-100 text-amber-800"
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
