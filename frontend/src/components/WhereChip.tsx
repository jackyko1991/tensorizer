import { backendChoice } from "../api";
import { useLang } from "../i18n";

/** Where something runs, so nobody has to guess whether their code leaves their computer:
 *  ⌂ Local (green): this computer · ↗ Your server (blue): a server you run · ☁ Pro Cloud (purple): hosted.
 *  The same three chips (and symbols) are used by Formalizer. */
export type Where = "local" | "server" | "cloud";

const STYLE: Record<Where, string> = {
  local: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  server: "bg-blue-50 text-blue-700 ring-blue-200",
  cloud: "bg-violet-50 text-violet-700 ring-violet-200",
};
const ICON: Record<Where, string> = { local: "⌂", server: "↗", cloud: "☁" };

/** Where runs and Ask AI's local agents go: the chosen backend. */
export const backendWhere = (): Where => (backendChoice() === "remote" ? "server" : "local");

export default function WhereChip({ where, className = "" }: { where: Where; className?: string }) {
  const { t } = useLang();
  return (
    <span
      title={t(`where.${where}Hint`)}
      data-testid={`where-${where}`}
      className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-px text-[11px] font-medium whitespace-nowrap ring-1 ${STYLE[where]} ${className}`}
    >
      <span aria-hidden>{ICON[where]}</span>
      {t(`where.${where}`)}
    </span>
  );
}
