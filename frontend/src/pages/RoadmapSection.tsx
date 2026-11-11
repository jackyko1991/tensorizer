import { useLang, type Key } from "../i18n";
import { useState } from "react";
import { hub } from "../account";
import SignInDialog from "../components/SignInDialog";
import WhereChip from "../components/WhereChip";
import { formatPageRoute } from "../pageRoute";

/** level, what you get, needs, available now (✓ in the table) */
const FREE: [Key, Key, Key, boolean][] = [
  ["rm.f1", "rm.f1What", "rm.f1Needs", true],
  ["rm.f2", "rm.f2What", "rm.f2Needs", true],
  ["rm.f3", "rm.f3What", "rm.f3Needs", true],
  ["rm.f4", "rm.f4What", "rm.f4Needs", true],
];

/** AI agent ▸ Product roadmap (#/agents/roadmap): what is free now (community), and the planned Pro and Campus tiers.
 *  Formalizer's AI agent page has the matching section. */
export default function RoadmapSection() {
  const { t } = useLang();
  const [joining, setJoining] = useState(false);
  return (
    <div className="space-y-6" data-testid="roadmap">
      <div>
        <h2 className="text-xl font-semibold">{t("rm.tab")}</h2>
        <p className="mt-2 text-sm text-gray-600">{t("rm.intro")}</p>
      </div>
      <div className="rounded-xl border border-amber-200 bg-amber-50 px-5 py-3 text-sm text-amber-900">
        <p className="font-semibold">{t("rm.pending")}</p>
        <p className="mt-1">{t("rm.pendingBody")}</p>
      </div>

      <section className="rounded-xl border border-gray-200 bg-white p-6">
        <h3 className="font-semibold text-gray-900">{t("rm.freeTitle")}</h3>
        <p className="mt-2 text-sm text-gray-700">{t("rm.freeBody")}</p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-xs uppercase tracking-wide text-gray-500">
                <th className="py-2 pr-4 font-semibold">{t("rm.hLevel")}</th>
                <th className="py-2 pr-4 font-semibold">{t("rm.hWhat")}</th>
                <th className="py-2 pr-4 font-semibold">{t("rm.hNeeds")}</th>
                <th className="py-2 text-center font-semibold">{t("rm.hNow")}</th>
              </tr>
            </thead>
            <tbody>
              {FREE.map(([level, what, needs, now]) => (
                <tr key={level} className="border-b border-gray-100 align-top last:border-0">
                  <td className="py-2.5 pr-4 font-medium whitespace-nowrap text-gray-900">{t(level)}</td>
                  <td className="py-2.5 pr-4 text-gray-700">{t(what)}</td>
                  <td className="py-2.5 pr-4 text-gray-600">{t(needs)}</td>
                  <td className="py-2.5 text-center">
                    {now ? (
                      <span className="font-semibold text-emerald-600" aria-label={t("rm.hNow")}>
                        ✓
                      </span>
                    ) : (
                      <span className="text-gray-300">–</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {(
          [
            ["rm.proTitle", "rm.proBody"],
            ["rm.researchTitle", "rm.researchBody"],
            ["rm.campusTitle", "rm.campusBody"],
          ] as [Key, Key][]
        ).map(([title, body]) => (
          <section key={title} className="rounded-xl border border-dashed border-gray-300 bg-white p-6">
            <h3 className="font-semibold text-gray-900">{t(title)}</h3>
            <p className="mt-2 text-sm leading-relaxed text-gray-700">{t(body)}</p>
          </section>
        ))}
      </div>

      <ul className="space-y-2 text-sm text-gray-700">
        {(
          [
            ["rm.p1Title", "rm.p1"],
            ["rm.p2Title", "rm.p2"],
            ["rm.p3Title", "rm.p3"],
          ] as [Key, Key][]
        ).map(([title, body]) => (
          <li key={title} className="flex gap-2">
            <span className="text-blue-600">✓</span>
            <span>
              <b className="text-gray-900">{t(title)}</b> {t(body)}
              {title === "rm.p2Title" && (
                <span className="mt-1.5 flex flex-wrap gap-1.5">
                  <WhereChip where="local" />
                  <WhereChip where="server" />
                  <WhereChip where="cloud" />
                </span>
              )}
            </span>
          </li>
        ))}
      </ul>

      {hub && (
        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-violet-200 bg-violet-50 px-5 py-4">
          <p className="min-w-0 flex-1 text-sm text-violet-900">{t("acct.waitlistIntro")}</p>
          <button
            onClick={() => setJoining(true)}
            data-testid="roadmap-waitlist"
            className="rounded-md bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-700"
          >
            {t("acct.waitlistTitle")}
          </button>
          {joining && <SignInDialog waitlist onClose={() => setJoining(false)} />}
        </div>
      )}

      <p className="flex flex-wrap gap-4 text-sm">
        <a href={`#${formatPageRoute("agents", "skills")}`} className="font-medium text-blue-700 hover:underline">
          {t("ya.skills")} →
        </a>
        <a href={`#${formatPageRoute("agents", "mcp")}`} className="font-medium text-blue-700 hover:underline">
          MCP →
        </a>
      </p>
    </div>
  );
}
