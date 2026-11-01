import type { Problem } from "../types";
import { useLang } from "../i18n";
import Tex, { RichText } from "./Tex";

export default function EquationPanel({ problem }: { problem: Problem }) {
  const { t } = useLang();
  return (
    <section className="rounded-xl border border-gray-200 bg-white p-5">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">{t("eq.vector")}</h3>
      <div className="mt-1 text-lg">
        <Tex tex={problem.vector_form} display />
      </div>

      <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("eq.index")}</h3>
      <div className="mt-1 text-lg">
        <Tex tex={problem.index_form} display />
      </div>

      <h3 className="mt-5 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("eq.translation")}</h3>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 text-left text-xs text-gray-500">
              <th className="py-1.5 pr-3 font-medium">{t("col.symbol")}</th>
              <th className="py-1.5 pr-3 font-medium">{t("col.code")}</th>
              <th className="py-1.5 pr-3 font-medium">{t("col.shape")}</th>
              <th className="py-1.5 font-medium">{t("col.meaning")}</th>
            </tr>
          </thead>
          <tbody>
            {problem.translation.map((row) => (
              <tr key={row.symbol + row.code} className="border-b border-gray-100 align-top">
                <td className="py-1.5 pr-3 whitespace-nowrap">
                  <Tex tex={row.symbol} />
                </td>
                <td className="py-1.5 pr-3">
                  <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-xs text-gray-800">{row.code}</code>
                </td>
                <td className="py-1.5 pr-3 whitespace-nowrap font-mono text-xs text-gray-600">{row.shape}</td>
                <td className="py-1.5 text-gray-700">
                  <RichText text={row.meaning} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {problem.notes && (
        <p className="mt-4 text-sm leading-relaxed text-gray-600">
          <RichText text={problem.notes} />
        </p>
      )}
    </section>
  );
}
