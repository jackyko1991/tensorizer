import { Fragment } from "react";
import { useLang } from "../i18n";
import type { TranslationRow } from "../types";
import Tex, { RichText } from "./Tex";

/**
 * Symbol → code → shape, with each row's meaning on its own full-width line underneath,
 * so the explanation stays readable however narrow the column is.
 */
export default function TranslationTable({ rows }: { rows: TranslationRow[] }) {
  const { t } = useLang();
  return (
    <div className="relative overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left text-xs text-gray-500">
            <th className="py-1.5 pr-3 font-medium">{t("col.symbol")}</th>
            <th className="py-1.5 pr-3 font-medium">{t("col.code")}</th>
            <th className="py-1.5 font-medium">{t("col.shape")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <Fragment key={row.symbol + row.code}>
              <tr className="align-top">
                <td className="pt-2.5 pr-3 whitespace-nowrap">
                  <Tex tex={row.symbol} />
                </td>
                <td className="pt-2.5 pr-3">
                  <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-xs break-words text-gray-800">
                    {row.code}
                  </code>
                </td>
                <td className="pt-2.5 font-mono text-xs whitespace-nowrap text-gray-600">{row.shape}</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td colSpan={3} className="pt-1 pb-2.5 text-[13px] leading-relaxed text-gray-600">
                  <span className="sr-only">{t("col.meaning")}: </span>
                  <RichText text={row.meaning} />
                </td>
              </tr>
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
