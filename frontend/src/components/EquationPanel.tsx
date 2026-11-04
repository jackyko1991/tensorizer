import type { Problem } from "../types";
import { useLang } from "../i18n";
import Tex, { RichText } from "./Tex";
import TranslationTable from "./TranslationTable";

export default function EquationPanel({ problem }: { problem: Problem }) {
  const { t } = useLang();
  return (
    <section data-ai-region="equations" className="rounded-xl border border-gray-200 bg-white p-5">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">{t("eq.vector")}</h3>
      <div className="mt-1 text-lg">
        <Tex tex={problem.vector_form} display />
      </div>

      <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("eq.index")}</h3>
      <div className="mt-1 text-lg">
        <Tex tex={problem.index_form} display />
      </div>

      <h3 className="mt-5 text-xs font-semibold uppercase tracking-wide text-gray-500">{t("eq.translation")}</h3>
      <div className="mt-2">
        <TranslationTable rows={problem.translation} />
      </div>

      {problem.notes && (
        <p data-ai-region="notes" className="mt-4 text-sm leading-relaxed text-gray-600">
          <RichText text={problem.notes} people />
        </p>
      )}
    </section>
  );
}
