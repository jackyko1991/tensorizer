import type { Problem } from "../types";
import { useLang } from "../i18n";
import { pickMd, toLatex, toMarkdown } from "../metaDoc";
import { saveText } from "../fileIO";
import DownloadIcon from "./DownloadIcon";
import Tex, { RichText } from "./Tex";
import TranslationTable from "./TranslationTable";

/** ⬇ Markdown / ⬇ LaTeX: the formulation (title, summary, equations, translation, notes) in the shown language. */
export function FormulationExport({ problem, className = "" }: { problem: Problem; className?: string }) {
  const fields = () => pickMd({ ...problem, notes: problem.notes ?? "" });
  const name = problem.slug || "problem";
  const button =
    "flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-gray-500 hover:bg-gray-100 hover:text-blue-700";
  return (
    <span className={`flex items-center gap-1 ${className}`}>
      <button
        onClick={() => saveText(`${name}.md`, toMarkdown(fields()), "text/markdown")}
        className={button}
        data-testid="export-md"
      >
        <DownloadIcon className="h-3 w-3" /> Markdown
      </button>
      <button
        onClick={() => saveText(`${name}.tex`, toLatex(fields()), "application/x-tex")}
        className={button}
        data-testid="export-tex"
      >
        <DownloadIcon className="h-3 w-3" /> LaTeX
      </button>
    </span>
  );
}

export default function EquationPanel({ problem }: { problem: Problem }) {
  const { t } = useLang();
  return (
    <section data-ai-region="equations" className="rounded-xl border border-gray-200 bg-white p-5">
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">{t("eq.vector")}</h3>
        <FormulationExport problem={problem} className="-mt-1 -mr-2" />
      </div>
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
