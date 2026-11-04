import { useState } from "react";
import type { Problem } from "../types";
import { useLang } from "../i18n";
import { pickMd, toLatex, toMarkdown } from "../metaDoc";
import { saveText } from "../fileIO";
import { paragraphs } from "../paragraphs";
import DownloadIcon from "./DownloadIcon";
import Tex, { RichText } from "./Tex";
import TranslationTable from "./TranslationTable";

/** ⧉ Copy (Markdown to the clipboard) / ⬇ Markdown / ⬇ LaTeX: the formulation (title, summary, equations, translation, notes) in the shown language. */
export function FormulationExport({ problem, className = "" }: { problem: Problem; className?: string }) {
  const fields = () => pickMd({ ...problem, notes: problem.notes ?? "" });
  const name = problem.slug || "problem";
  const { t } = useLang();
  const [copied, setCopied] = useState<"" | "ok" | "fail">("");
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(toMarkdown(fields()));
      setCopied("ok");
    } catch {
      setCopied("fail");
    }
    window.setTimeout(() => setCopied(""), 1500);
  };
  const button =
    "flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-gray-500 hover:bg-gray-100 hover:text-blue-700";
  return (
    <span className={`flex items-center gap-1 ${className}`}>
      <button onClick={copy} className={button} title={t("formula.copyHint")} data-testid="copy-md">
        <span aria-hidden className="text-xs leading-none">
          {copied === "ok" ? "✓" : "⧉"}
        </span>
        {copied === "ok" ? t("formula.copied") : copied === "fail" ? t("formula.copyFail") : t("formula.copy")}
      </button>
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

export default function EquationPanel({ problem, exportable = true }: { problem: Problem; exportable?: boolean }) {
  const { t } = useLang();
  return (
    <section data-ai-region="equations" className="rounded-xl border border-gray-200 bg-white p-5">
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">{t("eq.vector")}</h3>
        {exportable && <FormulationExport problem={problem} className="-mt-1 -mr-2" />}
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
        <div data-ai-region="notes" className="mt-4 space-y-2.5 text-sm leading-relaxed text-gray-600">
          {paragraphs(problem.notes).map((para, i) => (
            <p key={i}>
              <RichText text={para} people />
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
