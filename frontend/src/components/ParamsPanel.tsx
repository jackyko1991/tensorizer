import type { Params, ParamSpec } from "../types";
import { useLang } from "../i18n";
import ParamControls from "./ParamControls";

interface Props {
  specs: Record<string, ParamSpec>;
  values: Params;
  onChange: (values: Params) => void;
  onDefaults: () => void;
  onRun: () => void;
  canRun: boolean;
  running: boolean;
  hasEdits: boolean;
  verbose: boolean;
  onVerbose: (on: boolean) => void;
  open: boolean;
  onToggle: () => void;
}

/** Foldable side panel next to the code editor: parameter sliders and the single Run button. */
export default function ParamsPanel(props: Props) {
  const { specs, values, onChange, onDefaults, onRun, canRun, running, hasEdits, verbose, onVerbose, open, onToggle } = props;
  const { t } = useLang();
  const runLabel = t(running ? "run.running" : hasEdits ? "run.mine" : "run");
  const runTitle = canRun ? "" : t("run.needBackend");

  if (!open) {
    return (
      <aside className="flex w-10 shrink-0 flex-col items-center gap-2 border-l border-gray-200 bg-gray-50 py-2">
        <button
          onClick={onRun}
          disabled={!canRun || running}
          title={runTitle || runLabel}
          className="h-7 w-7 rounded-md bg-blue-600 text-xs text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          ▶
        </button>
        <button
          onClick={onToggle}
          className="flex-1 text-xs font-medium tracking-wide text-gray-500 [writing-mode:vertical-rl] hover:text-gray-800"
          title={t("params.open")}
        >
          {t("params.title")}
        </button>
      </aside>
    );
  }

  return (
    <aside className="flex w-full shrink-0 flex-col border-t border-gray-200 bg-white xl:w-60 xl:border-t-0 xl:border-l">
      <div className="flex h-8 shrink-0 items-center justify-between border-b border-gray-200 bg-gray-50 px-3 text-xs">
        <span className="font-semibold uppercase tracking-wide text-gray-500">{t("params.title")}</span>
        <button onClick={onToggle} className="hidden text-gray-500 hover:text-gray-800 xl:inline" title={t("params.fold")}>
          »
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3 [&>div]:!grid-cols-1">
        <ParamControls specs={specs} values={values} onChange={onChange} disabled={running} />
      </div>
      <div className="shrink-0 space-y-2 border-t border-gray-200 p-3">
        <label className="flex cursor-pointer items-center gap-2 text-sm text-gray-700" title={t("params.verboseHint")}>
          <input
            type="checkbox"
            checked={verbose}
            onChange={(e) => onVerbose(e.target.checked)}
            className="h-4 w-4 accent-blue-600"
          />
          {t("params.verbose")}
        </label>
        <button
          onClick={onDefaults}
          disabled={running}
          className="w-full rounded-md border border-gray-300 bg-white py-1.5 text-sm text-gray-700 hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-400"
        >
          {t("params.defaults")}
        </button>
        <button
          onClick={onRun}
          disabled={!canRun || running}
          title={runTitle}
          className="w-full rounded-md bg-blue-600 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          {runLabel}
        </button>
      </div>
    </aside>
  );
}
