import { python } from "@codemirror/lang-python";
import CodeMirror from "@uiw/react-codemirror";
import { useState, type ReactNode } from "react";
import { useLang, type Key } from "../i18n";
import type { CodeFile } from "../types";

const ROLE: Record<string, Key> = {
  "main.py": "role.main",
  "equation.py": "role.equation",
  "tensors.py": "role.tensors",
  "solve.py": "role.solve",
  "plot.py": "role.plot",
};

interface Props {
  files: CodeFile[];
  edits: Record<string, string>;
  editable: boolean;
  onEdit: (name: string, code: string) => void;
  onReset: () => void;
  /** Rendered to the right of the editor (the parameters panel). */
  side?: ReactNode;
  /** Rendered at the right of the info row (notebook export). */
  toolbar?: ReactNode;
  className?: string;
}

export default function CodePanel({ files, edits, editable, onEdit, onReset, side, toolbar, className = "" }: Props) {
  const { t } = useLang();
  const [active, setActive] = useState(files[0]?.name);
  const file = files.find((f) => f.name === active) ?? files[0];
  const edited = Object.keys(edits).length > 0;

  return (
    <section className={`flex flex-col overflow-hidden bg-white lg:flex-row ${className}`}>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex shrink-0 items-end gap-1 overflow-x-auto border-b border-gray-200 bg-gray-50 px-2 pt-2">
          {files.map((f, i) => (
            <button
              key={f.name}
              onClick={() => setActive(f.name)}
              title={t(ROLE[f.name])}
              className={`shrink-0 rounded-t-md border border-b-0 px-3 py-1.5 font-mono text-xs ${
                f.name === file.name
                  ? "border-gray-200 bg-white text-gray-900"
                  : "border-transparent text-gray-500 hover:text-gray-800"
              }`}
            >
              <span className="mr-1 text-gray-400">{i + 1}.</span>
              {f.name}
              {edits[f.name] !== undefined && <span className="ml-1 text-amber-600">●</span>}
            </button>
          ))}
        </div>

        <div className="flex h-7 shrink-0 items-center justify-between gap-3 border-b border-gray-100 px-3 text-xs text-gray-500">
          <span className="truncate">
            {t(ROLE[file.name])} · {t(editable ? "code.editable" : "code.readonly")}
            {edited && (
              <button onClick={onReset} className="ml-3 text-blue-600 hover:underline">
                {t("code.reset")}
              </button>
            )}
          </span>
          {toolbar && <span className="flex shrink-0 items-center gap-3">{toolbar}</span>}
        </div>

        <div className="max-h-[60vh] min-h-0 flex-1 overflow-auto lg:max-h-none">
          <CodeMirror
            value={edits[file.name] ?? file.code}
            extensions={[python()]}
            editable={editable}
            readOnly={!editable}
            theme="light"
            basicSetup={{ foldGutter: false, highlightActiveLine: editable }}
            onChange={(code) => onEdit(file.name, code)}
          />
        </div>
      </div>
      {side}
    </section>
  );
}
