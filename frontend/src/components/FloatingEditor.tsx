import { yaml } from "@codemirror/lang-yaml";
import CodeMirror, { type ViewUpdate } from "@uiw/react-codemirror";
import { useEffect, useRef, useState } from "react";
import { useLang } from "../i18n";
import type { MetaFields } from "../metaDoc";
import type { Problem } from "../types";
import { MarkdownEditor } from "./WsEditor";

export type FloatMode = "markdown" | "yaml";

/** The split button in a Workspace problem's header: opens the floating Markdown or meta.yaml editor. */
export function EditorButtons({ open, onOpen }: { open: FloatMode | null; onOpen: (m: FloatMode | null) => void }) {
  const { t } = useLang();
  return (
    <div className="inline-flex overflow-hidden rounded-md border border-gray-200 text-xs" title={t("edit.floatHint")}>
      {(["markdown", "yaml"] as const).map((m, i) => (
        <button
          key={m}
          onClick={() => onOpen(open === m ? null : m)}
          aria-pressed={open === m}
          className={`px-2.5 py-1 font-medium ${i ? "border-l border-gray-200" : ""} ${
            open === m ? "bg-blue-600 text-white" : "bg-white text-gray-600 hover:bg-gray-50 hover:text-gray-900"
          }`}
        >
          {m === "yaml" ? "meta.yaml" : "Markdown"}
        </button>
      ))}
    </div>
  );
}

interface Props {
  mode: FloatMode;
  onMode: (m: FloatMode) => void;
  onClose: () => void;
  problem: Problem;
  onMeta: (f: Partial<MetaFields>) => void;
  meta: string;
  onMetaText: (text: string) => void;
  metaError: string | null;
  onYamlUpdate: (u: ViewUpdate) => void;
}

/**
 * A floating window over the page for the Markdown document or the raw meta.yaml. Both stay in sync with the inline
 * view underneath, which is the live preview while typing. Drag it by its title bar.
 */
export default function FloatingEditor({
  mode,
  onMode,
  onClose,
  problem,
  onMeta,
  meta,
  onMetaText,
  metaError,
  onYamlUpdate,
}: Props) {
  const { t } = useLang();
  const [pos, setPos] = useState(() => ({ x: Math.max(16, window.innerWidth - 600), y: 72 }));
  const drag = useRef<{ dx: number; dy: number } | null>(null);

  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (!drag.current) return;
      const x = Math.min(Math.max(0, e.clientX - drag.current.dx), window.innerWidth - 120);
      const y = Math.min(Math.max(0, e.clientY - drag.current.dy), window.innerHeight - 48);
      setPos({ x, y });
    };
    const up = () => (drag.current = null);
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("keydown", key);
    };
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-label={mode === "yaml" ? "meta.yaml" : "Markdown"}
      data-testid="floating-editor"
      style={{ left: pos.x, top: pos.y }}
      className="fixed z-40 flex h-[min(70dvh,640px)] w-[min(580px,calc(100vw-2rem))] flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-2xl"
    >
      <div
        onPointerDown={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          drag.current = { dx: e.clientX - pos.x, dy: e.clientY - pos.y };
        }}
        className="flex shrink-0 cursor-move items-center justify-between gap-2 border-b border-gray-200 bg-gray-50 px-3 py-1.5 select-none"
      >
        <div className="flex items-center gap-1 text-xs" role="tablist">
          {(["markdown", "yaml"] as const).map((m) => (
            <button
              key={m}
              role="tab"
              aria-selected={mode === m}
              onClick={() => onMode(m)}
              className={`rounded px-2.5 py-1 font-medium ${mode === m ? "bg-white text-blue-700 shadow-sm" : "text-gray-500 hover:text-gray-800"}`}
            >
              {m === "yaml" ? "meta.yaml" : "Markdown"}
            </button>
          ))}
          <span className="ml-2 text-gray-400">{t("edit.floatSync")}</span>
        </div>
        <button
          onClick={onClose}
          title={t("plot.close")}
          className="rounded px-1.5 text-lg leading-none text-gray-400 hover:text-gray-700"
        >
          ×
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {mode === "markdown" ? (
          <MarkdownEditor problem={problem} onMeta={onMeta} preview={false} />
        ) : (
          <div className="overflow-hidden rounded-xl border border-gray-200" data-testid="ws-meta">
            <CodeMirror
              value={meta}
              extensions={[yaml()]}
              theme="light"
              basicSetup={{ foldGutter: false }}
              onChange={onMetaText}
              onUpdate={onYamlUpdate}
              className="text-sm"
            />
            {metaError && (
              <p className="border-t border-red-100 bg-red-50 px-4 py-2 text-xs text-red-700">
                ⚠ {t("edit.yamlError")}: {metaError}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
