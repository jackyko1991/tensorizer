import { yaml } from "@codemirror/lang-yaml";
import { pickText, saveText } from "../fileIO";
import { fromMarkdown, MarkdownError, pickMd, toLatex, toMarkdown } from "../metaDoc";
import { EditorView } from "@codemirror/view";
import CodeMirror, { type ViewUpdate } from "@uiw/react-codemirror";
import { useEffect, useRef, useState } from "react";
import { useLang } from "../i18n";
import type { MetaFields } from "../metaDoc";
import type { Problem } from "../types";
import { MarkdownEditor } from "./WsEditor";

export type FloatMode = "markdown" | "yaml";

const YAML_EXTENSIONS = [yaml(), EditorView.lineWrapping];
const BOX_KEY = "tensorizer.floatEditor"; // where the window was and how big: reopens the same way
interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** The saved box, kept inside the current window (it may have been resized since). */
/** First open: just to the right of the inline problem column (over the code), else at the window's right edge. */
function defaultBox(): Box {
  const inline = document.querySelector("[data-ws-inline]")?.getBoundingClientRect();
  const h = Math.min(640, window.innerHeight * 0.7);
  if (inline && window.innerWidth - inline.right >= 340) {
    const y = Math.max(8, inline.top + 8);
    return {
      x: inline.right + 8,
      y,
      w: Math.min(580, window.innerWidth - inline.right - 24),
      h: Math.min(h, window.innerHeight - y - 16),
    };
  }
  const w = Math.min(580, window.innerWidth - 32);
  return { x: window.innerWidth - w - 24, y: 72, w, h };
}

function loadBox(): Box {
  let box = defaultBox();
  try {
    const saved = JSON.parse(localStorage.getItem(BOX_KEY) ?? "null");
    if (saved && [saved.x, saved.y, saved.w, saved.h].every(Number.isFinite)) box = saved;
  } catch {
    // nothing saved (or storage blocked): the default box
  }
  const w = Math.min(Math.max(box.w, 320), window.innerWidth - 16);
  const h = Math.min(Math.max(box.h, 200), window.innerHeight - 16);
  return {
    w,
    h,
    x: Math.min(Math.max(0, box.x), window.innerWidth - w),
    y: Math.min(Math.max(0, box.y), window.innerHeight - 48),
  };
}
function saveBox(box: Box) {
  try {
    localStorage.setItem(BOX_KEY, JSON.stringify(box));
  } catch {
    // private window: the box lasts until the page closes
  }
}

/** The split button in a Workspace problem's header: opens the floating Markdown or meta.yaml editor. */
export function EditorButtons({ open, onOpen }: { open: FloatMode | null; onOpen: (m: FloatMode | null) => void }) {
  const { t } = useLang();
  return (
    <div
      className="inline-flex shrink-0 overflow-hidden rounded-md border border-gray-200 text-xs"
      title={t("edit.floatHint")}
    >
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
const IO = "rounded px-1.5 py-0.5 text-gray-500 hover:bg-white hover:text-blue-700";

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
  const [pos, setPos] = useState(loadBox);
  const drag = useRef<{ dx: number; dy: number } | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const box = useRef(pos); // the live box: updated by drags and resizes, saved when they end

  // Resizing (the corner handle, CSS resize) is read back with a ResizeObserver and remembered with the position.
  useEffect(() => {
    const el = frame.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      const next = { ...box.current, w: el.offsetWidth, h: el.offsetHeight };
      if (next.w !== box.current.w || next.h !== box.current.h) {
        box.current = next;
        saveBox(next);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (!drag.current) return;
      const x = Math.min(Math.max(0, e.clientX - drag.current.dx), window.innerWidth - 120);
      const y = Math.min(Math.max(0, e.clientY - drag.current.dy), window.innerHeight - 48);
      const next = { ...box.current, x, y };
      box.current = next; // the latest box right away (renders lag behind the pointer)
      setPos(next);
    };
    const up = () => {
      if (drag.current) saveBox(box.current);
      drag.current = null;
    };
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

  const name = problem.slug || "problem";
  const fields = () => pickMd({ ...problem, notes: problem.notes ?? "" });
  const [ioError, setIoError] = useState<string | null>(null);
  async function importMd() {
    const text = await pickText(".md,.markdown,text/markdown,text/plain");
    if (text === null) return;
    try {
      onMeta(fromMarkdown(text)); // the same parser as typing in the editor: a bad file changes nothing
      setIoError(null);
    } catch (e) {
      setIoError(e instanceof MarkdownError ? e.message : String(e));
    }
  }
  async function uploadYaml() {
    const text = await pickText(".yaml,.yml,text/yaml,text/plain");
    if (text !== null) onMetaText(text); // re-parsed on the server like an edit; errors show below the editor
  }

  return (
    <div
      role="dialog"
      aria-label={mode === "yaml" ? "meta.yaml" : "Markdown"}
      data-testid="floating-editor"
      ref={frame}
      style={{ left: pos.x, top: pos.y, width: pos.w, height: pos.h, resize: "both", minWidth: 320, minHeight: 200 }}
      className="fixed z-40 flex max-h-[calc(100dvh-1rem)] max-w-[calc(100vw-1rem)] flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-2xl"
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
          <span className="ml-2 hidden text-gray-400 sm:inline">{t("edit.floatSync")}</span>
        </div>
        <div className="ml-auto flex items-center gap-0.5 text-[11px]">
          {mode === "markdown" ? (
            <>
              <button onClick={importMd} className={IO} title={t("edit.importMd")} data-testid="import-md">
                ⬆ .md
              </button>
              <button onClick={() => saveText(`${name}.md`, toMarkdown(fields()), "text/markdown")} className={IO}>
                ⬇ .md
              </button>
              <button onClick={() => saveText(`${name}.tex`, toLatex(fields()), "application/x-tex")} className={IO}>
                ⬇ .tex
              </button>
            </>
          ) : (
            <>
              <button onClick={uploadYaml} className={IO} title={t("edit.uploadYaml")} data-testid="upload-yaml">
                ⬆ meta.yaml
              </button>
              <button onClick={() => saveText("meta.yaml", meta, "text/yaml")} className={IO}>
                ⬇ meta.yaml
              </button>
            </>
          )}
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
        {ioError && mode === "markdown" && (
          <p className="mb-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-800">
            ⚠ {t("edit.mdError")}: {ioError}
          </p>
        )}
        {mode === "markdown" ? (
          <MarkdownEditor problem={problem} onMeta={onMeta} preview={false} />
        ) : (
          <div className="overflow-hidden rounded-xl border border-gray-200" data-testid="ws-meta">
            <CodeMirror
              value={meta}
              extensions={YAML_EXTENSIONS}
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
