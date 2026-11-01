/** A Workspace problem's uploaded data: drop files in, see their size and a preview, rename or delete them.
 *  The bytes live in the browser (IndexedDB) and are synced into the sandbox's data/ folder for the code, the
 *  notebooks and the AI. */
import { useEffect, useMemo, useRef, useState } from "react";
import { emitContext } from "../agentBus";
import { useLang } from "../i18n";
import type { DataFile, Params } from "../types";
import { DATA_EXTENSIONS, getData, MAX_DATA_TOTAL } from "../workspaceStore";
import { RichText } from "./Tex";

interface Props {
  problemId: string;
  localId: string;
  problemTitle: string;
  params: Params;
  files: DataFile[];
  onUpload: (files: File[]) => Promise<void>;
  onRename: (name: string) => void;
  onDelete: (name: string) => void;
}

export const formatSize = (n: number) =>
  n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(1)} kB` : `${(n / 1024 ** 2).toFixed(1)} MB`;

const IMAGE = /\.(png|jpe?g)$/i;
const TABLE = /\.(csv|tsv|txt|dat)$/i;

export default function DataPanel({ problemId, localId, problemTitle, params, files, onUpload, onRename, onDelete }: Props) {
  const { t } = useLang();
  const [selected, setSelected] = useState<string | null>(files[0]?.name ?? null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const used = files.reduce((n, f) => n + f.size, 0);
  const current = files.find((f) => f.name === selected) ?? files[0] ?? null;

  const paramsKey = JSON.stringify(params);
  useEffect(() => {
    emitContext({
      problem: problemId,
      title: problemTitle.replace(/\$/g, ""),
      edits: {},
      params: JSON.parse(paramsKey),
      workspace: { kind: "data" },
    });
  }, [problemId, problemTitle, paramsKey]);

  async function upload(list: FileList | File[] | null) {
    if (!list?.length) return;
    setBusy(true);
    try {
      await onUpload([...list]);
      setSelected(list[list.length - 1].name);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-gray-50/60 lg:min-h-0" data-testid="ws-data">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-gray-200 bg-white px-4 py-2">
        <span className="text-lg">📊</span>
        <h2 className="text-base font-semibold text-gray-900">{t("data.title")}</h2>
        <span className="hidden truncate text-xs text-gray-400 sm:inline">
          <RichText text={problemTitle} />
        </span>
        <span className="ml-auto text-xs text-gray-500" data-testid="data-usage">
          {t("data.usage").replace("{used}", formatSize(used)).replace("{max}", formatSize(MAX_DATA_TOTAL))}
        </span>
      </div>
      <div className="h-1 w-full bg-gray-100">
        <div className="h-1 bg-blue-500" style={{ width: `${Math.min(100, (used / MAX_DATA_TOTAL) * 100)}%` }} />
      </div>

      <div className="space-y-4 p-3 sm:p-5 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">
        <button
          type="button"
          onClick={() => input.current?.click()}
          onDragOver={(e) => (e.preventDefault(), setDrag(true))}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            upload(e.dataTransfer.files);
          }}
          className={`flex w-full flex-col items-center gap-1 rounded-xl border-2 border-dashed px-4 py-6 text-sm ${
            drag ? "border-blue-400 bg-blue-50 text-blue-700" : "border-gray-300 bg-white text-gray-600 hover:border-blue-300"
          }`}
        >
          <span className="text-2xl">⬆</span>
          <span className="font-medium">{busy ? t("data.uploading") : t("data.drop")}</span>
          <span className="text-xs text-gray-400">{t("data.types")}</span>
        </button>
        <input
          ref={input}
          type="file"
          multiple
          accept={DATA_EXTENSIONS.join(",")}
          className="hidden"
          data-testid="data-input"
          onChange={(e) => {
            upload(e.target.files ? [...e.target.files] : null);
            e.target.value = "";
          }}
        />
        <p className="text-xs text-gray-500">
          <RichText text={t("data.hint")} />
        </p>

        {files.length === 0 ? (
          <p className="py-4 text-center text-sm text-gray-400">{t("data.empty")}</p>
        ) : (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,16rem)_minmax(0,1fr)]">
            <ul className="space-y-1" data-testid="data-list">
              {files.map((f) => (
                <li
                  key={f.name}
                  className={`group flex items-center gap-2 rounded-lg border px-2 py-1.5 text-sm ${
                    current?.name === f.name ? "border-blue-300 bg-blue-50" : "border-gray-200 bg-white hover:bg-gray-50"
                  }`}
                >
                  <button onClick={() => setSelected(f.name)} className="min-w-0 flex-1 text-left">
                    <span className="block truncate font-mono text-xs text-gray-800">{f.name}</span>
                    <span className="block text-[11px] text-gray-400">{formatSize(f.size)}</span>
                  </button>
                  <button
                    onClick={() => onRename(f.name)}
                    title={t("ws.rename")}
                    aria-label={t("ws.rename")}
                    className="rounded px-1 text-xs text-gray-500 hover:bg-gray-200"
                  >
                    ✎
                  </button>
                  <button
                    onClick={() => onDelete(f.name)}
                    title={t("ws.delete")}
                    aria-label={t("ws.delete")}
                    className="rounded px-1 text-xs text-gray-500 hover:bg-red-50 hover:text-red-600"
                  >
                    🗑
                  </button>
                </li>
              ))}
            </ul>
            {current && <Preview key={current.name + current.size} localId={localId} file={current} />}
          </div>
        )}
      </div>
    </main>
  );
}

/** First rows of a table, an image thumbnail, the start of a text file, or just the size. */
function Preview({ localId, file }: { localId: string; file: DataFile }) {
  const { t } = useLang();
  const [blob, setBlob] = useState<Blob | null>(null);
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    getData(localId, file.name).then((b) => {
      if (!live || !b) return;
      setBlob(b);
      if (TABLE.test(file.name) || /\.json$/i.test(file.name)) b.slice(0, 64 * 1024).text().then((s) => live && setText(s));
    });
    return () => {
      live = false;
    };
  }, [localId, file.name]);
  const url = useMemo(() => (blob && IMAGE.test(file.name) ? URL.createObjectURL(blob) : null), [blob, file.name]);
  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url]);
  const rows = useMemo(() => {
    if (text === null || !TABLE.test(file.name)) return null;
    const sep = /\.tsv$/i.test(file.name) || (text.split("\n")[0].includes("\t") && !text.includes(",")) ? "\t" : ",";
    return text
      .split(/\r?\n/)
      .filter((l) => l.trim())
      .slice(0, 11)
      .map((l) => l.split(sep).slice(0, 12));
  }, [text, file.name]);

  return (
    <div className="min-w-0 rounded-xl border border-gray-200 bg-white p-3" data-testid="data-preview">
      <div className="mb-2 flex items-center gap-2 text-xs text-gray-500">
        <span className="font-mono text-gray-800">data/{file.name}</span>
        <span>· {formatSize(file.size)}</span>
      </div>
      {url ? (
        <img src={url} alt={file.name} className="max-h-80 max-w-full rounded border border-gray-100" />
      ) : rows ? (
        <div className="overflow-x-auto">
          <table className="text-xs">
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className={i === 0 ? "font-semibold text-gray-700" : "border-t border-gray-100 text-gray-600"}>
                  {r.map((c, j) => (
                    <td key={j} className="px-2 py-1 font-mono whitespace-nowrap">
                      {c}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-[11px] text-gray-400">{t("data.firstRows")}</p>
        </div>
      ) : text !== null ? (
        <pre className="max-h-80 overflow-auto text-xs whitespace-pre-wrap text-gray-700">{text.slice(0, 3000)}</pre>
      ) : (
        <p className="text-xs text-gray-400">{t("data.noPreview")}</p>
      )}
    </div>
  );
}
