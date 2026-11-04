/** A library problem's bundled real data (data/*): its source lines, size and a preview chosen by the file's kind
 *  (a scrollable table for csv/tsv, plain text for txt/json; other kinds get the card and a download for now).
 *  Shown in the Data tab of the lower panel, next to the formulation. The file is fetched only when
 *  shown, and kept for the session. */
import { useEffect, useState, type ReactNode } from "react";
import { assetUrl } from "../api";
import { useLang } from "../i18n";
import type { Dataset } from "../types";
import DownloadIcon from "./DownloadIcon";
import { formatSize } from "./DataPanel";

const PREVIEW_ROWS = 200;
const cache = new Map<string, Promise<string>>();

function load(url: string): Promise<string> {
  if (!cache.has(url)) {
    const p = fetch(url).then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))));
    p.catch(() => cache.delete(url)); // retry next time
    cache.set(url, p);
  }
  return cache.get(url)!;
}

/** One CSV line → cells (double-quoted fields may contain the separator). */
function cells(line: string, sep: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (const ch of line) {
    if (ch === '"') quoted = !quoted;
    else if (ch === sep && !quoted) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out.map((c) => c.trim());
}

/** URLs, doi:… and arXiv:… in a source line become links. */
function linkify(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const re = /(https?:\/\/[^\s),;]+)|\bdoi:\s*(10\.[^\s),;]+)|\barXiv:\s*([\w.\-/]+\d)/gi;
  let last = 0;
  for (const m of text.matchAll(re)) {
    const i = m.index ?? 0;
    parts.push(text.slice(last, i));
    const href = m[1] ?? (m[2] ? `https://doi.org/${m[2]}` : `https://arxiv.org/abs/${m[3]}`);
    parts.push(
      <a key={i} href={href} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
        {m[0]}
      </a>,
    );
    last = i + m[0].length;
  }
  parts.push(text.slice(last));
  return parts;
}

export default function DataView({ dataset, tall = false }: { dataset: Dataset; tall?: boolean }) {
  const { t } = useLang();
  const url = assetUrl(dataset.url);
  const [text, setText] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const readable = dataset.kind === "table" || dataset.kind === "text";
  useEffect(() => {
    let live = true;
    setText(null);
    setFailed(false);
    if (!readable) return;
    load(url).then(
      (s) => live && setText(s),
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [url, readable]);

  const table = dataset.kind === "table";
  const sep = dataset.name.endsWith(".tsv") ? "\t" : ",";
  const body = text?.split(/\r?\n/).filter((l) => l.trim() && !l.startsWith("#")) ?? [];
  const rows = table ? body.slice(1, PREVIEW_ROWS + 1).map((l) => cells(l, sep)) : [];
  const shape = table
    ? t("dataset.shape").replace("{rows}", String(dataset.rows)).replace("{cols}", String(dataset.columns.length))
    : dataset.kind === "text"
      ? t("dataset.lines").replace("{n}", String(dataset.rows))
      : dataset.kind;

  return (
    <div
      className="flex min-h-0 flex-col overflow-hidden rounded-lg border border-gray-200 bg-white"
      data-testid="dataset"
    >
      <div className="flex items-center gap-3 border-b border-gray-100 px-3 py-2 text-xs">
        <span className="font-mono font-medium text-gray-800">📊 {dataset.name}</span>
        <span className="text-gray-500">
          {shape} / {formatSize(dataset.size)}
        </span>
        <a
          href={url}
          download={dataset.name.split("/").pop()}
          className="ml-auto flex items-center gap-1 text-blue-700 hover:underline"
        >
          <DownloadIcon className="h-3.5 w-3.5" />
          {t("dataset.download")}
        </a>
      </div>
      {dataset.comments.length > 0 && (
        <ul className="space-y-0.5 border-b border-gray-100 bg-gray-50/70 px-3 py-2 text-xs break-words text-gray-600">
          {dataset.comments.map((c, i) => (
            <li key={i}>{linkify(c)}</li>
          ))}
        </ul>
      )}
      <div className={`min-h-0 overflow-auto ${tall ? "flex-1" : "max-h-72"}`}>
        {/* one renderer per kind: table and text today; image / array canvases can slot in here */}
        {!readable ? (
          <p className="p-3 text-xs text-gray-500">{t("dataset.noPreview")}</p>
        ) : failed ? (
          <p className="p-3 text-xs text-red-600">{t("dataset.failed")}</p>
        ) : text === null ? (
          <p className="p-3 text-xs text-gray-400">…</p>
        ) : table ? (
          <table className="min-w-full border-collapse font-mono text-[11px]">
            <thead className="sticky top-0 bg-gray-100 text-gray-700">
              <tr>
                {dataset.columns.map((c) => (
                  <th key={c} className="border-b border-gray-200 px-2 py-1 text-left font-semibold whitespace-nowrap">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="odd:bg-white even:bg-gray-50/60">
                  {dataset.columns.map((_, j) => (
                    <td key={j} className="px-2 py-0.5 whitespace-nowrap text-gray-800">
                      {r[j] ?? ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <pre className="p-3 font-mono text-[11px] whitespace-pre-wrap text-gray-800">
            {body.slice(0, PREVIEW_ROWS).join("\n")}
          </pre>
        )}
        {(table ? dataset.rows : body.length) > PREVIEW_ROWS && text !== null && (
          <p className="px-3 py-1.5 text-[11px] text-gray-400">
            {t("dataset.more").replace("{n}", String(PREVIEW_ROWS))}
          </p>
        )}
      </div>
    </div>
  );
}

/** The Data tab of the lower panel: every bundled data file, or a note that the problem has none. */
export function DataSources({ datasets }: { datasets: Dataset[] }) {
  const { t } = useLang();
  if (!datasets.length)
    return (
      <p
        className="rounded-xl border border-dashed border-gray-300 p-6 text-center text-sm text-gray-400"
        data-testid="no-data"
      >
        {t("dataset.none")}
      </p>
    );
  return (
    <section data-testid="data-sources">
      <div className="space-y-3">
        {datasets.map((d) => (
          <DataView key={d.name} dataset={d} />
        ))}
      </div>
    </section>
  );
}
