/** Collections: ordered stories (series) and automatic groups built from library tags (prizes, problem lists).
 *
 *  A series is declared by tags on each problem, `series:<id>:<step>` plus an optional `role:<kind>`, and described
 *  (title, author, summary per language) in content/series.json → manifest.series. Automatic collections need no
 *  data: every problem with a matching tag joins, ordered by the number or year in the tag. */
import type { Key } from "./i18n";
import type { Manifest, Problem } from "./types";

export const ROLES = [
  "postulate",
  "theory",
  "experiment",
  "observation",
  "definition",
  "conjecture",
  "theorem",
  "proof",
  "background",
] as const;
export type Role = (typeof ROLES)[number];

export const ROLE_STYLE: Record<Role, string> = {
  postulate: "bg-purple-100 text-purple-800",
  theory: "bg-blue-100 text-blue-800",
  experiment: "bg-emerald-100 text-emerald-800",
  observation: "bg-teal-100 text-teal-800",
  definition: "bg-slate-100 text-slate-700",
  conjecture: "bg-fuchsia-100 text-fuchsia-800",
  theorem: "bg-indigo-100 text-indigo-800",
  proof: "bg-cyan-100 text-cyan-800",
  background: "bg-gray-100 text-gray-600",
};

export interface CollectionItem {
  problem: Problem;
  /** Position in a series, or the number / year in the tag that put it here. */
  step: number;
  role?: Role;
}

export interface Collection {
  id: string;
  kind: "series" | "tags";
  title: string;
  author?: string;
  summary?: string;
  items: CollectionItem[];
}

type T = (key: Key) => string;

/** Automatic collections: tag prefix → title key. The number after the prefix (problem number or year) orders them. */
const AUTO: { id: string; match: (tag: string) => number | null; title: Key }[] = [
  { id: "millennium", match: (t) => (t === "millennium" ? 0 : null), title: "coll.millennium" },
  { id: "hilbert", match: (t) => num(t, "hilbert"), title: "coll.hilbert" },
  { id: "smale", match: (t) => num(t, "smale"), title: "coll.smale" },
  { id: "landau", match: (t) => (t === "landau" ? 0 : null), title: "coll.landau" },
  { id: "nobel", match: (t) => num(t, "nobel"), title: "coll.nobel" },
  { id: "fields", match: (t) => num(t, "fields"), title: "coll.fields" },
  { id: "abel", match: (t) => num(t, "abel"), title: "coll.abel" },
  { id: "turing", match: (t) => num(t, "turing"), title: "coll.turing" },
  { id: "shaw", match: (t) => num(t, "shaw"), title: "coll.shaw" },
];

/** "nobel:physics:2013" → 2013, "hilbert:8" → 8, for the given prefix; null when the tag is something else. */
function num(tag: string, prefix: string): number | null {
  const [head, ...rest] = tag.split(":");
  if (head !== prefix) return null;
  const n = Number(rest.at(-1));
  return Number.isFinite(n) ? n : 0;
}

const pick = (text: Record<string, string> | undefined, lang: string) => text?.[lang] ?? text?.en;

export function buildCollections(manifest: Manifest, lang: string, t: T): Collection[] {
  const series = new Map<string, CollectionItem[]>();
  for (const p of manifest.problems) {
    const role = p.tags?.find((x) => x.startsWith("role:"))?.slice(5) as Role | undefined;
    for (const tag of p.tags ?? []) {
      const m = tag.match(/^series:([\w-]+):(\d+)$/);
      if (!m) continue;
      if (!series.has(m[1])) series.set(m[1], []);
      series.get(m[1])!.push({ problem: p, step: Number(m[2]), role: role && ROLES.includes(role) ? role : undefined });
    }
  }
  const out: Collection[] = [...series].map(([id, items]) => {
    const info = manifest.series?.[id];
    return {
      id,
      kind: "series" as const,
      title: pick(info?.title, lang) ?? id,
      author: info?.author,
      summary: pick(info?.summary, lang),
      items: items.sort((a, b) => a.step - b.step),
    };
  });
  for (const auto of AUTO) {
    const items: CollectionItem[] = [];
    for (const p of manifest.problems) {
      const steps = (p.tags ?? []).map(auto.match).filter((n): n is number => n !== null);
      if (steps.length) items.push({ problem: p, step: Math.min(...steps) });
    }
    if (items.length)
      out.push({ id: auto.id, kind: "tags", title: t(auto.title), items: items.sort((a, b) => a.step - b.step) });
  }
  return out;
}

/** The series (not automatic collections) a problem belongs to, with its position, for the step navigator. */
export function seriesOf(collections: Collection[], problemId: string) {
  return collections
    .filter((c) => c.kind === "series")
    .map((c) => ({ collection: c, index: c.items.findIndex((i) => i.problem.id === problemId) }))
    .filter((x) => x.index >= 0);
}
