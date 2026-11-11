// Hash routes of the pages with sub-tabs: #/concept, #/skills and #/agents, followed by /<tab>[/<section>]; Further
// reading (#/reading/<tab>) at the end.
// Pure helpers (no DOM), unit-tested in frontend/tests/pageRoute.test.ts.

export const PAGE_TABS = {
  concept: ["laws", "feynman", "maths", "dataflow", "limitations", "history"],
  skills: ["techniques", "loops", "blas", "nvlink", "cpuml"],
  agents: ["skills", "mcp"],
} as const;
export type TabbedPage = keyof typeof PAGE_TABS;
export type ConceptTab = (typeof PAGE_TABS.concept)[number];
export type SkillsTab = (typeof PAGE_TABS.skills)[number];
export type AgentsTab = (typeof PAGE_TABS.agents)[number];

/** localStorage keys remembering the last tab, used only when the address names none. */
export const TAB_STORE: Record<TabbedPage, string> = {
  concept: "tensorizer.conceptTab",
  skills: "tensorizer.skillsTab",
  agents: "tensorizer.agentsTab",
};
// Values stored before the tabs had URL ids.
const LEGACY: Record<string, string> = { law: "laws", flow: "dataflow" };

export interface PageRoute {
  page: TabbedPage;
  /** the tab named in the address, as written (may be unknown); null when absent */
  tab: string | null;
  /** optional section inside the tab (an anchor), null when absent */
  section: string | null;
}

/** "skills/loops/matmul" → { page: "skills", tab: "loops", section: "matmul" }; null for any other route. */
export function parsePageRoute(route: string | null): PageRoute | null {
  if (!route) return null;
  const [page, tab, section] = route.replace(/^\/+/, "").split("/");
  if (page !== "concept" && page !== "skills" && page !== "agents") return null;
  if (page === "concept" && tab === "maths" && (section === "roof" || section === "topo"))
    return { page, tab: "dataflow", section };
  return { page, tab: tab || null, section: section || null };
}

export function isTab<P extends TabbedPage>(page: P, tab: string | null | undefined): tab is (typeof PAGE_TABS)[P][number] {
  return !!tab && (PAGE_TABS[page] as readonly string[]).includes(tab);
}

/** The tab to show: the address wins; with no tab in the address the remembered one; otherwise the first. */
export function resolveTab<P extends TabbedPage>(
  page: P,
  fromUrl: string | null,
  remembered: string | null,
): (typeof PAGE_TABS)[P][number] {
  if (fromUrl !== null) return isTab(page, fromUrl) ? fromUrl : PAGE_TABS[page][0];
  const r = remembered ? (LEGACY[remembered] ?? remembered) : null;
  return isTab(page, r) ? r : PAGE_TABS[page][0];
}

/** Hash (without "#") for a page tab: "/skills/loops", or "/skills/loops/matmul" with a section. */
export function formatPageRoute(page: TabbedPage, tab: string, section?: string | null): string {
  return `/${page}/${tab}${section ? `/${section}` : ""}`;
}

// Further reading (#/reading/<tab>[/<group>]): Books & papers, Code labs, Explorables. #/reading alone always opens
// the books (no remembered tab). Older addresses still land on the books: #/reading/<group> (before the tabs) and
// #/concept/reading[/<group>] (when Further reading was a Concept tab).
export const READING_TAB_IDS = ["books", "labs", "explorables"] as const;
export type ReadingTab = (typeof READING_TAB_IDS)[number];

export interface ReadingRoute {
  tab: ReadingTab;
  /** a book group to scroll to (books only), null when absent */
  section: string | null;
  /** the canonical hash (without "#") for this route; differs from the address for legacy or bare routes */
  canonical: string;
}

/** "reading/labs" → { tab: "labs", … }; "reading", "reading/gpu", "concept/reading/gpu" → books; null otherwise. */
export function parseReadingRoute(route: string | null): ReadingRoute | null {
  if (!route) return null;
  let parts = route.replace(/^\/+/, "").split("/");
  if (parts[0] === "concept" && parts[1] === "reading") parts = parts.slice(1);
  if (parts[0] !== "reading") return null;
  const [, first, second] = parts;
  const isReadingTab = (s: string | undefined): s is ReadingTab => !!s && (READING_TAB_IDS as readonly string[]).includes(s);
  const tab: ReadingTab = isReadingTab(first) ? first : "books";
  const section = (tab === "books" ? (isReadingTab(first) ? second : first) : null) || null;
  return { tab, section, canonical: formatReadingRoute(tab, section) };
}

/** Hash (without "#") for a Further reading tab: "/reading/labs", or "/reading/books/gpu" with a group. */
export function formatReadingRoute(tab: ReadingTab, section?: string | null): string {
  return `/reading/${tab}${section ? `/${section}` : ""}`;
}
