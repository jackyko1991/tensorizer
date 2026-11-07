// Hash routes of the pages with sub-tabs: #/concept[/<tab>[/<section>]] and #/skills[/<tab>[/<section>]].
// Pure helpers (no DOM), unit-tested in frontend/tests/pageRoute.test.ts.

export const PAGE_TABS = {
  concept: ["laws", "feynman", "maths", "dataflow", "history"],
  skills: ["techniques", "loops", "blas", "nvlink", "cpuml"],
} as const;
export type TabbedPage = keyof typeof PAGE_TABS;
export type ConceptTab = (typeof PAGE_TABS.concept)[number];
export type SkillsTab = (typeof PAGE_TABS.skills)[number];

/** localStorage keys remembering the last tab, used only when the address names none. */
export const TAB_STORE: Record<TabbedPage, string> = {
  concept: "tensorizer.conceptTab",
  skills: "tensorizer.skillsTab",
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
  if (page !== "concept" && page !== "skills") return null;
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
