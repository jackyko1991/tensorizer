/**
 * Text of the Concept and Coding Skills pages. English lives in pages.en.json; each translation is
 * pages.<lang>.json with the same keys (validated by `python scripts/i18n.py pages-import`). Missing keys
 * fall back to English.
 */
import { useLang } from "../i18n";
import en from "./pages.en.json";

const translations = import.meta.glob<Record<string, string>>("./pages.*.json", { eager: true, import: "default" });

export type PageKey = keyof typeof en;

export function usePageText(): (key: PageKey) => string {
  const { lang } = useLang();
  const tr = translations[`./pages.${lang}.json`] ?? {};
  return (key) => tr[key] ?? en[key];
}
