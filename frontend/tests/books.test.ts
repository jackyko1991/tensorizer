// The "Further reading" tab: affiliate links built from one store ID, valid ISBN-10s, top picks first, rel="sponsored",
// the Amazon disclosure and every reading.* key in all languages, no em dashes. Run: node --test frontend/tests/*.test.ts
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { test } from "node:test";
import { AFFILIATE_REL, AMAZON_TAG, BOOK_GROUPS, BOOKS, amazonUrl, isValidIsbn10, isbn13to10 } from "../src/books.ts";

const root = new URL("../src/", import.meta.url);
const read = (p: string) => readFileSync(new URL(p, root), "utf-8");
const dir = new URL("content/", root);
const langs = readdirSync(dir).filter((f) => /^pages\.[\w-]+\.json$/.test(f));
const text = (f: string) => JSON.parse(readFileSync(new URL(f, dir), "utf-8")) as Record<string, string>;
const DASH = /[—―]| – /;

test("the store ID is tensorizer-21 and every link is amazon.co.uk/dp/<ISBN-10>?tag=<store ID>", () => {
  assert.equal(AMAZON_TAG, "tensorizer-21");
  assert.ok(BOOKS.length >= 18, `books: ${BOOKS.length}`);
  for (const b of BOOKS) {
    const url = new URL(amazonUrl(b.isbn10));
    assert.equal(url.origin, "https://www.amazon.co.uk", b.id);
    assert.equal(url.pathname, `/dp/${b.isbn10}`, b.id);
    assert.equal(url.searchParams.get("tag"), AMAZON_TAG, b.id);
    assert.ok(amazonUrl(b.isbn10).includes(`tag=${AMAZON_TAG}`), b.id);
  }
});

test("the store ID is written only once, in books.ts", () => {
  const src = read("books.ts");
  assert.equal(src.split(AMAZON_TAG).length - 1, 1, "books.ts writes the tag more than once");
  assert.ok(!read("pages/ReadingConcept.tsx").includes(AMAZON_TAG), "the component hand-writes the tag");
  assert.ok(!src.includes("amazon.co.uk/dp/0") && !/dp\/\d{9}[\dX]\?/.test(src), "a hand-written Amazon URL");
});

test("ISBN-10 checksum: known values pass, altered ones fail", () => {
  assert.ok(isValidIsbn10("0521171903"));
  assert.ok(isValidIsbn10("009917331X"));
  assert.ok(!isValidIsbn10("0521171904"));
  assert.ok(!isValidIsbn10("052117190"));
  assert.ok(!isValidIsbn10("X521171903"));
  assert.equal(isbn13to10("9780099173311"), "009917331X");
});

test("every book's ISBN-10 has a valid checksum and matches its ISBN-13", () => {
  for (const b of BOOKS) {
    assert.ok(isValidIsbn10(b.isbn10), `${b.id}: bad ISBN-10 ${b.isbn10}`);
    assert.equal(isbn13to10(b.isbn13), b.isbn10, `${b.id}: ISBN-13 ${b.isbn13} does not give ${b.isbn10}`);
  }
  assert.equal(new Set(BOOKS.map((b) => b.isbn10)).size, BOOKS.length, "duplicate ISBN");
  assert.equal(new Set(BOOKS.map((b) => b.id)).size, BOOKS.length, "duplicate id");
});

test("every group has books, the ★ top picks come first within their group, and the seven picks are the ★ ones", () => {
  for (const g of BOOK_GROUPS) {
    const tops = BOOKS.filter((b) => b.group === g).map((b) => b.top);
    assert.ok(tops.length > 0 && tops[0], `${g}: no top pick first`);
    assert.equal(tops.indexOf(false) === -1 ? tops.length : tops.indexOf(false), tops.lastIndexOf(true) + 1, `${g}: a ★ book after a plain one`);
  }
  assert.deepEqual(
    BOOKS.filter((b) => b.top).map((b) => b.id),
    ["fleisch", "grinfeld", "lectures", "qed", "pmpp", "udl", "chipwar"],
  );
});

test("affiliate links open in a new tab with rel=\"sponsored noopener noreferrer\"", () => {
  assert.equal(AFFILIATE_REL, "sponsored noopener noreferrer");
  const src = read("pages/ReadingConcept.tsx");
  assert.match(src, /href=\{amazonUrl\(b\.isbn10\)\}\s+target="_blank"\s+rel=\{AFFILIATE_REL\}/);
  // no prices, ratings, covers or Amazon scripts/widgets
  const code = src.replace(/\/\*[\s\S]*?\*\//g, ""); // not the doc comment, which says what is left out
  assert.ok(!/<img|<script|<iframe|amazon-adsystem|£|\bprice\b|rating/i.test(code), "no covers, widgets, prices or ratings");
});

test("site-tab links point at real tabs, free editions are https", () => {
  for (const b of BOOKS) {
    assert.match(b.site.href, /^#\/(concept\/(laws|feynman|maths|dataflow|history)|skills\/(techniques|loops|blas|nvlink|cpuml))(\/[\w-]+)?$/, b.id);
    if (b.free) assert.match(b.free, /^https:\/\//, b.id);
  }
  assert.deepEqual(BOOKS.filter((b) => b.free).map((b) => b.id).sort(), ["bishop", "lectures", "udl"]);
});

test("the disclosure sentence is on the tab in all languages, and English is the exact Amazon wording", () => {
  assert.ok(langs.length >= 10, `languages: ${langs}`);
  assert.equal(text("pages.en.json")["reading.disclosure"], "As an Amazon Associate I earn from qualifying purchases.");
  for (const f of langs) {
    const t = text(f);
    assert.ok(t["reading.disclosure"]?.includes("Amazon"), `${f}: reading.disclosure`);
    assert.ok(t["reading.note"], `${f}: reading.note`);
  }
  assert.ok(read("pages/ReadingConcept.tsx").includes('T("reading.disclosure")'));
});

test("every reading.* key the tab uses exists in all languages", () => {
  const keys = new Set<string>(["reading.tab"]);
  for (const f of ["pages/ReadingConcept.tsx", "pages/ConceptPage.tsx"])
    for (const m of read(f).matchAll(/["'`](reading\.[A-Za-z0-9_.]+)["'`]/g)) keys.add(m[1]);
  for (const g of BOOK_GROUPS) keys.add(`reading.group.${g}`);
  for (const b of BOOKS) keys.add(`reading.book.${b.id}.why`);
  assert.ok(keys.size >= 32, `found only ${keys.size} keys`);
  for (const f of langs) {
    const t = text(f);
    const missing = [...keys].filter((k) => typeof t[k] !== "string" || !t[k]);
    assert.deepEqual(missing, [], `${f} lacks ${missing.join(", ")}`);
    // the quoted link label in the note is the link label itself
    assert.ok(t["reading.note"].includes(t["reading.amazon"]), `${f}: reading.note does not quote reading.amazon`);
  }
});

test("no em dashes in the book data or the tab's text", () => {
  for (const b of BOOKS)
    for (const v of [b.title, b.edition ?? "", b.authors, b.publisher]) assert.ok(!DASH.test(v), `${b.id}: ${v}`);
  for (const f of langs)
    for (const [k, v] of Object.entries(text(f))) if (k.startsWith("reading.")) assert.ok(!DASH.test(v), `${f} ${k}`);
});
