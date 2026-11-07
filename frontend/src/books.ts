// Book data of the Concept tab "Further reading" (pages/ReadingConcept.tsx). Unit-tested in frontend/tests/books.test.ts.
// Each Amazon link is built from the ISBN-10 (for a print book the ASIN is its ISBN-10) and the one store ID below.
// No prices, ratings, covers or Amazon scripts: only a plain affiliate link per book.
import type { PageKey } from "./content/pages";

/** Amazon Associates store ID (Amazon UK). The only place the tag is written. */
export const AMAZON_TAG = "tensorizer-21";

/** rel of every affiliate link (paid link, per Google and the ASA/CMA guidance; no opener, no referrer). */
export const AFFILIATE_REL = "sponsored noopener noreferrer";

export const amazonUrl = (isbn10: string) => `https://www.amazon.co.uk/dp/${isbn10}?tag=${AMAZON_TAG}`;

/** ISBN-10 checksum: sum of (10 - i) * digit_i over the ten characters is divisible by 11 (last may be X = 10). */
export function isValidIsbn10(isbn: string): boolean {
  if (!/^\d{9}[\dX]$/.test(isbn)) return false;
  let sum = 0;
  for (let i = 0; i < 10; i++) sum += (10 - i) * (isbn[i] === "X" ? 10 : Number(isbn[i]));
  return sum % 11 === 0;
}

/** The ISBN-10 of a 978- ISBN-13: drop the prefix and the check digit, then add the ISBN-10 check digit. */
export function isbn13to10(isbn13: string): string | null {
  if (!/^978\d{10}$/.test(isbn13)) return null;
  const body = isbn13.slice(3, 12);
  let sum = 0;
  for (let i = 0; i < 9; i++) sum += (10 - i) * Number(body[i]);
  const c = (11 - (sum % 11)) % 11;
  return body + (c === 10 ? "X" : String(c));
}

export const BOOK_GROUPS = ["maths", "physics", "gpu", "history"] as const;
export type BookGroup = (typeof BOOK_GROUPS)[number];

export interface Book {
  /** why-line key: reading.book.<id>.why */
  id: string;
  group: BookGroup;
  /** a top pick (★): listed first within its group */
  top: boolean;
  /** title, authors and publisher stay in English in every language */
  title: string;
  edition?: string;
  authors: string;
  publisher: string;
  /** year of the printing linked */
  year: number;
  isbn10: string;
  isbn13: string;
  /** where the ISBN was checked (not shown) */
  source: string;
  /** the matching tab of this site; its label is that tab's own title */
  site: { href: string; label: PageKey };
  /** a free official online edition, checked to load (not an affiliate link) */
  free?: string;
}

export const BOOKS: Book[] = [
  // Tensors and maths
  {
    id: "fleisch",
    group: "maths",
    top: true,
    title: "A Student's Guide to Vectors and Tensors",
    authors: "Daniel A. Fleisch",
    publisher: "Cambridge University Press",
    year: 2011,
    isbn10: "0521171903",
    isbn13: "9780521171908",
    source: "https://cambridge.org/core/books/students-guide-to-vectors-and-tensors/CC1736FAE454A3BA2B56FDBBBCEF267C",
    site: { href: "#/concept/maths/tensor", label: "maths.tab" },
  },
  {
    id: "grinfeld",
    group: "maths",
    top: true,
    title: "Introduction to Tensor Analysis and the Calculus of Moving Surfaces",
    authors: "Pavel Grinfeld",
    publisher: "Springer",
    year: 2016,
    isbn10: "1493955055",
    isbn13: "9781493955053",
    source: "https://link.springer.com/10.1007/978-1-4614-7867-6 (softcover ISBN)",
    site: { href: "#/concept/maths/einsum", label: "maths.tab" },
  },
  {
    id: "strang",
    group: "maths",
    top: false,
    title: "Linear Algebra and Learning from Data",
    authors: "Gilbert Strang",
    publisher: "Wellesley-Cambridge Press",
    year: 2019,
    isbn10: "0692196382",
    isbn13: "9780692196380",
    source: "https://www.bookwire.com/book/AUS/Linear-Algebra-and-Learning-from-Data-9780692196380-Strang-Gilbert-67246919",
    site: { href: "#/skills/blas", label: "skills.tab.blas" },
  },
  {
    id: "penrose",
    group: "maths",
    top: false,
    title: "The Road to Reality: A Complete Guide to the Laws of the Universe",
    authors: "Roger Penrose",
    publisher: "Vintage",
    year: 2006,
    isbn10: "0099440687",
    isbn13: "9780099440680",
    source: "https://www.penguin.co.uk/books/357292/the-road-to-reality/9780099440680",
    site: { href: "#/concept/maths/penrose", label: "maths.tab" },
  },
  // Physics and Feynman
  {
    id: "lectures",
    group: "physics",
    top: true,
    title: "The Feynman Lectures on Physics",
    edition: "New Millennium Edition, boxed set",
    authors: "Richard P. Feynman, Robert B. Leighton & Matthew Sands",
    publisher: "Basic Books",
    year: 2011,
    isbn10: "0465023827",
    isbn13: "9780465023820",
    source: "https://www.hachettebookgroup.com/titles/richard-p-feynman/the-feynman-lectures-on-physics-boxed-set/9780465023820",
    site: { href: "#/concept/feynman", label: "feynman.tab" },
    free: "https://www.feynmanlectures.caltech.edu/",
  },
  {
    id: "qed",
    group: "physics",
    top: true,
    title: "QED: The Strange Theory of Light and Matter",
    authors: "Richard P. Feynman",
    publisher: "Princeton University Press",
    year: 2014,
    isbn10: "0691164096",
    isbn13: "9780691164090",
    source: "https://press.princeton.edu/books/paperback/9780691164090/qed",
    site: { href: "#/concept/feynman/life", label: "feynman.tab" },
  },
  {
    id: "surely",
    group: "physics",
    top: false,
    title: "\"Surely You're Joking, Mr. Feynman!\"",
    authors: "Richard P. Feynman, as told to Ralph Leighton",
    publisher: "Vintage",
    year: 1993,
    isbn10: "009917331X",
    isbn13: "9780099173311",
    source: "https://www.penguin.co.nz/books/surely-youre-joking-mr-feynman-9780099173311",
    site: { href: "#/concept/feynman/roots", label: "feynman.tab" },
  },
  {
    id: "genius",
    group: "physics",
    top: false,
    title: "Genius: The Life and Science of Richard Feynman",
    authors: "James Gleick",
    publisher: "Vintage",
    year: 1993,
    isbn10: "0679747044",
    isbn13: "9780679747048",
    source: "https://www.tatteredcover.com/products/9780679747048",
    site: { href: "#/concept/feynman/life", label: "feynman.tab" },
  },
  {
    id: "schutz",
    group: "physics",
    top: false,
    title: "A First Course in General Relativity",
    edition: "3rd edition",
    authors: "Bernard Schutz",
    publisher: "Cambridge University Press",
    year: 2022,
    isbn10: "1108492673",
    isbn13: "9781108492676",
    source: "https://blackwells.co.uk/bookshop/product/A-First-Course-in-General-Relativity-by-Bernard-F-Schutz/9781108492676",
    site: { href: "#/concept/history/field", label: "history.tab" },
  },
  // GPUs, performance and deep learning
  {
    id: "pmpp",
    group: "gpu",
    top: true,
    title: "Programming Massively Parallel Processors: A Hands-on Approach",
    edition: "4th edition",
    authors: "Wen-mei W. Hwu, David B. Kirk & Izzat El Hajj",
    publisher: "Morgan Kaufmann",
    year: 2022,
    isbn10: "0323912311",
    isbn13: "9780323912310",
    source: "https://inspectioncopy.elsevier.com/book/details/9780323912310",
    site: { href: "#/skills/loops", label: "skills.tab.loops" },
  },
  {
    id: "udl",
    group: "gpu",
    top: true,
    title: "Understanding Deep Learning",
    authors: "Simon J.D. Prince",
    publisher: "MIT Press",
    year: 2023,
    isbn10: "0262048647",
    isbn13: "9780262048644",
    source: "https://mitpressbookstore.mit.edu/book/9780262048644",
    site: { href: "#/concept/dataflow", label: "flow.tab" },
    free: "https://udlbook.github.io/udlbook/",
  },
  {
    id: "hp",
    group: "gpu",
    top: false,
    title: "Computer Architecture: A Quantitative Approach",
    edition: "7th edition",
    authors: "John L. Hennessy, David A. Patterson & Christos Kozyrakis",
    publisher: "Morgan Kaufmann",
    year: 2025,
    isbn10: "0443154066",
    isbn13: "9780443154065",
    source: "https://lehmanns.de/shop/mathematik-informatik/70726627-9780443154065-computer-architecture",
    site: { href: "#/concept/maths/roof", label: "maths.tab" },
  },
  {
    id: "pytorch",
    group: "gpu",
    top: false,
    title: "Deep Learning with PyTorch",
    edition: "2nd edition",
    authors: "Luca Antiga, Eli Stevens, Howard Huang & Thomas Viehmann",
    publisher: "Manning",
    year: 2026,
    isbn10: "1633438856",
    isbn13: "9781633438859",
    source: "https://manning.com/books/deep-learning-with-pytorch-second-edition",
    site: { href: "#/skills/cpuml", label: "skills.tab.cpuml" },
  },
  {
    id: "bishop",
    group: "gpu",
    top: false,
    title: "Deep Learning: Foundations and Concepts",
    authors: "Christopher M. Bishop & Hugh Bishop",
    publisher: "Springer",
    year: 2024,
    isbn10: "3031454677",
    isbn13: "9783031454677",
    source: "https://www.waterstones.com/book/9783031454677",
    site: { href: "#/concept/dataflow", label: "flow.tab" },
    free: "https://www.bishopbook.com/",
  },
  // History
  {
    id: "chipwar",
    group: "history",
    top: true,
    title: "Chip War: The Fight for the World's Most Critical Technology",
    authors: "Chris Miller",
    publisher: "Simon & Schuster UK",
    year: 2023,
    isbn10: "1398504122",
    isbn13: "9781398504127",
    source: "https://www.whsmith.co.uk/products/chip-war/9781398504127/",
    site: { href: "#/concept/history", label: "history.tab" },
  },
  {
    id: "witt",
    group: "history",
    top: false,
    title: "The Thinking Machine: Jensen Huang and Nvidia, the Company Shaping the Future of AI",
    authors: "Stephen Witt",
    publisher: "The Bodley Head",
    year: 2025,
    isbn10: "1847928285",
    isbn13: "9781847928283",
    source: "https://www.penguin.co.nz/books/the-thinking-machine-9781847928283",
    site: { href: "#/concept/history/cuda", label: "history.tab" },
  },
  {
    id: "nvidiaway",
    group: "history",
    top: false,
    title: "The Nvidia Way: Jensen Huang and the Making of a Tech Giant",
    authors: "Tae Kim",
    publisher: "W. W. Norton",
    year: 2024,
    isbn10: "1324086718",
    isbn13: "9781324086710",
    source: "https://www.barnesandnoble.com/w/the-nvidia-way-tae-kim/1146239571",
    site: { href: "#/skills/nvlink", label: "skills.tab.nvlink" },
  },
  {
    id: "worlds",
    group: "history",
    top: false,
    title: "The Worlds I See: Curiosity, Exploration, and Discovery at the Dawn of AI",
    authors: "Fei-Fei Li",
    publisher: "Flatiron Books",
    year: 2025,
    isbn10: "1250898102",
    isbn13: "9781250898104",
    source: "https://www.schweitzer-online.de/en/buch/Li/Worlds-I-See/9781250898104/A69488962/",
    site: { href: "#/concept/history/imagenet", label: "history.tab" },
  },
];
