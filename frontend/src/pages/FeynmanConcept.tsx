import { useState } from "react";
import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import Tex, { RichText } from "../components/Tex";
import { usePageText, type PageKey } from "../content/pages";
import { BridgeFigure, DiagramFigure, ORingFigure, SkyFigure, StepIcon, ZipFigure } from "../components/FeynmanFigures";
import { FEYNMAN, Feynman, Links } from "./MathsConcept";

// Sections in order; the address #/concept/feynman/<key> scrolls to id feynman-<key> (done by ConceptPage).
export const FEYNMAN_SECTIONS = ["steps", "roots", "life", "site", "example", "pitfalls"] as const;
type Section = (typeof FEYNMAN_SECTIONS)[number];

export const STEP_IDS = [1, 2, 3, 4] as const;
export const ROOTS = ["notebook", "create", "freshman", "lectures"] as const;
export const LIFE = ["losalamos", "diagrams", "nobel", "challenger"] as const;
export const SITE = ["ladder", "maths", "tdd"] as const; // in step order: 1, then 2 & 4, then 3
export const PITFALLS = [1, 2, 3] as const;
export const CHECKS = [1, 2, 3, 4, 5, 6] as const;

type Root = (typeof ROOTS)[number];
type Life = (typeof LIFE)[number];

// How sure we are of each quote's wording, and the English original (not translated: it is the source text).
const ROOT_META: Record<Root, { sure: "verbatim" | "reported"; orig?: string }> = {
  notebook: { sure: "reported", orig: "NOTEBOOK OF THINGS I DON'T KNOW ABOUT" },
  create: { sure: "verbatim", orig: "What I cannot create, I do not understand." },
  freshman: { sure: "reported", orig: "I couldn't reduce it to the freshman level. That means we really don't understand it." },
  lectures: { sure: "verbatim" },
};

const LIFE_YEAR: Record<Life, string> = { losalamos: "1943–1945", diagrams: "1949", nobel: "1965", challenger: "1986" };

// Internal links of section 4 (opened in a new tab like every link on the page).
const SITE_LINK: Record<(typeof SITE)[number], { href: string; label: PageKey }> = {
  maths: { href: "#/concept/maths", label: "maths.tab" },
  ladder: { href: "#/concept/laws/2", label: "flow.tab.law" },
  tdd: { href: "#/skills/techniques/tdd", label: "feynman.site.tdd.link" },
};

// External sources, opened in a new tab (titles not translated). Each checked: HTTP 200 or a valid DOI.
const LINKS: Partial<Record<Section, { label: string; url?: string }[]>> = {
  steps: [
    { label: "S. H. Young (September 2011), Learn faster with the Feynman technique (blog post)", url: "https://www.scotthyoung.com/blog/2011/09/01/learn-faster/" },
    { label: "C. Newport (2015), The Feynman notebook method (blog post)", url: "https://calnewport.com/the-feynman-notebook-method/" },
  ],
  roots: [
    { label: "J. Gleick (1992), Genius: The Life and Science of Richard Feynman, Pantheon (the notebook), quoted in C. Newport (2015)", url: "https://calnewport.com/the-feynman-notebook-method/" },
    { label: "Caltech Archives: Richard Feynman's blackboard at time of his death (photograph, 1988)", url: "https://digital.archives.caltech.edu/collections/Photographs/1.10-3" },
    { label: "D. L. Goodstein (1989), Richard P. Feynman, Teacher, Physics Today 42(2), 70–75", url: "https://doi.org/10.1063/1.881195" },
    { label: "Goodstein (1989), free copy at CaltechAUTHORS", url: "https://authors.library.caltech.edu/62878/1/1.881195.pdf" },
    { label: "The Feynman Lectures on Physics, online edition (Caltech)", url: "https://www.feynmanlectures.caltech.edu/" },
  ],
  life: [
    { label: "National Park Service: Manhattan Project scientists, Richard P. Feynman", url: "https://www.nps.gov/people/manhattan-project-scientists-richard-p-feynman.htm" },
    { label: "R. P. Feynman (1975 talk), Los Alamos from Below, Engineering & Science (Caltech)", url: "https://calteches.library.caltech.edu/34/3/FeynmanLosAlamos.htm" },
    { label: "R. P. Feynman (1949), Space-time approach to quantum electrodynamics, Phys. Rev. 76, 769", url: "https://doi.org/10.1103/PhysRev.76.769" },
    { label: "NobelPrize.org: The Nobel Prize in Physics 1965", url: "https://www.nobelprize.org/prizes/physics/1965/summary/" },
    { label: "R. P. Feynman (1986), Personal observations on the reliability of the Shuttle, Rogers Commission report, vol. 2, appendix F (NASA)", url: "https://www.nasa.gov/history/rogersrep/v2appf.htm" },
    { label: "APS News (2001), This month in physics history: the Challenger explosion", url: "https://www.aps.org/apsnews/2001/01/1986-the-challenger-explosion" },
    { label: "Wikipedia: Rogers Commission Report", url: "https://en.wikipedia.org/wiki/Rogers_Commission_Report" },
  ],
  example: [
    { label: "PyTorch: torch.einsum", url: "https://docs.pytorch.org/docs/stable/generated/torch.einsum.html" },
    { label: "Wikipedia: Einstein notation", url: "https://en.wikipedia.org/wiki/Einstein_notation" },
  ],
  pitfalls: [
    { label: "R. P. Feynman (1974), Cargo Cult Science (Caltech commencement address), Engineering & Science 37(7)", url: "https://calteches.library.caltech.edu/51/2/CargoCult.htm" },
  ],
};

// Further Commons pictures (public domain, licences checked on the file pages; credits in public/feynman/CREDITS.md)
const PHOTOS = {
  badge: {
    file: "badge.webp",
    w: 400,
    h: 463,
    author: "United States Army (Los Alamos, c. 1943)",
    licence: "Public domain",
    licenceUrl: "https://commons.wikimedia.org/wiki/Template:PD-USGov",
    source: "https://commons.wikimedia.org/wiki/File:Richard_Feynman_Los_Alamos_ID_badge.jpg",
  },
  ice: {
    file: "ice.webp",
    w: 381,
    h: 480,
    author: "NASA (Kennedy Space Center, 28 January 1986)",
    licence: "Public domain",
    licenceUrl: "https://commons.wikimedia.org/wiki/Template:PD-USGov-NASA",
    source: "https://commons.wikimedia.org/wiki/File:Ice_on_the_Pad_on_the_Day_of_STS-51-L%27s_Launch_-_GPN-2004-00011.jpg",
  },
} as const;

const PHOTO = {
  file: "feynman.webp",
  author: "Unknown photographer (The Big T, Caltech yearbook, 1959)",
  licence: "Public domain",
  licenceUrl: "https://commons.wikimedia.org/wiki/Template:PD-US-no_notice",
  source: "https://commons.wikimedia.org/wiki/File:Richard_Feynman_1959.png",
};

const EINSUM_CODE = `Q = torch.tensor([[2., 1.],    # customer 0: 2 of product 0, 1 of product 1
                  [0., 3.]])   # customer 1: 3 of product 1
p = torch.tensor([3., 5.])     # price of each product
torch.einsum("ij,j->i", Q, p)  # tensor([11., 15.]): j is summed and gone, i stays`;

const ext = { target: "_blank", rel: "noreferrer" } as const;

/** The Concept page's "The Feynman technique" tab: the four steps, their real roots, Feynman as an explainer,
 *  how this site uses the steps, a worked example on einsum, pitfalls and a self-check. */
export default function FeynmanConcept() {
  const T = usePageText();
  const R = (key: PageKey) => <RichText text={T(key)} />;
  const k = (s: string) => `feynman.${s}` as PageKey;

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pb-10">
      <PageToc
        scrollRoot="concept-scroll"
        items={FEYNMAN_SECTIONS.map((s, i) => ({ id: `feynman-${s}`, mark: i + 1, label: T(k(`${s}.title`)) }))}
      />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10" data-testid="feynman-concept">
        <header>
          <p className="text-sm font-medium text-blue-700">{T("feynman.kicker")}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("feynman.title")}</h1>
          <p className="mt-3 text-gray-600">{R("feynman.intro")}</p>
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900">
            {R("feynman.note")}
          </p>
        </header>

        {FEYNMAN_SECTIONS.map((s, i) => (
          <section key={s} id={`feynman-${s}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-4 sm:p-6">
            <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
                {i + 1}
              </span>
              {T(k(`${s}.title`))}
            </h2>
            <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">
              {s === "steps" && (
                <>
                  {/* "precise" is left out: the four numbered step cards already say it */}
                  {FEYNMAN.filter((f) => f !== "precise").map((f) => (
                    <div key={f} className="space-y-4">
                      <Feynman kind={f} text={T(k(`steps.${f}`))} label={T(`maths.label.${f}` as PageKey)} />
                      {f === "plain" && <StepCycle />}
                      {f === "analogy" && <BridgeFigure />}
                      {f === "example" && <SkyFigure />}
                    </div>
                  ))}
                  <p className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-2.5">{R("feynman.steps.origin")}</p>
                </>
              )}
              {s === "roots" && <Roots />}
              {s === "life" && <LifeStory />}
              {s === "site" && <SiteMap />}
              {s === "example" && <WorkedExample />}
              {s === "pitfalls" && <Pitfalls />}
              {LINKS[s] && <Links items={LINKS[s]!} title={T("feynman.label.sources")} />}
            </div>
          </section>
        ))}
      </article>
    </div>
  );
}

/** The four steps as numbered cards with the "repeat" loop under them. */
function StepCycle() {
  const T = usePageText();
  return (
    <figure className="rounded-lg border border-gray-200 bg-gray-50/60 p-3" data-testid="feynman-cycle">
      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STEP_IDS.map((n) => (
          <li key={n} className="relative rounded-lg border border-blue-100 bg-white p-3">
            <StepIcon n={n} />
            <div className="mt-1 flex items-center gap-2">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-600 text-xs font-semibold text-white">
                {n}
              </span>
              <span className="font-semibold text-gray-900">{T(`feynman.step${n}.title` as PageKey)}</span>
            </div>
            <p className="mt-2 text-[13px] text-gray-700">
              <RichText text={T(`feynman.step${n}.text` as PageKey)} />
            </p>
          </li>
        ))}
      </ol>
      <figcaption className="mt-3 flex items-center justify-center gap-2 text-xs font-medium text-blue-800">
        <span aria-hidden className="text-base">↻</span>
        {T("feynman.cycle")}
      </figcaption>
    </figure>
  );
}

function Badge({ sure }: { sure: "verbatim" | "reported" }) {
  const T = usePageText();
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ${
        sure === "verbatim" ? "bg-emerald-50 text-emerald-800 ring-emerald-200" : "bg-amber-50 text-amber-900 ring-amber-200"
      }`}
    >
      {T(sure === "verbatim" ? "feynman.label.verbatim" : "feynman.label.reported")}
    </span>
  );
}

function Roots() {
  const T = usePageText();
  return (
    <>
      <p>
        <RichText text={T("feynman.roots.intro")} />
      </p>
      <div className="grid gap-3 md:grid-cols-2" data-testid="feynman-roots">
        {ROOTS.map((r) => (
          <div key={r} className="flex flex-col rounded-lg border border-gray-200 bg-white p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-semibold text-gray-900">{T(`feynman.roots.${r}.title` as PageKey)}</h3>
              <Badge sure={ROOT_META[r].sure} />
            </div>
            <p className="mt-2">
              <RichText text={T(`feynman.roots.${r}.text` as PageKey)} />
            </p>
            {ROOT_META[r].orig && (
              <blockquote className="mt-3 border-l-4 border-gray-300 pl-3 text-[13px] text-gray-800 italic" lang="en">
                <span className="mr-1 text-[11px] font-semibold tracking-wide text-gray-500 uppercase not-italic">
                  {T("feynman.label.original")}:
                </span>
                “{ROOT_META[r].orig}”
              </blockquote>
            )}
            <p className="mt-auto pt-3 text-xs text-gray-500">
              <RichText text={T(`feynman.roots.${r}.how` as PageKey)} />
            </p>
          </div>
        ))}
      </div>
    </>
  );
}

function LifeStory() {
  const T = usePageText();
  return (
    <>
      <p>
        <RichText text={T("feynman.life.intro")} />
      </p>
      <div className="grid items-start gap-6 md:grid-cols-[minmax(0,15rem)_1fr]">
        <figure className="mx-auto w-full max-w-[15rem]" data-testid="feynman-photo">
          <img
            src={`${import.meta.env.BASE_URL}feynman/${PHOTO.file}`}
            alt={T("feynman.photo.alt")}
            width={479}
            height={480}
            className="w-full rounded-lg border border-gray-200 bg-gray-100"
          />
          <figcaption className="mt-2 text-xs text-gray-600">
            <span className="block text-gray-800">{T("feynman.photo.caption")}</span>
            <span className="mt-1 block text-[11px] text-gray-500">
              {T("feynman.picture")}: {PHOTO.author},{" "}
              <a href={PHOTO.licenceUrl} {...ext} className="underline hover:text-gray-800">
                {PHOTO.licence}
              </a>
              ,{" "}
              <a href={PHOTO.source} {...ext} className="underline hover:text-gray-800">
                Wikimedia Commons
              </a>
            </span>
          </figcaption>
        </figure>
        <ol className="relative space-y-5 border-l-2 border-blue-100 pl-5" data-testid="feynman-life">
          {LIFE.map((e) => (
            <li key={e} className="relative">
              <span aria-hidden className="absolute top-1.5 -left-[27px] h-3 w-3 rounded-full border-2 border-white bg-blue-600" />
              <p className="text-xs font-semibold text-blue-700">{LIFE_YEAR[e]}</p>
              <h3 className="font-semibold text-gray-900">{T(`feynman.life.${e}.title` as PageKey)}</h3>
              <p className="mt-1">
                <RichText text={T(`feynman.life.${e}.text` as PageKey)} />
              </p>
              {e === "losalamos" && <SmallPhoto photo={PHOTOS.badge} caption={T("feynman.fig.badge.caption")} alt={T("feynman.fig.badge.alt")} />}
              {e === "diagrams" && (
                <div className="mt-3 space-y-3">
                  <DiagramFigure />
                  {/* our own problem built on one Feynman diagram */}
                  <a
                    href="#/nuclear/feynman_ee_mumu"
                    className="inline-flex items-center gap-1.5 rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-sm font-medium text-blue-800 hover:border-blue-400 hover:bg-blue-100"
                    data-testid="feynman-problem-link"
                  >
                    <span aria-hidden>→</span>
                    <RichText text={T("feynman.life.diagrams.link")} />
                  </a>
                </div>
              )}
              {e === "challenger" && (
                <div className="mt-3 grid items-start gap-3 sm:grid-cols-[minmax(0,11rem)_1fr]">
                  <SmallPhoto photo={PHOTOS.ice} caption={T("feynman.fig.ice.caption")} alt={T("feynman.fig.ice.alt")} />
                  <IceWaterTabs />
                </div>
              )}
            </li>
          ))}
        </ol>
      </div>
    </>
  );
}

function IceWaterTabs() {
  const T = usePageText();
  const [tab, setTab] = useState<"diagram" | "video">("diagram");
  return (
    <div className="min-w-0" data-testid="feynman-ice-tabs">
      <div role="tablist" aria-label={T("feynman.ice.tabs")} className="mb-3 flex border-b border-gray-200">
        {(["diagram", "video"] as const).map((id) => (
          <button key={id} id={`ice-tab-${id}`} type="button" role="tab" aria-selected={tab === id}
            aria-controls={`ice-panel-${id}`} tabIndex={tab === id ? 0 : -1}
            onClick={() => setTab(id)} onKeyDown={(e) => {
              if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
                e.preventDefault();
                const next = e.key === "Home" ? "diagram" : e.key === "End" ? "video" : tab === "diagram" ? "video" : "diagram";
                setTab(next); document.getElementById(`ice-tab-${next}`)?.focus();
              }
            }}
            className={`border-b-2 px-4 py-2 text-sm font-medium ${tab === id ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500"}`}>
            {T(id === "diagram" ? "feynman.ice.diagram" : "feynman.ice.video")}
          </button>
        ))}
      </div>
      <div id={`ice-panel-${tab}`} role="tabpanel" aria-labelledby={`ice-tab-${tab}`} tabIndex={0}>
        {tab === "diagram" ? <ORingFigure /> : <>
          <iframe className="aspect-video w-full rounded-lg border-0" src="https://www.youtube-nocookie.com/embed/raMmRKGkGD4"
            title={T("feynman.ice.videoTitle")} loading="lazy" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; web-share" allowFullScreen
            referrerPolicy="strict-origin-when-cross-origin" />
        </>}
      </div>
    </div>
  );
}

function SiteMap() {
  const T = usePageText();
  return (
    <>
      <p>
        <RichText text={T("feynman.site.intro")} />
      </p>
      <div className="grid gap-3 md:grid-cols-3" data-testid="feynman-site">
        {SITE.map((c) => (
          <div key={c} className="flex flex-col rounded-lg border border-gray-200 bg-white p-4">
            <h3 className="font-semibold text-gray-900">{T(`feynman.site.${c}.title` as PageKey)}</h3>
            <p className="mt-2">
              <RichText text={T(`feynman.site.${c}.text` as PageKey)} />
            </p>
            <a
              href={SITE_LINK[c].href}
              {...ext}
              className="mt-auto self-start pt-3 text-sm font-medium text-blue-700 hover:underline"
            >
              {T(SITE_LINK[c].label)} →
            </a>
          </div>
        ))}
      </div>
    </>
  );
}

// Step styles of the worked example: notes, a first attempt, the gap, the repaired explanation.
const EX_STYLE: Record<(typeof STEP_IDS)[number], string> = {
  1: "border-gray-200 bg-gray-50 font-mono text-[12.5px] text-gray-800",
  2: "border-gray-200 bg-white",
  3: "border-rose-200 bg-rose-50 text-rose-950",
  4: "border-emerald-200 bg-emerald-50/70 text-emerald-950",
};

function WorkedExample() {
  const T = usePageText();
  return (
    <>
      <p>
        <RichText text={T("feynman.example.intro")} />
      </p>
      <div className="overflow-hidden rounded-lg border border-gray-200 bg-white">
        <div className="bg-gray-50 px-2 py-1 text-lg">
          <Tex tex={String.raw`C_{ik} = \sum_j A_{ij} B_{jk}`} display />
        </div>
        <CodeBlock code={'C = torch.einsum("ij,jk->ik", A, B)   # the same as A @ B'} />
      </div>
      <ol className="relative space-y-4" data-testid="feynman-example">
        {STEP_IDS.map((n) => (
          <li key={n} className="flex gap-3">
            <div className="flex flex-col items-center">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white">
                {n}
              </span>
              {n < 4 && <span aria-hidden className="mt-1 w-0.5 flex-1 bg-blue-100" />}
            </div>
            <div className="min-w-0 flex-1 pb-1">
              <p className="mb-1 font-semibold text-gray-900">{T(`feynman.example.s${n}.title` as PageKey)}</p>
              <div className={`rounded-lg border px-4 py-3 ${EX_STYLE[n]}`}>
                <RichText text={T(`feynman.example.s${n}.text` as PageKey)} />
              </div>
              {n === 4 && (
                <div className="mt-3 space-y-3">
                  <PForms />
                  <p>
                    <RichText text={T("feynman.example.check")} />
                  </p>
                  <CodeBlock code={EINSUM_CODE} />
                </div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </>
  );
}

// The same product two ways: as matrices (p is a column) and with indices (p_j has no row or column).
const PFORM_TEX = {
  matrix: String.raw`Q\,p = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix} \begin{pmatrix} 3 \\ 5 \end{pmatrix} = \begin{pmatrix} 11 \\ 15 \end{pmatrix}`,
  index: String.raw`\begin{aligned} (Qp)_i &= \textstyle\sum_j Q_{ij}\, p_j \hspace{2em} p_1 = 3 \hspace{1em} p_2 = 5 \\ (Qp)_1 &= Q_{11}p_1 + Q_{12}p_2 = 11 \\ (Qp)_2 &= Q_{21}p_1 + Q_{22}p_2 = 15 \end{aligned}`,
} as const;

function PForms() {
  const T = usePageText();
  const [form, setForm] = useState<keyof typeof PFORM_TEX>("matrix");
  const other = form === "matrix" ? "index" : "matrix";
  // top right of the figure: a switch between the two forms, and a (?) that explains them; the drawing and the formula follow it
  const corner = (
    <div className="flex items-center gap-2 text-xs" data-testid="feynman-pform">
      <span className={form === "matrix" ? "font-semibold text-blue-700" : "text-gray-500"}>{T("feynman.pform.matrix")}</span>
      <button
        type="button"
        role="switch"
        aria-checked={form === "index"}
        aria-label={`${T("feynman.pform.title")}: ${T(`feynman.pform.${other}`)}`}
        onClick={() => setForm(other)}
        className={`relative h-5 w-9 shrink-0 rounded-full transition ${form === "index" ? "bg-blue-600" : "bg-gray-300"}`}
      >
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${form === "index" ? "left-[18px]" : "left-0.5"}`} />
      </button>
      <span className={form === "index" ? "font-semibold text-blue-700" : "text-gray-500"}>{T("feynman.pform.index")}</span>
      <HelpCard heading={T("feynman.pform.title")} text={T("feynman.pform.help")} />
    </div>
  );
  return (
    <ZipFigure
      form={form}
      corner={corner}
      below={
        <div className="mt-1 overflow-x-auto px-2 py-1 text-sm">
          <Tex tex={PFORM_TEX[form]} display />
        </div>
      }
    />
  );
}

/** A (?) that opens a white card with rich text (maths rendered) on hover or keyboard focus. */
function HelpCard({ heading, text }: { heading: string; text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-flex" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        className="inline-flex h-5 w-5 cursor-help items-center justify-center rounded-full border border-gray-300 bg-white text-[11px] font-semibold text-gray-500"
        aria-expanded={open}
        aria-label={heading}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
      >
        ?
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute top-full right-0 z-30 mt-1.5 block w-72 max-w-[80vw] rounded-lg border border-gray-200 bg-white p-3 text-left text-xs leading-relaxed text-gray-700 shadow-lg"
        >
          <span className="mb-1 block font-semibold text-gray-900">{heading}</span>
          <RichText text={text} />
        </span>
      )}
    </span>
  );
}

function Pitfalls() {
  const T = usePageText();
  const [ticked, setTicked] = useState<Set<number>>(() => new Set());
  const toggle = (n: number) =>
    setTicked((s) => {
      const t = new Set(s);
      if (t.has(n)) t.delete(n);
      else t.add(n);
      return t;
    });
  return (
    <>
      <p>
        <RichText text={T("feynman.pitfalls.intro")} />
      </p>
      <ul className="list-disc space-y-2 pl-5">
        {PITFALLS.map((n) => (
          <li key={n}>
            <RichText text={T(`feynman.pitfalls.p${n}` as PageKey)} />
          </li>
        ))}
      </ul>
      <div className="rounded-lg border border-blue-100 bg-blue-50/60 p-4 text-blue-950" data-testid="feynman-check">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="font-semibold">{T("feynman.check.title")}</h3>
          <span className="text-xs font-medium text-blue-800 tabular-nums">
            {ticked.size} / {CHECKS.length}
          </span>
        </div>
        <ul className="mt-3 space-y-2">
          {CHECKS.map((n) => (
            <li key={n}>
              <label className="flex cursor-pointer items-start gap-2">
                <input
                  type="checkbox"
                  checked={ticked.has(n)}
                  onChange={() => toggle(n)}
                  className="mt-0.5 h-4 w-4 shrink-0 accent-blue-600"
                />
                <span>
                  <RichText text={T(`feynman.check.c${n}` as PageKey)} />
                </span>
              </label>
            </li>
          ))}
        </ul>
        {ticked.size === CHECKS.length && (
          <p className="mt-3 font-medium text-emerald-800">
            <RichText text={T("feynman.check.done")} />
          </p>
        )}
      </div>
    </>
  );
}

/** A small credited photo inside the timeline. */
function SmallPhoto({ photo, caption, alt }: { photo: (typeof PHOTOS)[keyof typeof PHOTOS]; caption: string; alt: string }) {
  const T = usePageText();
  return (
    <figure className="mt-3 max-w-[11rem]" data-testid={`feynman-photo-${photo.file.replace(".webp", "")}`}>
      <img
        src={`${import.meta.env.BASE_URL}feynman/${photo.file}`}
        alt={alt}
        width={photo.w}
        height={photo.h}
        className="w-full rounded-lg border border-gray-200 bg-gray-100"
      />
      <figcaption className="mt-1 text-xs text-gray-600">
        <span className="block text-gray-800">
          <RichText text={caption} />
        </span>
        <span className="mt-0.5 block text-[11px] text-gray-500">
          {T("feynman.picture")}: {photo.author},{" "}
          <a href={photo.licenceUrl} {...ext} className="underline hover:text-gray-800">
            {photo.licence}
          </a>
          ,{" "}
          <a href={photo.source} {...ext} className="underline hover:text-gray-800">
            Wikimedia Commons
          </a>
        </span>
      </figcaption>
    </figure>
  );
}
