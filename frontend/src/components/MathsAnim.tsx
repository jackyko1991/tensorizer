/**
 * One animation of the Concept page's "Tensor maths" tab: an SVG picture, the list of its steps (the current
 * one highlighted), a plain caption and the shared play / step / speed controls. Step logic in mathsAnim.ts.
 */
import { type ReactNode } from "react";
import { usePageText, type PageKey } from "../content/pages";
import { MATHS_PLAN, UNITS, UPDOWN_UNIT, mFrames, mStage, mStepCount, mStepStart, type MathsCase } from "../mathsAnim";
import { AnimControls, btn, useAnimClock } from "./animKit";
import { Address, EinEx, EqBatch, EqDot, EqMatmul, EqMatvec, EqOuter, EqRank, EqRotate, EqTrace, Metric, Photo, Receipt, Rotate, Updown, W, type PicProps } from "./MathsAnimPics";
import { Chain, EqChain, EqLoop, EqMps, EqRoof, Fusion, Ranks, Van, Wire } from "./MathsAnimPics2";
import { RichText } from "./Tex";

const HOLD = 6; // ticks the finished picture stays before the cycle restarts

const PICTURES: Record<MathsCase, { draw: (p: PicProps) => ReactNode; h: number }> = {
  address: { draw: (p) => <Address {...p} />, h: 118 },
  rotate: { draw: (p) => <Rotate {...p} />, h: 168 },
  updown: { draw: (p) => <Updown {...p} />, h: 188 },
  metric: { draw: (p) => <Metric {...p} />, h: 140 },
  photo: { draw: (p) => <Photo {...p} />, h: 166 },
  eqRank: { draw: (p) => <EqRank {...p} />, h: 126 },
  eqRotate: { draw: (p) => <EqRotate {...p} />, h: 146 },
  receipt: { draw: (p) => <Receipt {...p} />, h: 126 },
  einEx: { draw: (p) => <EinEx {...p} />, h: 132 },
  eqDot: { draw: (p) => <EqDot {...p} />, h: 122 },
  eqMatvec: { draw: (p) => <EqMatvec {...p} />, h: 118 },
  eqMatmul: { draw: (p) => <EqMatmul {...p} />, h: 118 },
  eqTrace: { draw: (p) => <EqTrace {...p} />, h: 118 },
  eqOuter: { draw: (p) => <EqOuter {...p} />, h: 120 },
  eqBatch: { draw: (p) => <EqBatch {...p} />, h: 134 },
  wire: { draw: (p) => <Wire {...p} />, h: 118 },
  eqLoop: { draw: (p) => <EqLoop {...p} />, h: 130 },
  eqMps: { draw: (p) => <EqMps {...p} />, h: 134 },
  chain: { draw: (p) => <Chain {...p} />, h: 220 },
  eqChain: { draw: (p) => <EqChain {...p} />, h: 138 },
  van: { draw: (p) => <Van {...p} />, h: 150 },
  fusion: { draw: (p) => <Fusion {...p} />, h: 164 },
  eqRoof: { draw: (p) => <EqRoof {...p} />, h: 178 },
  ranks: { draw: (p) => <Ranks {...p} />, h: 126 },
};

/** compact: steps below the picture at every width (inside a half-width equation card). */
export default function MathsAnim({ id, compact = false }: { id: MathsCase; compact?: boolean }) {
  const T = usePageText();
  const clock = useAnimClock(mFrames(id) + HOLD, MATHS_PLAN[id].tick);
  const { root, g } = clock;
  const s = mStage(id, g);
  const n = mStepCount(id);
  const pic = PICTURES[id];
  const caption = T(`maths.anim.${id}.caption` as PageKey);
  const jump = (step: number) => {
    // show the step finished: its last tick
    clock.setPlaying(false);
    clock.setG(step + 1 < n ? mStepStart(id, step + 1) - 1 : mFrames(id) - 1);
  };
  return (
    <div ref={root} className={compact ? "" : "mt-1"} data-testid={`maths-anim-${id}`}>
      <figure className="overflow-hidden rounded-lg border border-gray-200 bg-white">
        <figcaption className="flex items-center justify-between border-b border-gray-200 bg-gray-50 px-3 py-1.5 text-xs text-gray-600">
          <span>{T("maths.anim.title")}</span>
          {n > 1 && (
            <span className="font-mono tabular-nums text-blue-700" data-testid={`maths-anim-step-${id}`}>
              {s.step + 1}/{n}
            </span>
          )}
        </figcaption>
        <div className={compact ? "" : "grid md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]"}>
          <svg
            viewBox={`0 0 ${W} ${pic.h}`}
            className="mx-auto block h-auto w-full max-w-[540px] p-1"
            role="img"
            aria-label={caption.replace(/[$`\\]/g, "")}
          >
            {pic.draw({ g, s, T })}
          </svg>
          <ol className={`space-y-1 border-t border-gray-100 p-3 text-xs leading-snug ${compact ? "" : "md:border-t-0 md:border-l"}`}>
            {Array.from({ length: n }, (_, k) => (
              <li
                key={k}
                className={`flex gap-2 rounded px-1.5 py-1 ${k === s.step ? "bg-blue-50 text-gray-900" : "text-gray-500"}`}
                aria-current={k === s.step ? "step" : undefined}
              >
                <button
                  type="button"
                  onClick={() => jump(k)}
                  className={`mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] ${
                    k === s.step ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-600 hover:bg-gray-300"
                  }`}
                  aria-label={`${k + 1}`}
                >
                  {k + 1}
                </button>
                <span>
                  <RichText text={T(`maths.anim.${id}.s${k}` as PageKey)} />
                </span>
              </li>
            ))}
          </ol>
        </div>
        <p className="border-t border-gray-100 px-3 py-2 text-xs leading-snug text-gray-600">
          <RichText text={caption} />
        </p>
      </figure>
      <div className="flex flex-wrap items-center gap-x-4">
        <AnimControls clock={clock} />
        {id === "updown" && (
          <div className="mt-2 flex items-center gap-1.5 text-xs text-gray-600" data-testid="maths-unit-toggle">
            {T("maths.anim.updown.unit")}:
            {[0, 2].map((step) => (
              <button
                key={step}
                type="button"
                className={UPDOWN_UNIT[s.step] === step ? btn.replace("border-gray-300 bg-white", "border-blue-500 bg-blue-50 font-semibold").replace("text-gray-700", "text-blue-800") : btn}
                aria-pressed={UPDOWN_UNIT[s.step] === step}
                onClick={() => jump(step)}
              >
                {UNITS[step].name}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
