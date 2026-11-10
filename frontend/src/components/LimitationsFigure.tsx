import { useEffect, useState } from "react";
import { fallingLift, loopRadii, magneticLoop } from "../limitationsModel";
import { usePageText, type PageKey } from "../content/pages";
import Tex from "./Tex";
import { Arrow } from "./MathsAnimPics";

export type LimitationsFigureKind = "scope" | "spinors" | "gravity" | "connections" | "topology" | "bridge";

const BUTTON = "rounded border border-gray-200 bg-white px-3 py-1 text-xs text-blue-800 hover:border-blue-300";
const PRESETS: Partial<Record<LimitationsFigureKind, { max: number; marks: number[] }>> = {
  scope: { max: 360, marks: [0, 90, 180, 360] },
  spinors: { max: 720, marks: [0, 360, 720] },
  connections: { max: 720, marks: [0, 90, 360, 720] },
  bridge: { max: 180, marks: [0, 90, 180] },
};

/** Interactive illustration inside a card's "Tiny example" box; styled like the Figure of the other concept tabs. */
export default function LimitationsFigure({ kind }: { kind: LimitationsFigureKind }) {
  const T = usePageText();
  const F = (key: string) => T(`limitations.fig.${key}` as PageKey);
  const [angle, setAngle] = useState(kind === "bridge" ? 90 : 0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (kind !== "connections" || !playing) return;
    let frame = 0, last = 0;
    const tick = (now: number) => {
      if (last) setAngle(a => Math.min(720, a + Math.min(now - last, 60) * .035));
      last = now; frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [kind, playing]);
  useEffect(() => { if (kind === "connections" && angle >= 720) setPlaying(false); }, [kind, angle]);
  const flux = magneticLoop(angle);
  const theta = angle * Math.PI / 180;
  const radii = loopRadii(angle);
  const seconds = angle / 50; // gravity: the slider runs 0..100, i.e. 0..2 s
  const lift = fallingLift(seconds);
  const caption = F(`${kind}.caption`);
  const label = (x: number, y: number, text: string) => <text x={x} y={y} textAnchor="middle" fontSize="12" fill="#475569">{text}</text>;
  const preset = PRESETS[kind];

  return <figure className="rounded-lg border border-gray-200 bg-white p-2 sm:p-3" data-testid={`limitations-figure-${kind}`}>
    <svg viewBox="0 0 500 220" className="mx-auto w-full max-w-xl" role="img" aria-label={caption}>
      {kind === "scope" && <>
        <g transform={`rotate(${-angle} 150 112)`} stroke="#94a3b8" strokeWidth="2"><path d="M70 112H235M150 195V28"/><path d="M227 107L235 112L227 117M145 36L150 28L155 36" fill="none"/></g>
        <g fill="#2563eb"><path d="M150 112H205" fill="none" stroke="#2563eb" strokeWidth="4" strokeLinecap="round"/><polygon points="215,112 202,105 202,119"/></g>
        {label(150, 212, F("scope.fixed"))}
        <rect x="295" y="57" width="180" height="108" rx="12" fill="#f9fafb" stroke="#e5e7eb"/>
        {label(385, 89, F("scope.view"))}
        {label(385, 125, `(${Math.cos(theta).toFixed(2)}, ${(-Math.sin(theta)).toFixed(2)})`)}
      </>}
      {kind === "spinors" && <>
        {[125, 375].map(x => <circle key={x} cx={x} cy="90" r="55" fill="#f9fafb" stroke="#e5e7eb"/>)}
        <g transform={`rotate(${-angle} 125 90)`}><g fill="#2563eb"><path d="M125 90H161" fill="none" stroke="#2563eb" strokeWidth="3" strokeLinecap="round"/><polygon points="170,90 158,84 158,96"/></g></g>
        <g transform={`rotate(${-angle / 2} 375 90)`}><g fill="#7c3aed"><path d="M375 90H411" fill="none" stroke="#7c3aed" strokeWidth="3" strokeLinecap="round"/><polygon points="420,90 408,84 408,96"/></g></g>
        {label(125, 177, F("spinors.direction"))}
        {label(375, 177, F("spinors.phase"))}
      </>}
      {kind === "gravity" && <>
        {/* drawing scale: 19.6 m of drop -> 80 px; the squeeze is exaggerated (3 µm -> 22 px) */}
        {[{ x: 125, dy: lift.dropMetres / 19.62 * 80, ground: true }, { x: 375, dy: 0, ground: false }].map(p => {
          const s = lift.squeezeMetres / 3.083e-6 * 22;
          return <g key={p.x}>
            {p.ground && <path d="M30 200H220" stroke="#9ca3af" strokeWidth="2"/>}
            <g transform={`translate(0 ${p.dy})`}>
              <rect x={p.x - 70} y="22" width="140" height="90" rx="6" fill="#f9fafb" stroke="#9ca3af"/>
              {[-45 + s, 0, 45 - s].map((dx, i) => <circle key={i} cx={p.x + dx} cy="67" r="8" fill={i === 1 ? "#dc2626" : "#f87171"}/>)}
              {p.ground && <Arrow x1={p.x} y1={80} x2={p.x} y2={104} color="#2563eb" w={2} head={7}/>}
              {!p.ground && s > 1 && <>
                <Arrow x1={p.x - 58} y1={90} x2={p.x - 45 + s} y2={90} color="#7c3aed" w={2} head={6}/>
                <Arrow x1={p.x + 58} y1={90} x2={p.x + 45 - s} y2={90} color="#7c3aed" w={2} head={6}/>
              </>}
            </g>
            {label(p.x, 214, F(p.ground ? "gravity.ground" : "gravity.lift"))}
          </g>;
        })}
      </>}
      {kind === "connections" && <>
        <circle cx="165" cy="105" r="72" fill="none" stroke="#93c5fd" strokeWidth="2" strokeDasharray="5 4"/>
        <circle cx="165" cy="105" r="29" fill="#dbeafe"/>
        {label(165, 109, F("connections.flux"))}
        <circle cx={165 + 72 * Math.cos(theta)} cy={105 - 72 * Math.sin(theta)} r="7" fill="#0f766e"/>
        <circle cx="380" cy="105" r="43" fill="#f9fafb" stroke="#e5e7eb"/>
        <Arrow x1={380} y1={105} x2={380 + 35 * Math.cos(flux.phaseRadians)} y2={105 - 35 * Math.sin(flux.phaseRadians)} color="#2563eb" w={3} head={9}/>
        {label(380, 177, F("connections.phase"))}
        {label(165, 205, F("connections.around"))}
      </>}
      {kind === "topology" && <>
        {[125, 375].map((x, i) => <g key={x}><rect x={x - 95} y="22" width="190" height="154" rx="18" fill="#dbeafe" stroke="#93c5fd"/><circle cx={x} cy="99" r={Math.max(2, radii[i])} fill={i === 0 && radii[0] === 0 ? "#2563eb" : "none"} stroke="#2563eb" strokeWidth="3" strokeDasharray={i === 0 && radii[0] === 0 ? undefined : "5 3"}/></g>)}
        <circle cx="375" cy="99" r="28" fill="white" stroke="#93c5fd"/>
        {label(125, 207, F("topology.shrinks"))}
        {label(375, 207, F("topology.stuck"))}
      </>}
      {kind === "bridge" && <>
        <path d={`M120 140h85l${85 * Math.cos(theta)} ${-85 * Math.sin(theta)}h-85Z`} fill="#bfdbfe"/>
        <Arrow x1={120} y1={140} x2={205} y2={140} color="#0f766e" w={4} head={12}/>
        <Arrow x1={120} y1={140} x2={120 + 85 * Math.cos(theta)} y2={140 - 85 * Math.sin(theta)} color="#2563eb" w={4} head={12}/>
        {label(205, 162, "a")}{label(110 + 95 * Math.cos(theta), 135 - 95 * Math.sin(theta), "b")}
        {label(380, 80, `${F("bridge.alignment")}: ${Math.cos(theta).toFixed(2)}`)}
        {label(380, 118, `${F("bridge.area")}: ${Math.sin(theta).toFixed(2)}`)}
      </>}
    </svg>

    {kind === "connections" && <div className="mb-3 grid gap-3 text-center text-xs sm:grid-cols-2">
      <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2"><p className="mb-1 text-gray-500">{F("connections.fixedFlux")}</p><Tex tex={String.raw`\Phi\simeq${(flux.fluxWebers / 1e-15).toFixed(2)}\times10^{-15}\,\mathrm{Wb}`}/></div>
      <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2"><p className="mb-1 text-gray-500">{F("connections.electronPhase")}</p><Tex tex={String.raw`\Delta\varphi=${flux.phaseDegrees.toFixed(1)}^\circ`}/></div>
    </div>}
    {kind === "gravity" && <div className="mb-3 grid gap-3 text-center text-xs sm:grid-cols-2" data-testid="gravity-readout">
      <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2"><p className="mb-1 text-gray-500">{F("gravity.pull")}</p>
        <p className="flex flex-wrap justify-center gap-x-4"><span className="whitespace-nowrap"><span className="text-gray-500">{F("gravity.ground")}</span> <Tex tex={String.raw`9.81\,\mathrm{m/s^2}`}/></span><span className="whitespace-nowrap"><span className="text-gray-500">{F("gravity.lift")}</span> <Tex tex="0"/></span></p></div>
      <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2"><p className="mb-1 text-gray-500">{F("gravity.tide")}</p>
        <Tex tex={String.raw`${(lift.squeezeMetres * 1e6).toFixed(2)}\,\mu\mathrm m`}/> <span className="text-gray-500">({F("gravity.same")})</span></div>
    </div>}
    {kind === "gravity" && <label className="flex items-center gap-3 text-xs text-gray-700">{F("gravity.time")}: {seconds.toFixed(1)} s
      <input type="range" min="0" max="100" value={angle} onChange={e => setAngle(Number(e.target.value))} aria-label={F("gravity.time")} className="min-w-0 flex-1 accent-blue-600"/>
    </label>}
    {kind === "topology" && <div className="mt-2 space-y-3">
      <label className="flex items-center gap-3 text-xs text-gray-700">{F("topology.shrink")}<input type="range" min="0" max="100" value={angle} onChange={e => setAngle(Number(e.target.value))} aria-label={F("topology.shrink")} className="min-w-0 flex-1 accent-blue-600"/></label>
      <div className="flex gap-2">{[0, 100].map(value => <button key={value} type="button" onClick={() => setAngle(value)} className={BUTTON}>{F(value === 0 ? "topology.reset" : "topology.full")}</button>)}</div>
    </div>}
    {preset && <>
      <label className="mt-2 flex items-center gap-3 text-xs text-gray-700">{F("angle")}: {Math.round(angle)}°
        <input className="min-w-0 flex-1 accent-blue-600" aria-label={F("angleAria")} type="range" min="0" max={preset.max} value={angle} onChange={e => { setPlaying(false); setAngle(Number(e.target.value)); }}/>
      </label>
      <div className="mt-3 flex flex-wrap gap-2">
        {kind === "connections" && <button type="button" className={BUTTON} onClick={() => { if (angle >= 720) setAngle(0); setPlaying(p => !p); }}>{F(playing ? "connections.pause" : "connections.play")}</button>}
        {preset.marks.map(a => <button type="button" key={a} onClick={() => { setPlaying(false); setAngle(a); }} className={BUTTON}>{a}°</button>)}
      </div>
    </>}
    <figcaption className="mt-2 text-center text-xs text-gray-600">{caption}</figcaption>
  </figure>;
}
