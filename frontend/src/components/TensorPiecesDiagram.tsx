import { useState } from "react";
import Tex, { RichText } from "./Tex";
import { useLang } from "../i18n";
import { EXAMPLE_INPUT, tensorPieces, type Vec2 } from "../tensorDecomposition";
import { AnimControls, btn, useAnimClock } from "./animKit";

const colours = ["#2563eb", "#d97706", "#7c3aed", "#059669"];
const fmt = (x: number) => Math.abs(x) < 1e-9 ? "0" : Number(x.toFixed(2)).toString();
const pair = (v: Vec2) => `(${v.map(fmt).join(", ")})`;

export default function TensorPiecesDiagram() {
  const { lang } = useLang();
  const l = lang.startsWith("zh") ? 1 : 0;
  const [angle, setAngle] = useState(0);
  const clock = useAnimClock(69, 110);
  const stage = Math.min(3, Math.floor(clock.g / 16));
  const current = Math.min(3, Math.max(0, Math.floor((clock.g - 32) / 4)));
  const data = tensorPieces(angle);
  const steps = l ? ["選基底", "讀取輸入", "看四片規則", "加回完整作用"] : ["Choose basis", "Read input", "Inspect four pieces", "Add the action"];
  const origin: Vec2 = [72, 188];
  const project = (v: Vec2): Vec2 => [origin[0] + 18 * v[0], origin[1] - 18 * v[1]];
  const arrow = (a: Vec2, b: Vec2, colour: string, width = 2.5) => {
    const dx = b[0] - a[0], dy = b[1] - a[1], length = Math.hypot(dx, dy);
    if (length < .5) return <circle cx={a[0]} cy={a[1]} r="2.5" fill={colour} />;
    const ux = dx / length, uy = dy / length, h = Math.min(8, length * .4), w = h * .5;
    const base: Vec2 = [b[0] - h*ux, b[1] - h*uy];
    return <g fill={colour}><path d={`M${a}L${base}`} stroke={colour} strokeWidth={width} fill="none" strokeLinecap="round"/>
      <polygon points={`${b} ${base[0]-w*uy},${base[1]+w*ux} ${base[0]+w*uy},${base[1]-w*ux}`}/></g>;
  };
  let running: Vec2 = [0, 0];
  const changeFrame = (value: number) => { setAngle(value); clock.setPlaying(false); clock.setG(63); };
  return <div ref={clock.root} data-testid="tensor-pieces-animation">
    <figure className="overflow-hidden rounded-lg border border-gray-200 bg-white">
      <figcaption className="flex flex-wrap gap-2 border-b border-gray-200 bg-gray-50 p-3 text-xs">
        {steps.map((s,i)=><button key={s} type="button" onClick={()=>{clock.setPlaying(false);clock.setG(i*16+15);}}
          aria-current={stage===i ? "step" : undefined} className={`rounded px-2 py-1 ${stage===i ? "bg-blue-100 font-semibold text-blue-800" : "text-gray-500 hover:bg-gray-100"}`}>{i+1}. {s}</button>)}
      </figcaption>
      <div className="grid items-center md:grid-cols-[1.2fr_1fr]">
        <ol className="space-y-2 p-3 text-xs">
          {data.terms.map((t,index)=><li key={index}>
            <button type="button" onClick={()=>{clock.setPlaying(false);clock.setG(32+index*4+3);}} aria-pressed={stage===2 && current===index}
              className={`w-full rounded-lg border p-2.5 text-left transition-colors ${stage>=2 && (stage===3 || current===index) ? "border-blue-200 bg-blue-50/60" : "border-gray-100 bg-gray-50/30"}`}>
              <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{background:colours[index]}}/>
              <span className="font-medium"><Tex tex={String.raw`\textcolor{#2563eb}{${fmt(t.weight)}}\,\textcolor{#059669}{e_{${t.i+1}}}\otimes\textcolor{#d97706}{\varepsilon^{${t.j+1}}}`}/></span>
              <span className="mt-1 block pl-4 text-gray-600">{stage>=1 ? `${l?"讀到":"Read"} ${fmt(t.measurement)}` : (l?"等待讀取輸入":"Waiting to read input")}
                {stage>=2 && ` → ${l?"送出":"send"} ${pair(t.vector)}`}</span>
            </button>
          </li>)}
        </ol>
        <svg viewBox="0 0 240 238" className="mx-auto w-full max-w-[280px] p-2" role="img" aria-label={l?"四片張量規則，合成同一支輸出箭頭":"Four tensor pieces reconstruct the same output arrow"}>
          <path d="M20 188H220M72 220V20" stroke="#e5e7eb" fill="none"/>
          {data.basis.map((e,i)=><g key={i}>{arrow(origin,[origin[0]+32*e[0],origin[1]-32*e[1]],"#94a3b8",1.8)}<text x={origin[0]+44*e[0]} y={origin[1]-44*e[1]+4} fontSize="11" textAnchor="middle" fill="#64748b">e{i===0?"₁":"₂"}</text></g>)}
          {arrow(origin,project(EXAMPLE_INPUT),"#64748b",2)}
          <text x="12" y="146" fontSize="11" fill="#64748b">v = (1, 2)</text>
          {stage===2 && arrow(origin,project(data.terms[current].vector),colours[current],3)}
          {stage===3 && data.terms.map((t,i)=>{
            const start = running; running = [running[0]+t.vector[0],running[1]+t.vector[1]];
            return <g key={i} opacity={.65}>{arrow(project(start),project(running),colours[i],2)}</g>;
          })}
          {stage===3 && <>{arrow(origin,project(data.result),"#111827",3)}<text x="156" y="73" fontSize="12" fill="#111827">Tv = {pair(data.result)}</text></>}
          <text x="120" y="227" fontSize="11" textAnchor="middle" fill="#64748b">{l?"固定圖面：輸入 v 與總輸出 Tv 不變":"Fixed page view: v and total Tv stay put"}</text>
        </svg>
      </div>
      <div className="border-t border-gray-100 px-4 py-3 text-xs text-gray-600">
        <label className="flex items-center gap-3">{l?"量測基底":"Measuring basis"}: {angle}°<input type="range" min="0" max="90" value={angle} onChange={e=>changeFrame(Number(e.target.value))} aria-label={l?"轉動基底":"Turn the basis"} className="min-w-0 flex-1 accent-blue-600"/></label>
        <div className="mt-2 flex flex-wrap items-center gap-2">{[0,45,90].map(a=><button type="button" key={a} className={btn} onClick={()=>changeFrame(a)}>{a}°</button>)}<span>{l?"輸入在這組基底下":"Input in this basis"}: <Tex tex={pair(data.input)}/></span></div>
        <p className="mt-2"><RichText text={l?String.raw`每一片＝係數 $\times$ 輸出方向 $\otimes$ 輸入讀尺；點一片查看，最後把箭頭首尾相接。`:String.raw`Each piece = coefficient $\times$ output direction $\otimes$ input reader. Select one, then add the arrows head to tail.`}/></p>
      </div>
    </figure>
    <AnimControls clock={clock}/>
  </div>;
}
