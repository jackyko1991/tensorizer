import { useState } from "react";
import { loopRadii } from "../limitationsModel";
import { Arrow } from "./MathsAnimPics";

/** Interactive illustrations embedded in each card’s own experiment or challenge. */
export default function LimitationsFigure({ kind, l }: { kind: string; l: number }) {
  const [angle, setAngle] = useState(kind === "bridge" ? 90 : 0);
  const theta = angle * Math.PI / 180;
  const radii = loopRadii(angle);
  const captions: Record<string, string[]> = {
    scope: ["The blue arrow stays fixed. The grey measuring axes turn; only the coordinate numbers change.", "藍色箭頭不動，灰色量測方向轉動；改變的是分量數字。"],
    spinors: ["Left: direction. Right: the half-angle phase of one spin component, shown schematically.", "左邊記方向，右邊示意一個自旋分量的半角相位。"],
    connections: ["Two paths around excluded flux. Compare their wave-cycle clocks after they meet.", "兩條路繞過被排除的磁通，會合後比較各自的波週期小鐘。"],
    topology: ["A loop can shrink on the filled sheet. A loop around the hole cannot shrink while staying on the sheet.", "完整紙上的圈能縮成一點；繞著洞的圈，不能一直留在紙面上縮成一點。"],
    entanglement: ["One row for A, one column for B: keep the pair’s outcomes in one table.", "列記錄 A、欄記錄 B，把兩者如何配對放進同一張表。"],
    frontiers: ["A proposed rule needs a prediction that can be checked against a measurement.", "提出規則以後，還要做出能和量測比較的預測。"],
    bridge: ["Turn the second stick: its alignment and the shaded oriented area change together.", "轉動第二根棍子，平行程度和陰影的定向面積一起改變。"],
  };
  const label = (x: number, y: number, text: string) => <text x={x} y={y} textAnchor="middle" fontSize="12" fill="#475569">{text}</text>;
  return <figure className="rounded-lg border border-blue-100 bg-blue-50/40 p-4" data-testid={`limitations-figure-${kind}`}>
    <svg viewBox="0 0 500 220" className="mx-auto w-full max-w-xl" role="img" aria-label={captions[kind]?.[l]}>
      {kind === "scope" && <>
        <g transform={`rotate(${-angle} 150 112)`} stroke="#94a3b8" strokeWidth="2"><path d="M70 112H235M150 195V28"/><path d="M227 107L235 112L227 117M145 36L150 28L155 36" fill="none"/></g>
        <g fill="#2563eb"><path d="M150 112H205" fill="none" stroke="#2563eb" strokeWidth="4" strokeLinecap="round"/><polygon points="215,112 202,105 202,119"/></g>
        {label(150,212,l ? "桌上的箭頭沒有動" : "Arrow fixed on the table")}
        <rect x="295" y="57" width="180" height="108" rx="12" fill="white" stroke="#bfdbfe"/>
        {label(385,89,l ? "新視角裡的分量" : "Coordinates in the new view")}
        {label(385,125,`(${Math.cos(theta).toFixed(2)}, ${(-Math.sin(theta)).toFixed(2)})`)}
      </>}
      {kind === "spinors" && <>
        {[125,375].map(x => <circle key={x} cx={x} cy="90" r="55" fill="white" stroke="#bfdbfe"/>)}
        <g transform={`rotate(${-angle} 125 90)`}><g fill="#2563eb"><path d="M125 90H161" fill="none" stroke="#2563eb" strokeWidth="3" strokeLinecap="round"/><polygon points="170,90 158,84 158,96"/></g></g>
        <g transform={`rotate(${-angle/2} 375 90)`}><g fill="#7c3aed"><path d="M375 90H411" fill="none" stroke="#7c3aed" strokeWidth="3" strokeLinecap="round"/><polygon points="420,90 408,84 408,96"/></g></g>
        {label(125,177,l ? "方向：轉一圈復原" : "Direction: one turn")}
        {label(375,177,l ? "相位示意：轉兩圈復原" : "Phase schematic: two turns")}
      </>}
      {kind === "connections" && <>
        <path d="M45 112C120 5 240 5 310 112C240 215 120 215 45 112" stroke="#93c5fd" fill="none" strokeWidth="3"/>
        <circle cx="178" cy="112" r="30" fill="#dbeafe" stroke="#60a5fa"/>
        {label(178,116,l ? "磁通" : "Flux")}
        <circle cx={178 + 116*Math.cos(theta)} cy={112 - 76*Math.sin(theta)} r="6" fill="#2563eb"/>
        <circle cx="410" cy="95" r="43" fill="white" stroke="#c4b5fd"/>
        <g transform={`rotate(${-angle/2} 410 95)`}><path d="M410 95H445" stroke="#7c3aed" strokeWidth="3"/></g>
        {label(410,174,l ? "沿路累積的相位" : "Accumulated phase")}
      </>}
      {kind === "topology" && <>
        {[125,375].map((x,i) => <g key={x}><rect x={x-95} y="22" width="190" height="154" rx="18" fill="#dbeafe" stroke="#93c5fd"/><circle cx={x} cy="99" r={Math.max(2,radii[i])} fill={i===0&&radii[0]===0?"#2563eb":"none"} stroke="#2563eb" strokeWidth="3" strokeDasharray={i===0&&radii[0]===0?undefined:"5 3"}/></g>)}
        <circle cx="375" cy="99" r="28" fill="white" stroke="#93c5fd"/>
        {label(125,207,l ? "圈可以縮小到一點" : "Loop can shrink to a point")}
        {label(375,207,l ? "中間的洞擋住了" : "The hole gets in the way")}
      </>}
      {kind === "entanglement" && <>
        {[['00','50%',true],['01','0%',false],['10','0%',false],['11','50%',true]].map(([name,p,filled],i) => <g key={String(name)}>
          <rect x={140+(i%2)*115} y={18+Math.floor(i/2)*90} width="106" height="78" rx="9" fill={filled ? '#dbeafe' : 'white'} stroke="#93c5fd"/>
          {label(193+(i%2)*115,48+Math.floor(i/2)*90,String(name))}
          {label(193+(i%2)*115,76+Math.floor(i/2)*90,String(p))}
        </g>)}
        {label(250,215,l ? "兩種結果，一份共同紀錄" : "Two outcomes, one shared record")}
      </>}
      {kind === "frontiers" && <>
        {(l ? ['提出規則','算出預測','和量測比較'] : ['Propose a rule','Predict a result','Compare with data']).map((t,i) => <g key={t}>
          <rect x={10+i*170} y="65" width="140" height="80" rx="12" fill="white" stroke="#93c5fd"/>
          {label(80+i*170,110,t)}
          {i<2&&<path d={`M${155+i*170} 105h19l-6-5m6 5-6 5`} stroke="#2563eb" fill="none"/>}
        </g>)}
      </>}
      {kind === "bridge" && <>
        <path d={`M120 140h85l${85*Math.cos(theta)} ${-85*Math.sin(theta)}h-85Z`} fill="#bfdbfe"/>
        <Arrow x1={120} y1={140} x2={205} y2={140} color="#0f766e" w={4} head={12}/>
        <Arrow x1={120} y1={140} x2={120+85*Math.cos(theta)} y2={140-85*Math.sin(theta)} color="#2563eb" w={4} head={12}/>
        {label(205,162,'a')}{label(110+95*Math.cos(theta),135-95*Math.sin(theta),'b')}
        {label(380,80,`${l ? '平行程度' : 'Alignment'}: ${Math.cos(theta).toFixed(2)}`)}
        {label(380,118,`${l ? '定向面積' : 'Oriented area'}: ${Math.sin(theta).toFixed(2)}`)}
      </>}
    </svg>
    {kind === "topology" && <div className="mt-2 space-y-3">
      <label className="flex items-center gap-3 text-xs text-blue-900">{l?"把圈縮小":"Shrink the loops"}<input type="range" min="0" max="100" value={angle} onChange={e=>setAngle(Number(e.target.value))} aria-label={l?"把圈縮小":"Shrink the loops"} className="min-w-0 flex-1 accent-blue-600"/></label>
      <div className="flex gap-2">{[0,100].map(value=><button key={value} type="button" onClick={()=>setAngle(value)} className="rounded border border-blue-200 bg-white px-3 py-1 text-xs text-blue-800">{value===0?(l?"重新開始":"Reset"):(l?"縮到最小":"Shrink fully")}</button>)}</div>
    </div>}
    {['scope','spinors','connections','bridge'].includes(kind) && <>
      <label className="mt-2 flex items-center gap-3 text-xs text-blue-900">{l ? "角度" : "Angle"}: {angle}°
        <input className="min-w-0 flex-1 accent-blue-600" aria-label={l ? "圖解角度" : "Diagram angle"} type="range" min="0" max={kind === 'spinors' ? 720 : kind === 'bridge' ? 180 : 360} value={angle} onChange={e=>setAngle(Number(e.target.value))}/>
      </label>
      <div className="mt-3 flex gap-2">{(kind === 'spinors' ? [0,360,720] : kind === 'bridge' ? [0,90,180] : [0,90,180,360]).map(a=><button type="button" key={a} onClick={()=>setAngle(a)} className="rounded border border-blue-200 bg-white px-3 py-1 text-xs text-blue-800">{a}°</button>)}</div>
    </>}
    <figcaption className="mt-3 text-xs leading-relaxed text-gray-600">{captions[kind]?.[l]}</figcaption>
  </figure>;
}
