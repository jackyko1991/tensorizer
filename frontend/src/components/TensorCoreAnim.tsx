import {useId,useState} from "react";
import DataflowAnim from "./DataflowAnim";
import Tex,{RichText} from "./Tex";
import {AnimControls,COL,useAnimClock} from "./animKit";
import {usePageText,type PageKey} from "../content/pages";
import {TILE_A,TILE_B,TILE_C,TILE_D,TILE_N,TILE_CYCLE,TILE_PHASE_STARTS,productCell,scalarTile,tensorCoreFrame} from "../tensorCoreModel";
const project=(i:number,j:number,k:number)=>[150+17*(j-i),125+8*(i+j)-18*k];
const polygon=(points:number[][])=>points.map(p=>p.join(",")).join(" ");
type Frame=ReturnType<typeof tensorCoreFrame>;
function TileCube({tile,frame,selected,onSelect,label}:{tile:boolean;frame:Frame;selected:number[];onSelect:(i:number,j:number)=>void;label:string}){
 const values=tile?(frame.tileDone?TILE_D:TILE_C):scalarTile(frame.scalarUpdates);
 const cubes=Array.from({length:64},(_,q)=>({...productCell(q),q})).sort((a,b)=>a.k-b.k||(a.i+a.j)-(b.i+b.j));
 const selectedI=selected[0],selectedJ=selected[1];
 const active=frame.active;
 const cube=(i:number,j:number,k:number,q:number)=>{
  const focused=tile?frame.tileActive:active?.i===i&&active?.j===j&&active?.k===k;
  const done=tile?frame.tileDone:q<frame.scalarUpdates-(frame.phase===1?1:0);
  const fill=focused?COL.curLight:done?COL.blueLight:COL.idle;
  const edge=focused?COL.cur:done?COL.doneLine:COL.idleLine;
  const p=(x:number,y:number,z:number)=>project(i+.16+x,j+.16+y,k+.16+z),d=.68;
  return <g key={q} data-product-cube={q} data-active-product={focused?"true":undefined} stroke={edge} strokeWidth={focused?1.1:.55}>
   <polygon points={polygon([p(d,0,0),p(d,d,0),p(d,d,d),p(d,0,d)])} fill={focused?"#fbbf24":done?"#93c5fd":"#d1d5db"}/>
   <polygon points={polygon([p(0,d,0),p(d,d,0),p(d,d,d),p(0,d,d)])} fill={focused?"#fcd34d":done?"#bfdbfe":"#e5e7eb"}/>
   <polygon points={polygon([p(0,0,d),p(d,0,d),p(d,d,d),p(0,d,d)])} fill={fill}/>
  </g>;
 };
 const outputDone=(i:number,j:number)=>tile?frame.tileDone:frame.scalarUpdates>=(i*4+j+1)*4;
 return <section className="min-w-0 rounded-lg border border-gray-200 bg-white p-3" data-tensor-core-panel={tile?"mma":"fma"}>
  <h4 className="text-center text-sm font-semibold text-gray-800">{label}</h4>
  <svg viewBox="40 10 220 280" className="mx-auto block max-h-80 w-full max-w-[300px]" role="img" aria-label={label}>
   {Array.from({length:16},(_,q)=>{const row=Math.floor(q/4),k=q%4,z=k+.1;
    const hot=tile?frame.tileActive:active?.i===row&&active?.k===k;
    const points=[project(row+.1,-.9,z),project(row+.9,-.9,z),project(row+.9,-.9,z+.8),project(row+.1,-.9,z+.8)];const at=project(row+.5,-.9,z+.4);
    return <g key={`a${q}`}><polygon points={polygon(points)} fill={COL.blueLight} stroke={hot?COL.cur:COL.blue} strokeWidth={hot?1.6:.6}/><text x={at[0]} y={at[1]+2} textAnchor="middle" fontSize="6" fill={COL.blue}>{TILE_A[row][k]}</text></g>;
   })}
   {Array.from({length:16},(_,q)=>{const col=q%4,k=Math.floor(q/4),z=k+.1;
    const hot=tile?frame.tileActive:active?.j===col&&active?.k===k;
    const points=[project(-.9,col+.1,z),project(-.9,col+.9,z),project(-.9,col+.9,z+.8),project(-.9,col+.1,z+.8)];const at=project(-.9,col+.5,z+.4);
    return <g key={`b${q}`}><polygon points={polygon(points)} fill={COL.violetLight} stroke={hot?COL.cur:COL.violet} strokeWidth={hot?1.6:.6}/><text x={at[0]} y={at[1]+2} textAnchor="middle" fontSize="6" fill={COL.violet}>{TILE_B[k][col]}</text></g>;
   })}
   <text x="91" y="46" fontSize="12" fill={COL.blue} textAnchor="middle">A</text><text x="209" y="46" fontSize="12" fill={COL.violet} textAnchor="middle">B</text>
   {cubes.map(c=>cube(c.i,c.j,c.k,c.q))}
   <path d={`M${project(0,0,4)}L${project(4,0,4)}L${project(4,4,4)}L${project(0,4,4)}Z M${project(4,0,4)}L${project(4,0,0)}L${project(4,4,0)}L${project(0,4,0)}L${project(0,4,4)} M${project(4,4,4)}L${project(4,4,0)}`} fill="none" stroke="#94a3b8" strokeWidth=".65" opacity=".7"/>
   {Array.from({length:16},(_,q)=>{const i=Math.floor(q/4),j=q%4,ready=outputDone(i,j),chosen=i===selectedI&&j===selectedJ;const p=(x:number,y:number)=>project(i+x,j+y,-3.85),at=p(.5,.5);
    return <g key={`d${q}`} role="button" tabIndex={0} aria-label={`D ${i+1}, ${j+1}`} aria-pressed={chosen} className="cursor-pointer" onClick={()=>onSelect(i,j)} onKeyDown={e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();onSelect(i,j);}}}>
     <polygon points={polygon([p(.07,.07),p(.93,.07),p(.93,.93),p(.07,.93)])} fill={ready?COL.greenLight:COL.idle} stroke={chosen?COL.cur:ready?COL.green:COL.idleLine} strokeWidth={chosen?1.6:.7}/>
     <text x={at[0]} y={at[1]+2} fontSize="6.5" fill={ready?COL.green:COL.muted} textAnchor="middle">{values[i][j]}</text>
    </g>;
   })}
   <text x="150" y="279" fontSize="12" fill={COL.green} textAnchor="middle">D = AB + C</text>
  </svg>
  <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs"><Tex tex={tile?`\\mathrm{MMA}: ${frame.tileDone?1:0}/1`:`\\mathrm{FMA}: ${frame.scalarUpdates}/64`}/><Tex tex={`${outputDone(selectedI,selectedJ)?"D":"\\mathrm{acc}"}_{${selectedI+1}${selectedJ+1}}=${values[selectedI][selectedJ]}`}/></div>
 </section>;
}
function TensorCoreAnimation(){
 const T=usePageText(),clock=useAnimClock(TILE_CYCLE,150),frame=tensorCoreFrame(clock.g);
 const [selected,setSelected]=useState([0,0]);const [i,j]=selected;
 return <div ref={clock.root} data-testid="tensor-core-animation">
  <figure className="overflow-hidden rounded-lg border border-gray-200 bg-white">
   <figcaption className="flex items-center justify-between gap-3 border-b border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-600"><span>{T("flow.tensorcore.title")}</span><span className="font-mono text-blue-700">{frame.phase+1}/4</span></figcaption>
   <div className="grid gap-3 p-3 md:grid-cols-2"><TileCube tile={false} frame={frame} selected={selected} onSelect={(i,j)=>setSelected([i,j])} label="Pascal · FMA"/><TileCube tile frame={frame} selected={selected} onSelect={(i,j)=>setSelected([i,j])} label="Volta · Tensor Core MMA"/></div>
   <ol className="grid gap-1 border-t border-gray-100 p-3 text-xs sm:grid-cols-2">{[0,1,2,3].map(k=><li key={k}><button type="button" aria-current={frame.phase===k?"step":undefined} onClick={()=>{clock.setPlaying(false);clock.setG(TILE_PHASE_STARTS[k]);}} className={`flex h-full w-full gap-2 rounded px-2 py-2 text-left ${frame.phase===k?"bg-blue-50 text-gray-900":"text-gray-500 hover:bg-gray-50"}`}><span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] ${frame.phase===k?"bg-blue-600 text-white":"bg-gray-200"}`}>{k+1}</span><span><RichText text={T(`flow.tensorcore.s${k}` as PageKey)}/></span></button></li>)}</ol>
   <div className="border-t border-gray-100 bg-gray-50/50 px-3 py-3"><div className="flex flex-wrap justify-center gap-x-2 gap-y-1 text-sm" aria-live="polite"><Tex tex={`D_{${i+1}${j+1}}=${TILE_C[i][j]}`}/>{Array.from({length:TILE_N},(_,k)=><Tex key={k} tex={`+${TILE_A[i][k]}\\cdot ${TILE_B[k][j]}`}/>)}<Tex tex={`=${TILE_D[i][j]}`}/></div></div>
   <p className="border-t border-gray-100 px-3 py-2 text-xs leading-relaxed text-gray-600"><RichText text={T("flow.tensorcore.caption")}/></p>
  </figure>
  <AnimControls clock={clock}/>
  <details className="mt-3 text-xs"><summary className="cursor-pointer text-blue-700">{T("flow.tensorcore.result")}</summary><div className="mt-2 flex justify-center"><Tex tex={`D=\\begin{pmatrix}${TILE_D.map(r=>r.join("&")).join("\\\\")}\\end{pmatrix}`}/></div></details>
  <p className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs"><a href="https://images.nvidia.com/aem-dam/Solutions/Data-Center/tensorcore/Volta-Tensor-Core_30fps_FINAL_994x559.gif" target="_blank" rel="noreferrer" className="text-blue-700 underline">NVIDIA · GIF ↗</a><a href="https://docs.nvidia.com/cuda/archive/11.4.4/volta-tuning-guide/index.html#tensor-operations" target="_blank" rel="noreferrer" className="text-blue-700 underline">NVIDIA · Volta Tuning Guide ↗</a></p>
 </div>;
}
export default function VectorizationExamples(){
 const [mode,setMode]=useState(0),T=usePageText(),uid=useId().replaceAll(":","");
 return <div data-testid="vectorization-examples" className="space-y-3">
  <div role="tablist" aria-label={T("flow.tensorcore.title")} className="flex flex-wrap gap-1 border-b border-gray-200">{["flow.tensorcore.tabVector","flow.tensorcore.tabHardware"].map((key,k)=><button key={key} type="button" id={`${uid}-tab-${k}`} aria-selected={mode===k} aria-controls={`${uid}-panel`} role="tab" tabIndex={mode===k?0:-1} onClick={()=>setMode(k)} onKeyDown={e=>{if(["ArrowLeft","ArrowRight","Home","End"].includes(e.key)){e.preventDefault();const next=e.key==="Home"?0:e.key==="End"?1:1-k;setMode(next);document.getElementById(`${uid}-tab-${next}`)?.focus();}}} className={`border-b-2 px-3 py-2 text-xs font-medium ${mode===k?"border-blue-600 text-blue-800":"border-transparent text-gray-500 hover:text-blue-700"}`}>{T(key as PageKey)}</button>)}</div>
  <div id={`${uid}-panel`} role="tabpanel" aria-labelledby={`${uid}-tab-${mode}`}>{mode===0?<DataflowAnim id="s2"/>:<><p className="mb-3 text-sm leading-relaxed text-gray-700">{T("flow.tensorcore.intro")}</p><TensorCoreAnimation/></>}</div>
 </div>;
}
