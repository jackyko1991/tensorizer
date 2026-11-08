import { useLang } from "../i18n";
import { Arrow } from "./MathsAnimPics";

export type AnalogyKind = "tensor" | "einsum" | "penrose" | "topo" | "roof" | "geometry";
const blue = "#2563eb", amber = "#d97706", green = "#059669", ink = "#334155";
const descriptions: Record<AnalogyKind, string[]> = {
  tensor: ["A street address, an apartment floor and door, and a choice of buildings.", "街道上的門牌、公寓的樓層和門號，以及城市裡的不同大樓。"],
  einsum: ["Groceries and a receipt: multiply quantity by price, then add the items.", "購物袋與收據：數量乘單價，再把各品項加起來。"],
  penrose: ["Two devices joined by one cable, with other ports still open.", "兩個小裝置以一條線相接，其餘接頭仍留在外面。"],
  topo: ["Chop all onions before frying, or chop and fry one onion at a time.", "先把洋蔥全部切好再一起炒，或每切一顆就炒一回。"],
  roof: ["A delivery truck brings ingredients to a chef working at a stove.", "送貨車把食材運到廚房，主廚在爐前料理。"],
  geometry: ["A sorting station measures an input with two rulers and routes it along output directions.", "分送站用兩把讀尺量取輸入，再沿指定的方向送出。"],
};
const text = (x: number, y: number, value: string, colour = ink, size = 11) => <text x={x} y={y} textAnchor="middle" fill={colour} fontSize={size}>{value}</text>;
function Onion({ x, y, scale = 1 }: { x: number; y: number; scale?: number }) {
  return <g transform={`translate(${x} ${y}) scale(${scale})`}>
    <ellipse cy="3" rx="12" ry="14" fill="#f5d0fe" stroke="#a855f7" strokeWidth="1.5"/>
    <path d="M0-10Q-6 2 0 16M0-10Q6 2 0 16" stroke="#c084fc" fill="none"/>
    <path d="M-1-10L-5-21M0-10L2-24M1-10L7-20" stroke={green} strokeWidth="2" strokeLinecap="round"/>
  </g>;
}
function Pan({ x, y, scale = 1 }: { x: number; y: number; scale?: number }) {
  return <g transform={`translate(${x} ${y}) scale(${scale})`}>
    <path d="M-27 0Q-24 25 0 26Q24 25 27 0Z" fill="#d1fae5" stroke={green} strokeWidth="2"/>
    <ellipse rx="27" ry="7" fill="#ecfdf5" stroke={green} strokeWidth="2"/>
    <path d="M25 9L49-2" stroke={ink} strokeWidth="5" strokeLinecap="round"/>
    {[-11,0,11].map(x=><ellipse key={x} cx={x} cy="0" rx="4" ry="2.5" fill="#e9d5ff" stroke="#c084fc"/>)}
  </g>;
}
function Building({ x, y, rows = 3, cols = 3, hot = false, scale = 1 }: {x:number;y:number;rows?:number;cols?:number;hot?:boolean;scale?:number}) {
  return <g transform={`translate(${x} ${y}) scale(${scale})`}>
    <rect width={cols*19+10} height={rows*24+15} rx="3" fill="#eff6ff" stroke={blue} strokeWidth="1.5"/>
    {Array.from({length:rows},(_,r)=>Array.from({length:cols},(_,c)=><rect key={`${r}-${c}`} x={8+c*19} y={9+r*24} width="12" height="16" rx="1" fill={hot&&r===1&&c===1?"#fbbf24":"white"} stroke={hot&&r===1&&c===1?amber:"#93c5fd"}/>))}
  </g>;
}

/** Literal, static pictures of the analogy, before its explanatory text. */
export default function AnalogyPicture({ kind }: { kind: AnalogyKind }) {
  const { lang } = useLang();
  const l = lang.startsWith("zh") ? 1 : 0;
  return <figure className="rounded-lg border border-amber-100 bg-white/80 px-2 py-1" data-testid={`analogy-picture-${kind}`}>
    <svg viewBox="0 0 400 200" className="mx-auto block w-full max-w-[520px]" role="img" aria-label={descriptions[kind][l]}>
      {kind === "tensor" && <>
        <path d="M130 20V170M260 20V170" stroke="#e2e8f0" strokeDasharray="3 4"/>
        {[12,51,90].map((x,i)=><g key={x}><path d={`M${x-4} 104L${x+13} 86L${x+30} 104Z`} fill="#fde68a" stroke={amber}/><rect x={x} y="104" width="26" height="36" fill="#fffbeb" stroke={amber}/><rect x={x+10} y="119" width="8" height="21" fill={i===1?"#fbbf24":"white"} stroke={amber}/>{text(x+13,156,String(i+1))}</g>)}
        <Building x={163} y={50} hot/>
        <Building x={277} y={53} rows={3} cols={2} scale={.85}/>
        <Building x={313} y={30} rows={4} cols={3} hot/>
        {text(64,184,l?"門牌":"House number")}{text(197,184,l?"樓層 ＋ 門號":"Floor + door")}{text(330,184,l?"大樓 ＋ 樓層 ＋ 門號":"Building + floor + door",ink,10)}
      </>}
      {kind === "einsum" && <>
        <path d="M35 81H136L124 176H47Z" fill="#fef3c7" stroke={amber} strokeWidth="2"/>
        <path d="M57 86V64a28 28 0 0 1 56 0V86" stroke={amber} strokeWidth="3" fill="none"/>
        {[67,99].map(x=><g key={x}><circle cx={x} cy="112" r="15" fill="#fca5a5" stroke="#dc2626"/><path d={`M${x} 99l4-11`} stroke={green} strokeWidth="3"/><ellipse cx={x+7} cy="94" rx="7" ry="3" fill="#86efac"/></g>)}
        <path d="M58 155Q45 141 58 133Q72 127 86 132L114 148Q119 164 105 166Z" fill="#fcd9a8" stroke={amber}/>
        <rect x="205" y="16" width="161" height="168" rx="5" fill="white" stroke="#94a3b8" strokeWidth="1.5"/>
        {text(285,42,l?"收據":"Receipt",ink,14)}
        <path d="M220 53H351M220 130H351" stroke="#cbd5e1" strokeDasharray="3 3"/>
        {text(285,79,"2 × 3 = 6",ink,14)}{text(285,111,"1 × 4 = 4",ink,14)}
        {text(285,157,`${l?"總計":"Total"}  10`,green,16)}
        <Arrow x1={145} y1={112} x2={190} y2={112} color={amber} w={2} head={7}/>
      </>}
      {kind === "penrose" && <>
        <rect x="80" y="61" width="82" height="65" rx="8" fill="#dbeafe" stroke={blue} strokeWidth="2"/>
        <rect x="257" y="61" width="82" height="65" rx="8" fill="#dbeafe" stroke={blue} strokeWidth="2"/>
        {text(121,101,"A",blue,22)}{text(298,101,"B",blue,22)}
        <path d="M31 93H80M121 126V167M298 30V61M339 93H379" fill="none" stroke={blue} strokeWidth="3"/>
        <path d="M162 93H257" stroke={amber} strokeWidth="4"/>
        <rect x="202" y="85" width="14" height="16" rx="3" fill="#d1fae5" stroke={green} strokeWidth="2"/>
        {[[31,93],[121,167],[298,30],[379,93]].map(([x,y])=><circle key={`${x}-${y}`} cx={x} cy={y} r="5" fill="white" stroke={blue} strokeWidth="2"/>)}
        {text(48,78,"i",blue,13)}{text(209,73,"j",amber,13)}{text(108,159,"k",blue,13)}{text(365,78,"l",blue,13)}
        {text(209,187,l?"接起來的線  ／  留在外面的接頭":"Joined cable  /  remaining open ports",ink,11)}
      </>}
      {kind === "topo" && <>
        {text(200,20,l?"先全部切好，再一起炒":"Chop the batch, then fry together",ink,12)}
        {[34,66,98].map(x=><Onion key={x} x={x} y={64} scale={.8}/>)}
        <Arrow x1={122} y1={67} x2={163} y2={67} color={blue} w={2} head={6}/>
        <rect x="176" y="71" width="60" height="12" rx="3" fill="#fcd9a8" stroke={amber}/>
        <path d="M180 54H224V64H180Z" fill="#e2e8f0" stroke="#64748b"/><path d="M224 54H240V64H224Z" fill="#a16207"/>
        <Arrow x1={250} y1={67} x2={286} y2={67} color={blue} w={2} head={6}/><Pan x={329} y={62}/>
        <path d="M20 106H380" stroke="#e2e8f0"/>
        {text(200,126,l?"或是每切一顆，就炒一回":"Or chop and fry one at a time",ink,12)}
        {[15,145,275].map(x=><g key={x}><Onion x={x+15} y={164} scale={.6}/><Arrow x1={x+34} y1={166} x2={x+57} y2={166} color={amber} w={1.5} head={5}/><Pan x={x+82} y={160} scale={.6}/></g>)}
      </>}
      {kind === "roof" && <>
        {text(96,27,l?"送食材":"Deliver ingredients",blue,12)}{text(303,27,l?"做料理":"Cook the meal",amber,12)}
        <rect x="24" y="87" width="98" height="54" rx="3" fill="#dbeafe" stroke={blue} strokeWidth="2"/>
        <path d="M122 106H150L166 125V141H122Z" fill="#93c5fd" stroke={blue} strokeWidth="2"/>
        <path d="M132 111H146L157 124H132Z" fill="white"/>
        {[51,144].map(x=><g key={x}><circle cx={x} cy="144" r="11" fill={ink}/><circle cx={x} cy="144" r="5" fill="white"/></g>)}
        <path d="M16 159H179" stroke="#cbd5e1" strokeWidth="2"/>
        {[43,69,95].map(x=><rect key={x} x={x} y="102" width="18" height="20" rx="2" fill="#fef3c7" stroke={amber}/>)}
        <Arrow x1={182} y1={114} x2={233} y2={114} color={blue} w={2} head={7}/>
        <circle cx="267" cy="72" r="13" fill="#ffedd5" stroke={ink}/>
        <path d="M252 65V56C241 45 253 35 261 43C266 30 282 35 280 45C292 40 299 56 282 60V65Z" fill="white" stroke={ink} strokeWidth="1.5"/>
        <path d="M250 91Q267 82 284 91L291 144H244Z" fill="white" stroke={ink}/>
        <path d="M256 96H278L281 141H252Z" fill="#dbeafe" stroke={blue}/>
        <path d="M283 100L311 112" stroke={ink} strokeWidth="3" strokeLinecap="round"/>
        <rect x="291" y="147" width="83" height="24" rx="3" fill="#e2e8f0" stroke="#64748b"/>
        <Pan x={326} y={120} scale={.85}/>
        <path d="M316 139Q320 134 324 140M332 139Q336 134 340 140" stroke="#f59e0b" strokeWidth="2" fill="none"/>
      </>}
      {kind === "geometry" && <>
        <rect x="13" y="72" width="66" height="55" rx="5" fill="#dbeafe" stroke={blue} strokeWidth="2"/>
        {text(46,93,l?"輸入":"Input",blue)}{text(46,114,"(1, 2)",blue,14)}
        <Arrow x1={79} y1={92} x2={118} y2={53} color={blue} w={2} head={6}/><Arrow x1={79} y1={107} x2={118} y2={148} color={blue} w={2} head={6}/>
        {[42,137].map((y,i)=><g key={y}><rect x="122" y={y} width="99" height="23" rx="2" fill="#fef3c7" stroke={amber}/>{[0,1,2,3,4,5,6].map(k=><line key={k} x1={133+k*12} y1={y} x2={133+k*12} y2={y+(k%2?8:13)} stroke={amber}/>)}{text(171,y-8,i?"ε²":"ε¹",amber,13)}</g>)}
        <Arrow x1={225} y1={54} x2={258} y2={85} color={amber} w={2} head={6}/><Arrow x1={225} y1={149} x2={258} y2={117} color={amber} w={2} head={6}/>
        <rect x="262" y="69" width="63" height="63" rx="6" fill="#d1fae5" stroke={green} strokeWidth="2"/>
        {text(293,104,l?"分送":"Route",green,14)}
        <Arrow x1={325} y1={101} x2={382} y2={101} color={green} w={3} head={8}/>{text(369,87,"e₁",green,13)}
        <Arrow x1={294} y1={69} x2={294} y2={25} color={green} w={3} head={8}/>{text(312,33,"e₂",green,13)}
        {text(207,190,l?"先量份量，再沿指定方向送出":"Read an amount, then choose its output direction",ink,11)}
      </>}
    </svg>
  </figure>;
}
