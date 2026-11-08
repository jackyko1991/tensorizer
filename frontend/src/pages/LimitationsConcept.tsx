import { useState } from "react";
import { useLang } from "../i18n";
import PageToc from "../components/PageToc";
import Tex from "../components/Tex";

const sections = [
  { id: "scope", title: ["Which kind of tensor?", "先分清楚：哪一種張量？"],
    text: ["A spacetime tensor transforms through products of the tangent and cotangent representations. A numerical tensor is an array that can store tensor components, spinor components or a quantum wavefunction. These are different uses of the same word. Quantum mechanics relies on tensor products; its success does not stop at fermions.", "時空張量依切空間與對偶空間的張量表示變換；程式裡的 tensor 則是數值陣列，可以儲存張量分量、旋量分量或量子波函數。兩者不能混為一談。量子力學大量使用張量積，並沒有在費米子面前整體失效。"],
    eq: String.raw`\mathrm{Cl}(V,q)=T(V)/\langle v\otimes v-q(v)1\rangle` },
  { id: "spinors", title: ["Fermions need spinors", "費米子需要旋量"],
    text: ["An ordinary tensor built from spacetime vectors returns under a 2π spatial rotation. A spin-½ state changes sign under the lifted rotation and returns after 4π. Spinors transform under Spin(3) ≅ SU(2), or Spin⁺(1,3) ≅ SL(2,ℂ) in relativity. A global minus sign alone changes no measurement probability; interference can reveal a phase relative to an unrotated branch. Spinors are not ordinary spacetime tensors, but their components can be stored in arrays and their states still form tensor products.", "由時空向量建立的普通張量在空間旋轉 2π 後回到原樣；自旋 ½ 的態則在提升到自旋群的旋轉下變號，到 4π 才復原。三維使用 Spin(3) ≅ SU(2)，相對論使用 Spin⁺(1,3) ≅ SL(2,ℂ)。單獨的整體負號不改變測量機率；與未旋轉分支比較的干涉才能顯示相對相位。旋量不是普通時空張量，但可用陣列儲存，也能形成量子張量積。"],
    eq: String.raw`U(\theta)=e^{-i\theta\sigma_z/2},\qquad U(2\pi)=-1,\quad U(4\pi)=1` },
  { id: "connections", title: ["Connections compare neighbouring fibres", "聯絡：如何比較不同位置的幾何量"],
    text: ["Christoffel symbols are connection coefficients with an inhomogeneous coordinate transformation, not tensor components. They can vanish at a point in a suitable freely falling frame while curvature remains. A gauge potential is a connection: in a fixed gauge, the electromagnetic potential is also a spacetime one-form, but it changes inhomogeneously under gauge transformations. In the Aharonov–Bohm effect, local field strength along the electron paths does not determine the loop holonomy around excluded magnetic flux.", "克里斯多福符號是聯絡係數，其座標變換含非齊次項，因此不是張量分量。在合適的自由落體座標中，它們可在一點消失，但曲率未必消失。規範勢是聯絡：固定規範下，電磁勢也是時空一形式；非齊次的是規範變換，不能因此說它不是時空協向量。AB 效應說明電子路徑上的局域場強，無法單獨決定繞過被排除磁通的迴路相位。"],
    eq: String.raw`D_\mu=\partial_\mu+\Gamma_\mu,\qquad W(C)=\exp\!\left(\frac{iq}{\hbar}\oint_C A_\mu\,dx^\mu\right)` },
  { id: "topology", title: ["Global topology and quantum entanglement", "全局拓撲與量子糾纏"],
    text: ["Local tensor values alone do not specify bundle topology or boundary conditions. Curvature forms can nevertheless be integrated to obtain topological invariants such as Chern numbers. Entangled states belong to tensor-product Hilbert spaces; Bell correlations exclude local hidden-variable models under the theorem’s assumptions, not tensor algebra or relativistic quantum field theory. They cannot transmit information faster than light.", "局部張量數值本身不能指定纖維叢拓撲與邊界條件；但曲率微分形式的積分仍可給出陳數等拓撲不變量。糾纏態正是張量積希爾伯特空間中的態。貝爾關聯排除的是定理假設下的局域隱變數模型，並非張量代數或相對論量子場論，也不能用來超光速傳訊。"],
    eq: String.raw`|\Phi^+\rangle=\frac{|0\rangle\otimes|0\rangle+|1\rangle\otimes|1\rangle}{\sqrt{2}}` },
  { id: "frontiers", title: ["Research frontiers are not established failures", "研究前沿不等於已證實的失效"],
    text: ["Noncommutative and nonassociative geometry occur in proposed models of quantum gravity and string backgrounds. They are not established properties of observed spacetime. Tensor products and suitable generalized differential calculi still exist in many of these settings. Octonion multiplication is nonassociative, while Clifford algebras are associative: one is not a universal cure for the other.", "非對易與非結合幾何出現在部分量子引力、弦論背景模型中，尚不是已觀測時空的定論。許多這類模型仍有張量積與廣義微分演算。八元數乘法不滿足結合律，但克利福德代數滿足結合律；後者不是解決所有前沿問題的萬用替代品。"],
    eq: String.raw`(ab)c=a(bc)\quad\text{in a Clifford algebra}` },
  { id: "bridge", title: ["Next: Clifford algebra", "下一步：克利福德幾何代數"],
    text: ["Clifford algebra puts a metric into vector multiplication. Its bivectors describe oriented planes; rotors implement rotations and boosts; its representations supply the gamma matrices acting on spinors. Tensors, spinors, connections and topology are complementary structures, not nested levels of a single hierarchy. Explore their relationships through diagrams, animation and code.", "克利福德代數把度規放進向量乘法：雙向量表示定向平面，轉子實現旋轉與洛侖茲推進，其表示提供作用於旋量的 γ 矩陣。張量、旋量、聯絡與拓撲是互補的結構，不是單一由小到大的包含階層。下一站用圖解、動畫與程式探索它們的關係。"],
    eq: String.raw`ab=a\cdot b+a\wedge b,\qquad v'=Rv\widetilde R` },
];

export default function LimitationsConcept() {
  const { lang } = useLang();
  const l = lang.startsWith("zh") ? 1 : 0;
  const [angle, setAngle] = useState(0);
  return <div className="mx-auto flex max-w-6xl gap-8 px-5 pb-10">
    <PageToc scrollRoot="concept-scroll" items={sections.map((s,i) => ({id:s.id,label:s.title[l],mark:i+1}))}/>
    <article className="min-w-0 max-w-4xl flex-1 space-y-10" data-testid="limitations-concept">
      <header><p className="text-sm font-medium text-blue-700">{l ? "概念 · 適用範圍" : "Concept · Scope"}</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight">{l ? "張量的適用範圍與限制" : "The scope and limits of tensors"}</h1>
        <p className="mt-3 text-gray-600">{l ? "當普通時空張量不夠用，我們需要加上什麼幾何結構？" : "What additional geometry do we need when ordinary spacetime tensors are not enough?"}</p></header>
      {sections.map((s,i) => <section id={s.id} key={s.id} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-4 sm:p-6">
        <h2 className="flex items-center gap-3 text-lg font-semibold"><span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm text-white">{i+1}</span>{s.title[l]}</h2>
        <p className="mt-4 text-sm leading-relaxed text-gray-700">{s.text[l]}</p>
        <div className="mt-4 overflow-x-auto rounded-lg bg-gray-50 p-3"><Tex tex={s.eq} display/></div>
        {s.id === "spinors" && <figure className="mt-4 rounded-lg border border-blue-100 bg-blue-50/50 p-4">
          <svg viewBox="0 0 400 135" role="img" aria-label={l ? "向量與旋量的旋轉比較" : "Vector and spinor rotation comparison"} className="mx-auto w-full max-w-lg">
            <circle cx="90" cy="62" r="40" fill="white" stroke="#bfdbfe"/>
            <g transform={`rotate(${-angle} 90 62)`}><path d="M90 62H125l-8-5m8 5-8 5" stroke="#2563eb" strokeWidth="3" fill="none"/></g>
            <text x="90" y="125" textAnchor="middle" fontSize="13">{l ? "向量方向" : "Vector direction"}</text>
            <circle cx="285" cy="62" r="40" fill="white" stroke="#bfdbfe"/>
            <g transform={`rotate(${-angle/2} 285 62)`}><path d="M285 62H320l-8-5m8 5-8 5" stroke="#7c3aed" strokeWidth="3" fill="none"/></g>
            <text x="285" y="125" textAnchor="middle" fontSize="13">{l ? "旋量相位（示意）" : "Spinor phase (schematic)"}</text>
          </svg>
          <label className="mt-3 flex items-center gap-3 text-sm">θ = {angle}°<input className="min-w-0 flex-1 accent-blue-600" aria-label={l ? "旋轉角度" : "Rotation angle"} type="range" min="0" max="720" step="1" value={angle} onChange={e=>setAngle(Number(e.target.value))}/></label>
          <div className="mt-2 flex gap-2">{[0,360,720].map(a=><button key={a} className="rounded border border-blue-200 bg-white px-3 py-1 text-xs" onClick={()=>setAngle(a)}>{a}°</button>)}</div>
          <figcaption className="mt-3 text-xs text-gray-600">{l ? "右圖表示一個自旋分量的半角相位，不是電子在空間中的實體自轉。" : "The right dial depicts the half-angle phase of a spin component, not a physical spinning electron."}</figcaption>
        </figure>}
        {s.id === "bridge" && <a className="mt-4 inline-block rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white" href="https://jackyko1991.github.io/clifford/" target="_blank" rel="noreferrer">{l ? "探索克利福德代數" : "Explore Clifford algebra"} →</a>}
      </section>)}
      <footer className="space-y-2 text-sm text-gray-600"><p className="font-semibold">{l ? "延伸閱讀與來源" : "Sources and further reading"}</p>
        <a className="block text-blue-700 underline" href="https://www.damtp.cam.ac.uk/user/tong/qft/qfthtml/S4.html" target="_blank" rel="noreferrer">David Tong: The Dirac Equation</a>
        <a className="block text-blue-700 underline" href="https://www.damtp.cam.ac.uk/user/tong/gaugetheory.html" target="_blank" rel="noreferrer">David Tong: Gauge Theory</a>
        <a className="block text-blue-700 underline" href="https://iqim.caltech.edu/wp-content/uploads/2019/08/ACM270_complete_lecture_notes.pdf" target="_blank" rel="noreferrer">Caltech: Quantum and classical information processing with tensors</a>
      </footer>
    </article>
  </div>;
}
