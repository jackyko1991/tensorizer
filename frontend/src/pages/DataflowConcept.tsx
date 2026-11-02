import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import { RichText } from "../components/Tex";
import { usePageText, type PageKey } from "../content/pages";

const SECTIONS = [1, 2, 3, 4, 5, 6, 7] as const;

const CODE: Record<number, string> = {
  1: `x = torch.arange(6.0).reshape(2, 3)   # shape (2, 3), one buffer of 6 numbers
x.stride()            # (3, 1): +3 to the next row, +1 to the next column
x.T.stride()          # (1, 3): the same memory, strides swapped, nothing copied
x.T.is_contiguous()   # False
x.dtype, x.device     # torch.float32, device(type='cpu')`,
  2: `# a Python loop: one interpreter round trip per pair
for a in range(N):
    for b in range(N):
        r[a, b] = x[b] - x[a]

# one kernel for all N² pairs: broadcasting reuses x with stride 0
r = x[None, :, :] - x[:, None, :]          # (N, N, 3)
x[:, None, :].expand(N, N, 3).stride()     # (3, 0, 1): the new axis costs no memory`,
  3: `dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(4096, 4096, device=dev)    # created on the GPU: no copy
y = x @ x                                  # queued; returns at once
if dev.type == "cuda":
    torch.cuda.synchronize()               # wait for the GPU before reading a timer
y_cpu = y.cpu()                            # the slow part: a copy back over the bus`,
  4: `F = torch.einsum("abi,b->ai", K, m)    # F_ai = K_abi m_b: a batched GEMM underneath
torch.finfo(torch.float32).eps          # 1.2e-07, about 7 digits
torch.finfo(torch.float64).eps          # 2.2e-16, about 16 digits`,
  5: `x = torch.tensor([1.0, 0.5, 0.0], requires_grad=True)
phi = -G * M / x.norm()                 # forward: each operation is recorded in the graph
(grad,) = torch.autograd.grad(phi, x)   # backward: the chain rule, node by node
force = -grad                           # F = -∇φ, no derivative written by hand

fast_step = torch.compile(rk4_step)     # fuse the element-wise kernels of one step`,
  7: `def device() -> torch.device:          # problems/_common/runtime.py
    choice = os.environ.get("TENSORIZER_DEVICE", "auto")   # set by the Backend panel
    if choice == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if choice in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")`,
};

// Same ideas, different frameworks: API names only (not translated)
const FRAMEWORKS = ["PyTorch", "JAX", "MLX", "TensorFlow", "NumPy"];
const ROWS: [string, string[]][] = [
  ["array", ["torch.tensor", "jnp.array", "mx.array", "tf.constant", "np.array"]],
  ["device", ['x.to("cuda" | "mps")', "jax.device_put", "unified memory", 'with tf.device("/GPU:0")', "CPU only"]],
  ["contraction", ["torch.einsum", "jnp.einsum", "mx.einsum", "tf.einsum", "np.einsum"]],
  ["gradient", ["torch.autograd.grad", "jax.grad", "mx.grad", "tf.GradientTape", "—"]],
  ["compile / fuse", ["torch.compile", "jax.jit", "mx.compile", "tf.function", "—"]],
  ["execution", ["eager", "traced (jit) or eager", "lazy", "eager or graph", "eager"]],
];

/** The Concept page's second tab: how tensor code actually runs (memory, kernels, GPUs, graphs, frameworks). */
export default function DataflowConcept() {
  const T = usePageText();
  const R = (key: PageKey) => <RichText text={T(key)} />;
  const has = (key: string) => Boolean(T(key as PageKey)); // sections 6 and 7 have one paragraph

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-5 pb-10">
      <PageToc
        scrollRoot="concept-scroll"
        items={SECTIONS.map((n) => ({ id: `flow-${n}`, mark: n, label: T(`flow.s${n}.title` as PageKey) }))}
      />
      <article className="min-w-0 max-w-4xl flex-1 space-y-10">
        <header>
          <p className="text-sm font-medium text-blue-700">{T("flow.kicker")}</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">{T("flow.title")}</h1>
          <p className="mt-3 text-gray-600">{R("flow.intro")}</p>
          <p className="mt-3 rounded-lg border border-blue-100 bg-blue-50/60 px-4 py-2.5 text-sm text-blue-900">
            {R("flow.name")}
          </p>
        </header>

        {SECTIONS.map((n) => (
          <section key={n} id={`flow-${n}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6">
            <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
                {n}
              </span>
              {T(`flow.s${n}.title` as PageKey)}
            </h2>
            <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">
              <p>{R(`flow.s${n}.p1` as PageKey)}</p>
              {n === 1 && <StridesDiagram />}
              {has(`flow.s${n}.p2`) && <p>{R(`flow.s${n}.p2` as PageKey)}</p>}
              {n === 6 ? <FrameworkTable concept={T("flow.table.concept")} /> : CODE[n] && <CodeBlock code={CODE[n]} />}
            </div>
          </section>
        ))}
      </article>
    </div>
  );
}

function FrameworkTable({ concept }: { concept: string }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200" data-testid="framework-table">
      <table className="w-full text-left text-xs">
        <thead className="bg-gray-50 text-gray-500">
          <tr>
            <th className="px-3 py-2 font-medium">{concept}</th>
            {FRAMEWORKS.map((f) => (
              <th key={f} className="px-3 py-2 font-medium">
                {f}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {ROWS.map(([name, cells]) => (
            <tr key={name}>
              <td className="px-3 py-2 font-medium text-gray-700">{name}</td>
              {cells.map((c, i) => (
                <td key={i} className="px-3 py-2 font-mono text-[11px] text-gray-800">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A (2, 3) tensor, its flat buffer, and the strides that map one to the other; the transpose only swaps strides. */
function StridesDiagram() {
  const cell = (x: number, y: number, v: number, fill = "#dbeafe") => (
    <g key={`${x}-${y}-${v}`}>
      <rect x={x} y={y} width={30} height={26} rx={3} fill={fill} stroke="#3b82f6" />
      <text x={x + 15} y={y + 17.5} textAnchor="middle" fontSize="12" fill="#1e3a8a">
        {v}
      </text>
    </g>
  );
  return (
    <figure className="flex justify-center rounded-lg bg-gray-50 py-3" data-testid="strides-diagram">
      <svg viewBox="0 0 520 170" className="h-44 w-full max-w-2xl" role="img" aria-label="shape (2, 3), strides (3, 1)">
        <text x="20" y="18" fontSize="12" fill="#374151" fontWeight="600">
          shape (2, 3)
        </text>
        {[0, 1].map((i) => [0, 1, 2].map((j) => cell(20 + 34 * j, 28 + 30 * i, 3 * i + j)))}
        <text x="20" y="108" fontSize="11" fill="#6b7280">
          x[i, j] → offset 3i + j
        </text>
        <text x="190" y="18" fontSize="12" fill="#374151" fontWeight="600">
          memory: one flat buffer
        </text>
        {[0, 1, 2, 3, 4, 5].map((k) => cell(190 + 34 * k, 48, k, k < 3 ? "#dbeafe" : "#e0e7ff"))}
        <path d="M205 80 q51 22 102 0" fill="none" stroke="#f59e0b" strokeWidth="1.6" markerEnd="url(#arrow)" />
        <text x="232" y="112" fontSize="11" fill="#b45309">
          next row: +3
        </text>
        <path d="M205 46 q17 -12 34 0" fill="none" stroke="#10b981" strokeWidth="1.6" markerEnd="url(#arrow)" />
        <text x="246" y="38" fontSize="11" fill="#047857">
          next column: +1
        </text>
        <text x="190" y="138" fontSize="12" fill="#374151">
          strides (3, 1)
        </text>
        <text x="190" y="158" fontSize="11" fill="#6b7280">
          x.T: the same buffer with strides (1, 3), no copy
        </text>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0 0 L10 5 L0 10 z" fill="#6b7280" />
          </marker>
        </defs>
      </svg>
    </figure>
  );
}
