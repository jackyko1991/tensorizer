import CodeBlock from "../components/CodeBlock";
import PageToc from "../components/PageToc";
import { RichText } from "../components/Tex";
import TensorLab, { PairMemory } from "../components/TensorLab";
import { usePageText, type PageKey } from "../content/pages";

// "mem" (the cost of the output) sits after vectorisation; sections are numbered by position
const SECTIONS = ["s1", "s2", "mem", "s3", "s4", "s5", "s6", "s7"] as const;
type Section = (typeof SECTIONS)[number];

const CODE: Partial<Record<Section, string>> = {
  s1: `x = torch.arange(6.0).reshape(2, 3)   # shape (2, 3), one buffer of 6 numbers
x.stride()            # (3, 1): +3 to the next row, +1 to the next column
x.T.stride()          # (1, 3): the same memory, strides swapped, nothing copied
x.T.is_contiguous()   # False: reading it jumps through memory
y = x.T.contiguous()  # the real copy, laid out row by row again
x[:, ::2].stride()    # (3, 2): a slice can change the strides, still no copy
x[[0, 1, 1]]          # advanced indexing: a new tensor (copied)
x.dtype, x.device     # torch.float32, device(type='cpu')`,
  mem: `D = X[None, :, :] - X[:, None, :]     # (N, N, 3): 12·N² bytes in float32

# in chunks: never more than chunk·N·3 numbers alive at once
for a0 in range(0, N, chunk):
    Dc = X[None, :, :] - X[a0:a0 + chunk, None, :]   # (chunk, N, 3)
    F[a0:a0 + chunk] = force(Dc).sum(dim=1)          # reduce before the next chunk`,
  s2: `# a Python loop: one interpreter round trip per pair
for a in range(N):
    for b in range(N):
        r[a, b] = x[b] - x[a]

# one kernel for all N² pairs: broadcasting reuses x with stride 0
r = x[None, :, :] - x[:, None, :]          # (N, N, 3)
x[:, None, :].expand(N, N, 3).stride()     # (3, 0, 1): the new axis costs no memory`,
  s3: `dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(4096, 4096, device=dev)    # created on the GPU: no copy
y = x @ x                                  # queued; returns at once
if dev.type == "cuda":
    torch.cuda.synchronize()               # wait for the GPU before reading a timer
y_cpu = y.cpu()                            # the slow part: a copy back over the bus`,
  s4: `F = torch.einsum("abi,b->ai", K, m)    # F_ai = K_abi m_b: a batched GEMM underneath
torch.finfo(torch.float32).eps          # 1.2e-07, about 7 digits
torch.finfo(torch.float64).eps          # 2.2e-16, about 16 digits`,
  s5: `x = torch.tensor([1.0, 0.5, 0.0], requires_grad=True)
phi = -G * M / x.norm()                 # potential per unit mass; each operation is recorded
(grad,) = torch.autograd.grad(phi, x)   # backward: the chain rule, node by node
acc = -grad                             # -∇φ is an acceleration (GM/r² towards the mass), not a force

U = -G * M * m / x.norm()               # potential energy of a mass m
force = -torch.autograd.grad(U, x)[0]   # F = -∇U = m · acc

fast_step = torch.compile(rk4_step)     # fuse the element-wise kernels of one step`,
  s7: `def device() -> torch.device:          # problems/_common/runtime.py
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
        items={SECTIONS.map((key, i) => ({ id: `flow-${key}`, mark: i + 1, label: T(`flow.${key}.title` as PageKey) }))}
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

        {SECTIONS.map((key, i) => (
          <section key={key} id={`flow-${key}`} className="scroll-mt-6 rounded-xl border border-gray-200 bg-white p-6">
            <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">
                {i + 1}
              </span>
              {T(`flow.${key}.title` as PageKey)}
            </h2>
            <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">
              {[1, 2, 3, 4].map(
                (k) =>
                  has(`flow.${key}.p${k}`) && (
                    <div key={k} className="space-y-4">
                      <p className={key === "s5" && k === 4 ? "rounded-lg bg-amber-50 px-3 py-2 text-amber-900" : ""}>
                        {R(`flow.${key}.p${k}` as PageKey)}
                      </p>
                      {key === "s1" && k === 2 && <TensorLab />}
                      {key === "mem" && k === 1 && <PairMemory />}
                    </div>
                  ),
              )}
              {key === "s6" ? (
                <FrameworkTable concept={T("flow.table.concept")} />
              ) : (
                CODE[key] && <CodeBlock code={CODE[key]!} />
              )}
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
