import { generatedUrl } from "../api";
import CodeBlock, { extractDef } from "../components/CodeBlock";
import Tex, { RichText } from "../components/Tex";
import type { Manifest } from "../types";

const PENDULUM = "mechanics/simple_pendulum";

/**
 * "How it works", told with one problem: the simple pendulum goes from a physical picture to
 * an equation, to tensors, to code, to numbers and a picture again. Code excerpts are pulled
 * from the problem's real files in the manifest, so the page always matches the library.
 */
export default function ConceptPage({ manifest }: { manifest: Manifest }) {
  const p = manifest.problems.find((x) => x.id === PENDULUM);
  if (!p) return <p className="p-8 text-sm text-gray-500">The pendulum problem is not in the library.</p>;
  const file = (name: string) => p.files.find((f) => f.name === name)?.code ?? "";

  return (
    <div className="h-full overflow-y-auto bg-gray-50/60">
      <article className="mx-auto max-w-4xl space-y-10 px-5 py-10">
        <header>
          <p className="text-sm font-medium text-blue-700">Concept</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">From a physics law to a running simulation</h1>
          <p className="mt-3 text-gray-600">
            Every problem in Tensorizer follows the same seven steps. Here they are for the simplest one: a pendulum
            swinging under gravity. Each step matches a file you can open, run and edit in the Problems tab.
          </p>
          <Pipeline />
        </header>

        <Step n={1} title="Start from the physical picture">
          <div className="grid items-center gap-6 sm:grid-cols-[1fr_auto]">
            <div className="space-y-3">
              <RichText text="A point mass $m$ hangs on a light rod of length $L$. Gravity $g$ pulls it down; the only coordinate that matters is the angle $\theta$ from the vertical." />
              <p>
                Decide what you want to know before writing any equation: <em>how does θ change in time, and does a
                big swing take longer than a small one?</em>
              </p>
            </div>
            {p.diagram && <img src={generatedUrl(p.diagram)} alt="Pendulum setup" className="h-40 w-60 object-contain" />}
          </div>
        </Step>

        <Step n={2} title="Write the law in vector form">
          <RichText text="Newton's second law along the arc of the swing, $m L\ddot\theta = -mg\sin\theta$, gives the equation of motion:" />
          <Eq tex={p.vector_form} />
          <RichText text="It is nonlinear because of $\sin\theta$, so there is no simple formula for $\theta(t)$ — a computer will march it forward in time instead. For small swings $\sin\theta\approx\theta$ and the familiar solution $\theta_0\cos(\sqrt{g/L}\,t)$ comes back." />
        </Step>

        <Step n={3} title="Rewrite it in index (tensor) form">
          <RichText text="Instead of one pendulum, release $B$ of them from different angles and give each an index $b$. The law becomes one statement about a whole tensor of angles:" />
          <Eq tex={p.index_form} />
          <RichText text="Read the indices like a spreadsheet: $\theta_b$ is a column of $B$ numbers. Nothing is summed here — each pendulum obeys the same law independently. In other problems a repeated index is summed (Einstein convention), e.g. $v_j v_j = \lvert v\rvert^2$, and that sum becomes one `torch.einsum` call." />
          <TranslationTable rows={p.translation} />
        </Step>

        <Step n={4} title="Turn the quantities into tensors" file="tensors.py">
          <RichText text="The state of every pendulum is its angle and angular velocity, so the whole system is one tensor $y$ of shape $(B, 2)$. A second-order equation becomes two first-order ones: $\dot\theta_b = \omega_b$ and $\dot\omega_b = -(g/L)\sin\theta_b$." />
          <CodeBlock title="tensors.py" code={extractDef(file("tensors.py"), "initial_state")} />
        </Step>

        <Step n={5} title="Write the law as a function" file="equation.py">
          <RichText text="`equation.py` is the translation step: each line of the index form becomes a line of PyTorch, applied to all pendulums at once. Comments carry the shapes so you can check them like units." />
          <CodeBlock title="equation.py" code={`${extractDef(file("equation.py"), "angular_acceleration")}\n\n\n${extractDef(file("equation.py"), "rhs")}`} />
          <RichText text="The small-angle solution is an outer product — two indices, nothing summed — so it is written with einsum as `t,b->tb`:" />
          <CodeBlock code={extractDef(file("equation.py"), "small_angle")} />
        </Step>

        <Step n={6} title="Solve: march forward in time" file="solve.py">
          <RichText text="Fourth-order Runge–Kutta takes small steps $\Delta t$: it samples the right-hand side four times per step and combines them, with an error that shrinks like $\Delta t^4$. Because the state is a tensor, one RK4 step updates all $B$ pendulums together — on a GPU this is how thousands of simulations run in parallel." />
          <CodeBlock title="solve.py" code={extractDef(file("solve.py"), "solve")} />
        </Step>

        <Step n={7} title="Look at the result — and check it" file="plot.py">
          <div className="grid items-start gap-6 md:grid-cols-2">
            <div className="space-y-3">
              <RichText text="`plot.py` turns the trajectory tensor of shape $(T, B, 2)$ into an animation: each pendulum swings next to its dashed small-angle partner, and the big swings visibly fall behind." />
              <RichText text="Never trust a simulation you have not checked. The tests compare against things we know exactly: energy $\tfrac12L^2\omega^2 + gL(1-\cos\theta)$ must stay constant, small swings must have period $2\pi\sqrt{L/g}$, and large swings must match the elliptic-integral formula." />
            </div>
            {p.output && <img src={generatedUrl(p.output)} alt="Pendulum animation" className="w-full rounded-lg border border-gray-200 bg-white" />}
          </div>
        </Step>

        <section className="rounded-xl border border-blue-200 bg-blue-50 p-6">
          <h2 className="text-lg font-semibold text-blue-900">Your turn</h2>
          <p className="mt-2 text-sm text-blue-900">
            Open the pendulum, change the largest amplitude or gravity, tick <em>Verbose output</em> to watch every
            tensor shape go by, or edit <code className="font-mono">equation.py</code> — try replacing{" "}
            <code className="font-mono">sin(theta)</code> with <code className="font-mono">theta</code> and see the
            pendulums stay in step.
          </p>
          <a
            href={`#/${PENDULUM}`}
            className="mt-4 inline-block rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          >
            Open the pendulum problem →
          </a>
        </section>
      </article>
    </div>
  );
}

function Pipeline() {
  const steps = ["Physical picture", "Vector form", "Index form", "Tensors", "Law as code", "Solver", "Plot & check"];
  return (
    <ol className="mt-6 flex flex-wrap items-center gap-2 text-xs">
      {steps.map((s, i) => (
        <li key={s} className="flex items-center gap-2">
          <span className="rounded-full border border-blue-200 bg-white px-3 py-1 font-medium text-blue-800">
            {i + 1}. {s}
          </span>
          {i < steps.length - 1 && <span className="text-gray-400">→</span>}
        </li>
      ))}
    </ol>
  );
}

function Step({ n, title, file, children }: { n: number; title: string; file?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-gray-200 bg-white p-6">
      <h2 className="flex flex-wrap items-baseline gap-3 text-lg font-semibold">
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-sm text-white">{n}</span>
        {title}
        {file && <code className="rounded bg-gray-100 px-2 py-0.5 font-mono text-xs font-normal text-gray-600">{file}</code>}
      </h2>
      <div className="mt-4 space-y-4 text-sm leading-relaxed text-gray-700">{children}</div>
    </section>
  );
}

function Eq({ tex }: { tex: string }) {
  return (
    <div className="rounded-lg bg-gray-50 py-2 text-lg">
      <Tex tex={tex} display />
    </div>
  );
}

function TranslationTable({ rows }: { rows: { symbol: string; code: string; shape: string; meaning: string }[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left text-xs text-gray-500">
            <th className="py-1.5 pr-3 font-medium">Symbol</th>
            <th className="py-1.5 pr-3 font-medium">Code</th>
            <th className="py-1.5 pr-3 font-medium">Shape</th>
            <th className="py-1.5 font-medium">Meaning</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.symbol + r.code} className="border-b border-gray-100 align-top">
              <td className="py-1.5 pr-3 whitespace-nowrap">
                <Tex tex={r.symbol} />
              </td>
              <td className="py-1.5 pr-3">
                <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-xs">{r.code}</code>
              </td>
              <td className="py-1.5 pr-3 font-mono text-xs whitespace-nowrap text-gray-600">{r.shape}</td>
              <td className="py-1.5">
                <RichText text={r.meaning} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
