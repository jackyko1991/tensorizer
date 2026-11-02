---
name: tensorizer
description: Write, solve and check physics (and maths) problems the Tensorizer way. The law goes vector form → index (tensor) form → batched PyTorch, with the physics tests written first (TDD). Use when creating, editing, explaining or debugging a simulation, turning an equation into PyTorch tensors, or working in a Tensorizer problem folder (meta.yaml, test.py, main.py, equation.py, tensors.py, solve.py, plot.py).
license: Apache-2.0
---

# The Tensorizer method

Tensorizer (https://github.com/jackyko1991/tensorizer) teaches one idea: write a physical law in **index notation**,
and the index notation *is* the PyTorch code. Every problem follows the same **1 + 7 steps**. Step 0 comes first,
and each step has its own file.

| Step | What | File |
|---|---|---|
| 0 | **Tests first**: the physical laws any correct answer must satisfy | `test.py` |
| 1 | Physical picture: what moves, what you want to know | `diagram.svg`, `meta.yaml` (`summary`) |
| 2 | The law in **vector form** | `meta.yaml` (`vector_form`) |
| 3 | The law in **index (tensor) form**, with batch indices | `meta.yaml` (`index_form`, `translation`) |
| 4 | Turn the quantities into tensors, with shapes | `tensors.py` |
| 5 | Write the law as a function of tensors | `equation.py` |
| 6 | Solve: time stepping, `linalg.solve`, `eigh`, FFT, optimisation | `solve.py` |
| 7 | Plot and check the result | `plot.py`, run by `main.py` |

## Step 0: write `test.py` first (TDD, physically constrained)

Before any solver code, write the tests. Each test is a **physical law**, never an implementation detail:

- **Conservation**: energy, momentum, angular momentum, mass, charge, probability or normalisation.
- **A limiting case against a known formula**: small angle → $2\pi\sqrt{L/g}$, high energy → $1+\cos^2\theta$,
  the classical limit.
- **An exact solution** (Kepler's third law, a closed form) or **a symmetry/invariant** (time reversal, rotation,
  Liouville volume, detailed balance).
- **Bounds**: $0 \le p \le 1$, $T > 0$, efficiency ≤ Carnot.

For maths, computer-science or graphics problems, use the mathematical equivalent: orthogonality, reconstruction
error, decoded = input, a known closed form, an energy bound.

```python
"""Physics checks, written first (TDD): what any correct <topic> code must satisfy."""

import math
import pytest
import torch
from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_energy_is_conserved():
    y0 = tensors.initial_state(tensors.amplitudes(4, 160.0), CPU)
    _, traj = solve.solve(y0, 9.81, 1.0, t_max=6.0)
    e = equation.energy(traj, 9.81, 1.0)  # (T, B)
    assert torch.allclose(e, e[0].expand_as(e), rtol=1e-4)  # no friction: total energy is constant


@pytest.mark.slow  # long training or sampling runs: CI only
def test_converges_on_a_fine_grid(): ...
```

Rules:
- Use relative imports (`from . import …`), small grids and short times. A file should run in a few seconds.
- End key asserts with a short comment naming the law.
- **Never weaken a test to make it pass.** A red test means the physics is wrong, or the test's assumptions are.

## Steps 2–3: vector form → index form

Write the law once in vector form, then give every object an index, and add a **batch index** (`b` for many
initial conditions, `a` for particles…) so one call solves many cases at once. Indices that appear twice are
summed (Einstein convention). Indices that appear once are free, and each free index is a tensor axis.

Fill the **translation table** in `meta.yaml`: one row per symbol, with `{symbol (LaTeX), code, shape, meaning}`.
Shapes use the index letters, for example `(B,)`, `(T, B, 2)` or `(N, N, 3)`.

## Steps 4–6: tensors, equation, solver

- **One line of index form → one line of PyTorch.** Use `torch.einsum` for contractions and broadcasting for free
  indices. Never loop in Python over particles, pixels or batch members.
- **Shapes in comments** on every non-trivial line: `r = x[:, None, :] - x[None, :, :]  # (N, N, 3) separations`.
- Build tensors on `device()` from `problems._common.runtime`. It honours `TENSORIZER_DEVICE` (auto, cpu, cuda,
  mps) and falls back to CPU.
- Use `float64` when conservation must hold to many digits. Note that Apple MPS has no float64, so those problems
  run on CPU.
- Time stepping: `problems._common.integrators.rk4_step(f, y, t, dt)` and `integrate(f, y0, dt, n_steps)`, or a
  symplectic scheme (leapfrog or Verlet) for long Hamiltonian runs.
- Keep functions pure (tensors in, tensors out). `solve.py` returns tensors and `plot.py` draws them.

## Step 7: `main.py` and `plot.py`

```python
"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path
from problems._common.runtime import cli, device
from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    y0 = tensors.initial_state(params, dev)  # (B, 2)
    times, traj = solve.solve(y0, params["t_max"])  # (T,), (T, B, 2)
    print(f"energy drift {equation.drift(traj):.2e}")  # print the numbers you check
    return plot.animate(times, traj, out_dir / "result.gif")  # returns the saved GIF/PNG path


if __name__ == "__main__":
    cli(run, __file__)
```

- Plot with matplotlib. Save with `problems._common.plotting.save_gif(fig, update, frame_indices(len(t)), path,
  fps=20)` for animations or `save_png(fig, path)` for still images.
- Show the conserved quantities or errors on the plot, so the picture checks itself.

## `meta.yaml`

```yaml
title: Simple pendulum beyond small angles
category: mechanics            # mechanics, optics, waves, thermodynamics, em, fluids, quantum, astro, nuclear, ...
level: beginner                # beginner | intermediate | advanced
order: 1
summary: One or two sentences; inline math as $...$.
animated: true                 # GIF (true) or PNG (false)
vector_form: '\ddot\theta = -\frac{g}{L}\,\sin\theta'
index_form: '\ddot\theta_b = -\frac{g}{L}\,\sin\theta_b'
translation:
  - {symbol: '\theta_b', code: 'y[:, 0]', shape: '(B,)', meaning: 'angle of pendulum $b$'}
notes: What to notice; the physics behind it.
params:
  length: {default: 1.0, min: 0.2, max: 3.0, step: 0.1, label: 'Rod length $L$', unit: m}
  n: {default: 4, min: 1, max: 6, step: 1, type: int, label: Pendulums}
```

Use single-quoted YAML strings for LaTeX, so backslashes survive.

## Workflow

```bash
pixi run pytest -q problems/<category>/<slug>/test.py                 # step 0: red, then green
pixi run python -m problems.<category>.<slug>.main --out out           # render the GIF/PNG
pixi run python -m problems.<category>.<slug>.main --out out --verbose # every call with tensor shapes and timings
```

In the Tensorizer web app, `test.py` runs after every render and shows a green or red **TDD light**.

## Checklist before you finish

- [ ] `test.py` exists, ran red first, and is now green, and it tests physical laws.
- [ ] The index form in `meta.yaml` matches `equation.py` line by line, and the translation table lists every symbol.
- [ ] No Python loops over batch, particle or grid indices. Shapes are commented.
- [ ] Runs on CPU in under about a minute. The plot shows the check (a conserved quantity or the error).
