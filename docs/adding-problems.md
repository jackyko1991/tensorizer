# Adding a problem

Every problem is a self-contained package that the site shows as: title + tags + setup sketch, vector form, index (tensor) form, a translation table, five syntax-highlighted code files, parameters, a terminal and an animated result. **The code shown on the page is exactly the code that runs.**

Start by reading the reference problem `problems/mechanics/projectile/` and copying its style.

## Files

```
problems/<category>/<slug>/
  __init__.py     empty
  meta.yaml       metadata, LaTeX, translation table, parameters (see below)
  main.py         main loop: run(params, out_dir) -> Path   +   if __name__ == "__main__": cli(run, __file__)
  equation.py     physics → tensor translation: the law as small functions, one-to-one with the index form
  tensors.py      builds the concrete PyTorch tensors (grids, operators, initial states), shapes in comments
  solve.py        the solver (time stepping, linalg.solve, eigh, FFT, …)
  plot.py         matplotlib 2D/3D → GIF via save_gif (or PNG via save_png for static problems)
  diagram.svg     hand-drawn sketch of the physical setup
```

- Import shared helpers absolutely (`from problems._common.runtime import cli, device`) and the problem's own files relatively (`from . import solve`).
- `problems/_common/` provides: `runtime.device()`, `integrators.integrate` / `rk4_step` (RK4 on tensors of any shape), `plotting.frame_indices` / `save_gif` / `save_png` (white theme, ≤ 120 frames, atomic writes), and `progress.steps(n, "label")`. Wrap any long time-stepping loop in `steps()`, which turns into a tqdm progress bar when the user ticks **Verbose**.
- Write `equation.py` with `torch.einsum` wherever the index form has a summed index, and say in the docstring which indices are summed. The page is for students: keep the code short and readable, comment shapes and indices, and avoid clever tricks. Beginner problems should be especially gentle.
- Default parameters should render in ≲ 30 s on a CPU, with a GIF ≲ 2 MB (fewer frames via `frame_indices(n, 90)`, a smaller figure around 7×4 in at dpi 80, a discrete colormap for heatmaps). Use a fixed seed for stochastic problems. Code must stay device-agnostic.

## meta.yaml

```yaml
title: Projectile with air drag
category: mechanics            # folder name
level: beginner                # beginner | intermediate | advanced
order: 1                       # position within the level (1–3)
summary: >
  One or two sentences. Inline math goes between dollar signs: $2\times 2$ matrices.
animated: true                 # true → GIF, false → PNG
vector_form: 'm\,\ddot{\mathbf r} = m\,\mathbf g - c\,\lVert\dot{\mathbf r}\rVert\,\dot{\mathbf r}'
index_form: '\ddot x_{bi} = g_i - k\,\sqrt{v_{bj}v_{bj}}\;v_{bi}'   # separate relations with \qquad
translation:
  - {symbol: 'v_{bj}v_{bj}', code: "einsum('bj,bj->b', v, v)", shape: '(B,)', meaning: 'repeated $j$ ⇒ summed'}
notes: >
  Extra explanation, simplifications, what to try. Inline math with $...$.
params:
  v0: {default: 30.0, min: 5, max: 80, step: 1, label: 'Launch speed $v_0$', unit: m/s}
  n:  {default: 6, min: 1, max: 12, step: 1, type: int, label: Count}
  mode: {default: loop, type: choice, options: [loop, helmholtz]}
```

- `vector_form`, `index_form` and `symbol` are pure KaTeX LaTeX. `title`, `summary`, `notes`, `meaning` and `label` are prose with inline `$…$` math (titles render it in the library and heading; keep title math short).
- Use single-quoted YAML strings (or `>` block scalars) for anything with backslashes, never double quotes.

## diagram.svg

Copy the style of `problems/mechanics/simple_pendulum/diagram.svg`:

- `viewBox="0 0 240 160"`, no background, `role="img"` and an `aria-label`.
- Colours:
  - outlines `#374151` at width 2
  - bodies `#2563eb` (stroke `#1e40af`), light fills `#93c5fd`
  - dashed guides `#9ca3af` with `stroke-dasharray="4 3"`
  - forces `#dc2626` with the red arrow marker, angles and highlights `#f59e0b`
- Labels: italic symbols, at most about 5, sans-serif at 13–14 px, still readable at 160 px wide.
- It shows the physical setup, not the result. No scripts, external references or raster images.

## Wikipedia link

Add the problem to `content/wiki.json`: `"<category>/<slug>": {"wikidata": "Q…", "titles": {"en": "…", "fr": "…", …}}`.
Take the titles from the article's Wikidata sitelinks (`en`, `zh`, `fr`, `de`, `es`, `ja`, `ru`, `ko`, `hi`; leave out languages
without an article). The app links to the reader's language when it exists and to English otherwise. The file lives outside
the problem folder, so editing it never forces a re-render.

## Paper links

For a problem built on specific research, link the key papers next to the Wikipedia link:

```bash
pixi run python scripts/papers.py add <category>/<slug> 10.1038/s41586-021-03819-2 2010.08895   # DOIs or arXiv ids
pixi run python scripts/papers.py check   # re-verify every stored link
```

The script looks each one up on Crossref or arXiv and stores the label (first author, year, venue) and the title
from that record in `content/papers.json`, so never type citations by hand.

## Tests (write them first)

Every problem has its own `test.py`, step 0 of the workflow (test-driven development): before the solver, write
what any correct answer must satisfy, as physical laws: conservation (energy, momentum, mass, probability), an
analytic solution or a limiting case, a symmetry or invariant, or bounds. Then write the other files until the tests
pass. Use relative imports (`from . import equation, solve, tensors`) and see
`problems/mechanics/n_body_gravity/test.py` for the style. The web app runs `test.py` after every render and shows
a green or red TDD light; mark checks that take more than a few seconds with `@pytest.mark.slow` (CI runs them, the
after-run check skips them). `tests/test_smoke.py` already checks that every problem follows the convention and
renders.

```bash
pixi run pytest -q problems/<category>/<slug> tests/test_smoke.py -k <slug>
pixi run lint
pixi run python -m problems.<category>.<slug>.main --out out/<slug>      # look at the GIF
pixi run build-assets && pixi run readme                                   # publish to the site and README
```

A new category also needs an entry in `CATEGORIES` in `scripts/build_assets.py` and `scripts/update_readme.py`.
