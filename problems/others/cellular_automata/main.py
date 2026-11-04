"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors

FEATURED = {90: "single", 30: "single", 110: "random", 184: "random"}  # Sierpinski, chaos, complexity, traffic


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    w, n_steps = params["width"], params["generations"]
    rules = torch.arange(256, device=dev)
    table = tensors.rule_table(rules)  # (256, 8)

    # every rule from the same random line: one (T+1, 256, W) tensor
    mosaic = solve.run_eca(tensors.random_rows(256, 32, 0.5, 0, dev), table, 23)  # small thumbnails
    kind = solve.behaviour(tensors.random_rows(256, 151, 0.5, 1, dev), table, 600)  # (256,)
    reps = solve.equivalence_classes(rules)
    print(
        f"256 rules advanced together by one gather on the 3-cell code; {reps.unique().numel()} classes under symmetry"
    )
    names = ["uniform (class 1)", "periodic up to a shift (class 2)", "no period found (class 3 or 4)"]
    for c, name in enumerate(names):
        members = (kind == c).nonzero().flatten()
        print(f"{name:34s}: {len(members):3d} rules, {reps[members].unique().numel():2d} classes")
    print(f"rule 30 → {names[kind[30]]}, rule 90 → {names[kind[90]]}, rule 110 → {names[kind[110]]}")

    user = params["rule"]
    featured = {user: params["start"], **{r: s for r, s in FEATURED.items() if r != user}}
    featured = dict(list(featured.items())[:4])
    starts = {
        "single": tensors.single_cell(1, w, dev),
        "random": tensors.random_rows(1, w, params["density"], 2, dev),
    }
    x0 = torch.cat([starts[s] for s in featured.values()])  # (4, W)
    spacetime = solve.run_eca(x0, table[list(featured)], n_steps)  # (T+1, 4, W)
    for (r, s), cars in zip(featured.items(), spacetime.sum(-1).T):
        print(f"rule {r:3d} from a {s} start: live cells {int(cars[0])} → {int(cars[-1])}")

    a0 = tensors.orbium(params["lenia_size"], dev)
    kernel = tensors.lenia_kernel(params["lenia_size"], 13.0, dev)
    frames, mass = solve.run_lenia(a0, kernel, params["lenia_steps"])
    settled = mass[min(200, len(mass) // 2) :]
    print(
        f"Lenia Orbium: mass {mass[0]:.1f} → {mass[-1]:.1f}, "
        f"varying by {(settled.max() - settled.min()) / settled.mean():.2%} once settled"
    )
    shift, overlap = solve.best_shift(frames[len(frames) // 2], frames[-1])
    print(f"over the last {len(frames) // 2} steps the glider moved by {shift} cells, shape overlap {overlap:.3f}")
    return plot.animate(spacetime, list(featured), mosaic, kind, frames, mass, out_dir / "cellular_automata.gif")


if __name__ == "__main__":
    cli(run, __file__)
