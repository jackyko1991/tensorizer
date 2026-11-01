"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    L = params["L"]
    T = tensors.temperatures(params["T"], dev)  # (B,)
    s0 = tensors.random_spins(len(T), L, dev)  # (B, 1, L, L)
    masks = tensors.checkerboard(L, dev)

    r = solve.solve(s0, T, masks, params["n_sweeps"])

    exact = equation.onsager_magnetization(T)  # (B,)
    tail = r["m"][-50:].abs().mean(dim=0)  # (B,)  |m| averaged over the last sweeps
    for b in range(len(T)):
        print(f"T = {T[b]:.3f}   |m| = {tail[b]:.3f}  (Onsager {exact[b]:.3f})   E/N = {r['E'][-1, b]:.3f}")
    print(f"T_c = {equation.T_C:.4f}")
    return plot.animate(r["spins"], r["m"], r["E"], T, out_dir / "ising_model.gif")


if __name__ == "__main__":
    cli(run, __file__)
