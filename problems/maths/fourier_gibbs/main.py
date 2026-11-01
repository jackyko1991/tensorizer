"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import coefficients, energy

GIBBS = 0.0895  # (1/π) Si(π) − 1/2: the Wilbraham–Gibbs constant


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_max = params["n_max"]
    x, w = tensors.grid(params["n_x"], dev)  # (X,), (X,)
    f = tensors.waves(x)  # (3, X)
    b = tensors.basis(x, n_max)  # (X, N)
    print(f"{f.shape[0]} waves on {x.shape[0]} grid points, basis matrix B of shape {tuple(b.shape)}")

    c = coefficients(f, b, w)  # (3, N)
    exact = tensors.exact_coefficients(n_max, dev)
    for k, name in enumerate(tensors.WAVES):
        first = ", ".join(f"{v:+.4f}" for v in c[k, :5].tolist())
        print(f"{name:8s} b_1..b_5 = {first}   max |numerical − exact| = {(c[k] - exact[k]).abs().max():.1e}")
    tail = (energy(f, w) - (c**2).sum(dim=1)).tolist()
    print("Parseval (1/π)∫f² − Σ b_n² (what the first N_max harmonics miss): " + ", ".join(f"{t:.1e}" for t in tail))

    s = solve.all_partial_sums(b, c)  # (N, 3, X)
    over = solve.overshoot(s, f, tensors.JUMP)  # (N, 3)
    err = solve.errors(f, s, w)  # (N, 3)
    for n in (1, 9, n_max // 2, n_max):
        n = n if n % 2 == 1 else n - 1  # odd N: the square wave's even harmonics are 0
        print(f"N = {n:4d}: overshoot / jump  square {over[n - 1, 0]:.4f}  sawtooth {over[n - 1, 1]:.4f}")
    print(f"Gibbs: the overshoot tends to {GIBBS} × jump, not to 0 (only its width shrinks, like 1/N)")
    rate = solve.decay_rate(err)
    print("L² error² ∝ N^(−p):  " + "  ".join(f"{nm} p = {p:.2f}" for nm, p in zip(tensors.WAVES, rate.tolist())))
    return plot.animate(x, f, s, over, err, out_dir / "fourier_gibbs.gif")


if __name__ == "__main__":
    cli(run, __file__)
