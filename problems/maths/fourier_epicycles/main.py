"""Main loop: closed curve → DFT → circles sorted by radius → chains with more and more circles → plot."""

from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    name, n = params["shape"], params["n_points"]
    z = tensors.shape(name, n, dev)  # (N,) complex
    m_stages = solve.stages(n, params["max_terms"])
    if fast_mode():
        m_stages = m_stages[:2]
    res = solve.analyse(z, m_stages)
    print(f"{name}: N = {n} samples z_n = x_n + i y_n, frequencies k = {res['k'][0].item()} … {res['k'][-1].item()}")
    print(f"Parseval: (1/N) Σ|z_n|² = {res['power']:.12f},  Σ|c_k|² = {res['coef_power']:.12f}")
    top = res["order"][:6]
    print("largest circles: " + ", ".join(f"k = {k:+d} r = {r:.3f}" for k, r in
          zip(res["k"][top].tolist(), res["c"][top].abs().tolist())))  # fmt: skip
    err = res["err"]
    for m in m_stages:
        e, p = err[m - 1].item(), res["parseval"][m - 1].item()
        print(f"  M = {m:4d} circles: mean squared error {e:.3e}  (Parseval {p:.3e})")
    mono = bool((err[1:] <= err[:-1] + 1e-12).all())
    print(f"error falls monotonically with M: {mono};  all {n} circles: error {err[-1].item():.1e}")
    return plot.animate(z, res, m_stages, name, out_dir / "fourier_epicycles.gif",
                        frames_per_stage=4 if fast_mode() else 12)  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
