"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n, b = params["size"], params["b"] * 1e-3  # b in s/mm² → per (10⁻³ mm²/s) units of D
    g = tensors.gradient_directions(params["n_dirs"], dev)  # (G, 3)
    D_true, fibre, label = tensors.phantom(n, dev)  # (V, 3, 3), (V, 3), (V,)
    B = tensors.design_matrix(g)  # (G, 6)

    S0, S = solve.measure(D_true, g, b, params["noise"])
    D_fit, lam, e1, md, fa = solve.fit(S0, S, B, b)

    names = ["grey matter", "arc bundle", "vertical bundle", "crossing", "CSF"]
    print("region            MD (1e-3 mm²/s)   FA     |e1 · fibre|")
    for k, name in enumerate(names):
        sel = label == k
        align = (e1[sel] * fibre[sel]).sum(-1).abs().mean() if k in (1, 2) else float("nan")
        print(f"{name:16s} {md[sel].mean():10.3f}      {fa[sel].mean():6.3f}   {align:8.3f}")
    print(f"max |D_fit - D_true| = {(D_fit - D_true).abs().max():.2e} (noise σ = {params['noise']})")

    lines = solve.track(D_fit.reshape(n, n, 3, 3), tensors.seeds(n, dev))
    print(f"{lines.shape[1]} streamlines, longest {int((~lines[..., 0].isnan()).sum(0).max())} RK2 steps")
    return plot.animate(D_fit, e1, fa, lines, n, out_dir / "diffusion_tensor_imaging.gif")


if __name__ == "__main__":
    cli(run, __file__)
