"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import gauss_legendre, integral


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    kink, alpha = params["kink"], params["alpha"]
    ns = tensors.n_values(params["n_max"])
    print(f"{len(ns)} values of N from {ns[0]} to {ns[-1]}, 3 rules, 3 integrands: {', '.join(tensors.NAMES)}")

    x, w = gauss_legendre(tensors.jacobi_matrix(5, dev))
    print(f"Gauss–Legendre N = 5 nodes {[round(v, 6) for v in x.tolist()]}, weights sum to {w.sum():.15f}")
    print(f"  ∫x⁸ dx = 2/9 = {2 / 9:.15f}, Gauss with 5 nodes gives {integral(w, x[None] ** 8).item():.15f}")

    rules = solve.all_rules(ns, dev)  # rules[method][i] = (nodes, weights)
    err = solve.errors(rules, kink, alpha)  # (3 methods, M, 3 integrands)
    p = solve.orders(ns, err)
    for m, method in enumerate(solve.METHODS):
        cells = [f"{name} {p[m, f]:.2f}" for f, name in enumerate(tensors.NAMES)]
        if m == 2:  # Gauss on a smooth function has no fixed order: the error falls faster than any power of N
            cells[0] = "smooth exp."
        print(f"order p (error ∝ N^-p)  {method:15s} " + "   ".join(cells))
    rho = solve.exponential_rate(ns, err)
    print(f"Gauss–Legendre on the smooth integrand: error ∝ {rho:.2f}^(−N), exponential (no fixed order)")
    print(f"at N = {ns[-1]}: smallest errors " + ", ".join(f"{e:.1e}" for e in err[:, -1].amin(dim=0).tolist()))
    return plot.animate(ns, rules, err, kink, alpha, out_dir / "numerical_integration.gif")


if __name__ == "__main__":
    cli(run, __file__)
