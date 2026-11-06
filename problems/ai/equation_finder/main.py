"""Main loop: simulate the four targets, run the finder on every target × noise level × derivative estimator (plus the
annealed-gate alternative), look for an implicit law in the Maxwell library (∇·B = 0), re-simulate the equations found
from clean data, write the tables and the animation."""

import csv
import time
from pathlib import Path

from problems._common.runtime import cli, fast_mode

from . import equation as eq
from . import plot, solve
from . import tensors as T


def run(params: dict, out_dir: Path) -> Path:
    fast = fast_mode()
    t_all = time.perf_counter()
    datasets = {}
    for name in T.TARGETS:
        t0 = time.perf_counter()
        if name == "maxwell":
            datasets[name] = T.maxwell(n=64, steps=120) if fast else T.maxwell(n=params["maxwell_grid"])
        else:
            datasets[name] = T.dataset(name, small=fast)
        ds = datasets[name]
        print(f"{name:8s} data {tuple(ds['U'].shape)} (fields, axes…, time), library of {len(ds['terms'])} terms, "
              f"simulated in {time.perf_counter() - t0:.1f} s")  # fmt: skip

    levels = (0.0, 0.01) if fast else T.NOISE_LEVELS
    methods = ("fd", "weak") if fast else ("fd", "sg", "weak")

    def log(r):
        print(f"{r['target']:8s} {r['method']:9s} noise {r['noise'] * 100:4.1f} %: "
              f"{'right terms' if r['success'] else 'WRONG terms'}, largest coefficient error {r['coef_err']:.1e}, "
              f"{r['seconds']:.1f} s", flush=True)  # fmt: skip
        for line in r["lines"]:
            print("      ", line)

    rows = solve.noise_study(levels, methods, n_boot=10 if fast else params["bootstrap"], gate=not fast,
                             gate_steps=params["gate_steps"], datasets=datasets, log=log)  # fmt: skip

    ds = datasets["maxwell"]  # implicit law: the smallest singular vector of the (clean) Maxwell library
    th, _, _ = eq.strong_library(ds["U"], ds["h"], ds["terms"], "fd")
    vec, sv = solve.null_vector(th)
    vec = vec * vec.sign()[vec.abs().argmax()]
    law = solve.write_equation("0", ds["terms"], vec.where(vec.abs() > 0.05, 0.0), digits=3)
    print(f"implicit law hidden in the Maxwell library (smallest singular vector of the normalised Θ): {law}; "
          f"σ_min / σ_next = {sv[0] / sv[1]:.1e}")  # fmt: skip

    show = {n: next(r for r in rows if r["target"] == n and r["method"] == "fd" and r["noise"] == 0) for n in datasets}
    sims = {}
    for n, r in show.items():
        sims[n] = solve.simulate_found(datasets[n], r["xi"])
        U = datasets[n]["U"]
        rel = float((sims[n][..., -1] - U[..., -1]).norm() / U[..., -1].norm())
        print(f"{n:8s} clean data, finite differences: {plot.constants(n, r['terms'], r['xi']).replace('$', '')}; "
              f"re-simulated from the first snapshot, off the data by {rel:.1e} (relative) at the end")  # fmt: skip

    res = out_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "noise_study.csv", "w", newline="", encoding="utf-8") as f:
        f.write("# equation_finder: one row per (target, regression/derivative method, noise level). noise = std "
                "of the added Gaussian noise / std of the field; coef_error = largest |xi - xi_true| / |xi_true| "
                "over the true terms (1 = a true term was missed)\n")  # fmt: skip
        w = csv.writer(f)
        w.writerow(
            ["target", "method", "noise", "right_terms", "coef_error", "n_terms", "rows", "seconds", "equations"]
        )
        for r in rows:
            w.writerow([r["target"], r["method"], r["noise"], int(r["success"]), f"{r['coef_err']:.3e}",
                        int(r["support"].sum()), r["rows"], f"{r['seconds']:.2f}", " ; ".join(r["lines"])])  # fmt: skip
    with open(res / "equations.tex", "w", encoding="utf-8") as f:
        f.write("% Equations found by equation_finder (STRidge or annealed gates on a library of candidate terms).\n"
                "% fd = finite differences, sg = Savitzky-Golay, weak = weak form; +- = bootstrap standard deviation\n"
                "% over rows (statistical only: no bias of the derivative estimate).\n")  # fmt: skip
        for r in rows:
            f.write(f"% {r['target']}, {r['method']}, noise {r['noise'] * 100:g} %: "
                    f"{'right terms' if r['success'] else 'wrong terms'}\n")  # fmt: skip
            f.write("\\begin{aligned}\n" + " \\\\\n".join(ln.replace(" = ", " &= ", 1) for ln in r["lines"]))
            f.write("\n\\end{aligned}\n\n")
        f.write(
            f"% implicit law from the null space of the Maxwell library (sigma_min/sigma_next = {sv[0] / sv[1]:.1e}):\n"
        )
        f.write(law + "\n")
    print(f"study {time.perf_counter() - t_all:.1f} s; animation next{' (fast mode)' if fast else ''}")
    return plot.animate(datasets, rows, show, sims, out_dir)


if __name__ == "__main__":
    cli(run, __file__)
