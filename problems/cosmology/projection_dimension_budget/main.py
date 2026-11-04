"""Main loop: (n, R) grid -> Gauss's law and ADD -> cited bounds -> allowed map -> dimension budget."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator().manual_seed(1998)
    n_max, R_points = int(params["n_max"]), int(params["R_points"])
    M_star, qubits = float(params["M_star"]), int(params["qubits"])
    n, R = tensors.grid(n_max, R_points, dev)
    rows = tensors.bounds()

    # 1. gravity in 3 + n dimensions: Gauss's law and the return of Newton's law beyond R
    flux = solve.gauss_check(min(n_max, 6), gen)
    print(f"Gauss  flux / |S^(n+2)| for n = 0..{min(n_max, 6)} at r = 0.5, 1, 2: max deviation "
          f"{(flux - 1).abs().max():.1e}")  # fmt: skip
    fc = solve.force_curves(dev)
    for k, c in fc["curves"].items():
        print(f"       n = {k}: r^2 g / G_4 M = {c[0]:.3g} at r = 0.01 R (expect ~(R/r)^n x const), "
              f"{c[-1]:.6f} at r = 10 R (Newton: 1)")  # fmt: skip

    # 2. ADD relation, checked on the PDG numbers
    n2 = torch.tensor(2.0, dtype=torch.float64)
    print(f"ADD    M_Pl^2 = 8 pi R^n M_D^(n+2): R = 30 um, n = 2 -> M_D = {equation.add_md_tev(n2, 30e-6):.2f} TeV "
          f"(PDG: 4.0)")  # fmt: skip
    nn = n[1:]
    R_star = equation.add_radius_m(nn, torch.tensor(M_star))
    print(f"       M_D = {M_star:g} TeV needs R = " + ", ".join(f"{r:.2g} m (n={int(k)})" for k, r in zip(nn, R_star)))

    # 3. the bounds, one family at a time
    em = solve.exclusion_map(n, R, rows)
    rb = solve.radius_budget(n, rows)
    print("Bounds (cited): " + "; ".join(f"{r['source']} {r['year']}: {r['kind']} {r['value']:g}"
                                         + (f" (n={int(r['n'])})" if r["n"] else "") for r in rows))  # fmt: skip
    print("   n   R_max (m)    M_D at R_max (TeV)   set by")
    for i, k in enumerate(n.tolist()):
        rm = rb["rmax"][i].item()
        if k == 0:
            print("   0   -            -                    no extra dimension: nothing to bound")
            continue
        by = [r["bound"] for r in rows if abs(equation._limit(n[i : i + 1], r)[0].item() - rm) <= 1e-9 * rm]
        print(f"  {int(k):2d}   {rm:.3e}    {rb['md_at_rmax'][i]:.3g}" + " " * 14 + ", ".join(sorted(set(by))))
    tested = [(int(k), bool(r > rm)) for k, r, rm in zip(nn, R_star, rb["rmax"][1:]) if k <= 6]
    print(f"       ADD with M_D = {M_star:g} TeV, n = 1..6: excluded for n = {[k for k, x in tested if x] or 'none'}, "
          f"allowed for n = {[k for k, x in tested if not x] or 'none'}; n >= 7: only the torsion bound is bundled, "
          "so this map does not test it (collider limits would, if extended)")  # fmt: skip
    print(f"       allowed fraction of the grid (log R from the Planck length to 1 m): "
          f"{(~em['excluded'][1:]).double().mean():.3f} for n >= 1")  # fmt: skip

    # 4. the quantum side (step 1) and the verdict
    qb = solve.quantum_budget(3, 2, qubits, dev)
    print(f"Quantum  Born race: d + K = {qb['race']} (d = 3, K = 2); {qubits} entangled qubits: Hilbert D >= "
          f"{qb['hilbert_q']}; if spatial directions: n >= {qb['spatial_q']} extra real dimensions")  # fmt: skip
    print("Budget   minimum n = 0 (Hilbert-space reading: hidden dimensions need not be spatial); maximum n: none "
          "set by data, every n is allowed if R < R_max(n)")  # fmt: skip
    print("Verdict  constrained, not ruled out: compact extra dimensions must be small; no new prediction yet")

    return plot.animate(n, R, em, rb, fc, qb, R_star, M_star, out_dir / "projection_dimension_budget.gif")


if __name__ == "__main__":
    cli(run, __file__)
