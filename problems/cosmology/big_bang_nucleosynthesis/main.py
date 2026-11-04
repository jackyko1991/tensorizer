"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    eta_mark = params["eta10"] * 1e-10
    etas = tensors.eta_batch(params["n_eta"], eta_mark, dev)  # (B,) the whole Schramm plot in one batch
    res = solve.run_bbn(etas, params["tau_n"], params["n_nu"], params["n_steps"], dev)

    print(
        f"{len(etas)} values of η integrated together, {params['n_steps']} implicit BDF2 steps from T = 3 MeV to 1 keV"
    )
    print(f"τ_n = {params['tau_n']} s, N_ν = {params['n_nu']}")
    Y = res["Y"][:, -1]
    for T_show in (1.0, 0.25, 0.08):
        k = int((res["T"] - T_show).abs().argmin())
        print(f"T = {T_show:5.2f} MeV  t = {float(res['t'][k]):7.1f} s   n/p = 1/{float(Y[k, 1] / Y[k, 0]):.2f}")
    print(f"{'η × 10¹⁰':>9} {'Y_p':>8} {'D/H':>10} {'³He/H':>10} {'⁷Li/H':>10}")
    for b in range(len(etas)):
        print(f"{float(etas[b]) * 1e10:9.2f} {float(res['Yp'][b]):8.4f} {float(res['DH'][b]):10.3e}"
              f" {float(res['He3H'][b]):10.3e} {float(res['LiH'][b]):10.3e}")  # fmt: skip
    obs = tensors.OBSERVED
    print(f"at η = {params['eta10']:.2f} × 10⁻¹⁰:")
    print(f"  Y_p   = {float(res['Yp'][-1]):.4f}   observed {obs['Yp'][0]} ± {obs['Yp'][1]} (PDG)")
    print(f"  D/H   = {float(res['DH'][-1]):.3e}  observed (2.527 ± 0.030)e-5 (Cooke et al. 2018)")
    li = float(res["LiH"][-1])
    print(f"  ⁷Li/H = {li:.2e}   observed (1.6 ± 0.3)e-10 (Spite plateau): {li / obs['LiH'][0]:.1f}× too high,"
          " the cosmological lithium problem (unresolved)")  # fmt: skip
    return plot.animate(res, tensors.ETA_PLANCK, out_dir / "big_bang_nucleosynthesis.gif")


if __name__ == "__main__":
    cli(run, __file__)
