"""Main loop: λ grid → train a batch of profiles (the funnels) → pin down λ → blow-up of the found profile → CCF."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve

EXACT = [0.5, 0.25, 1 / 6]  # λ = 1/(2i + 2) of the smooth Burgers profiles, i = 0, 1, 2
CCF_REFERENCE = {  # published exponents for u_t − (Hu) u_x = 0 (same λ convention as here)
    "stable ≈ 1.18 (EF 2020)": 1.18,
    "unstable 0.6057 (Wang 2023)": 0.6057,
    "unstable 0.4713 (Wang 2025)": 0.47132,
}


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    f64 = dict(dtype=torch.float64, device=dev)
    n_lam = 13 if fast_mode() else params["n_lambda"]
    iters = 8 if fast_mode() else params["gn_steps"]
    lams = torch.linspace(0.1, 0.7, n_lam, **f64)  # the batch: one candidate profile per trial λ
    spacing = (lams[1] - lams[0]).item()
    timing, t0 = {}, time.perf_counter()

    # 1. Funnels: every λ of the grid trained at once, for the three smoothness levels.
    funnels, paths, found = [], [], []
    for level in range(3):
        path, hist = solve.burgers_funnel_history(lams, level, iters)
        funnels.append(hist)
        paths.append(path)
    print(f"Burgers: {n_lam} profiles (λ = 0.1 … 0.7) × 3 smoothness levels, {iters} Gauss–Newton steps each")
    for level, name in enumerate(["no constraint", "U‴(0) = 0", "U‴(0) = U⁽⁵⁾(0) = 0"]):
        res = funnels[level][-1]
        minima = solve.local_minima(res)
        start = lams[minima[0]] if len(minima) else lams[res.argmin()]
        out = solve.burgers_search(
            start.reshape(1), level, width=1.5 * spacing, n_golden=6 if fast_mode() else 20, n_fit=16
        )
        lam = out["lam"][0].item()
        i = max(0, round(1 / (2 * lam)) - 1)
        y = torch.linspace(-5, 5, 401, **f64)
        U, _ = equation.burgers_profile(out["coef"][0], out["lam"][0], y)
        sup = (U - equation.burgers_exact(y, i)[0]).abs().max().item()
        found.append({"lam": lam, "coef": out["coef"][0], "res": out["res"][0].item(), "hist": out["lam_hist"][:, 0]})
        print(
            f"  level {level} ({name:<20}): deepest funnel at λ ≈ {start.item():.3f};  search → λ = {lam:.6f} "
            f"(exact family member 1/{2 * i + 2} = {1 / (2 * i + 2):.6f}, error {abs(lam - 1 / (2 * i + 2)):.1e}), "
            f"‖R‖ = {out['res'][0]:.1e}, sup|U − U_exact| on |y| ≤ 5 = {sup:.1e}"
        )
    timing["Burgers"] = time.perf_counter() - t0
    j = int((lams - 0.37).abs().argmin())
    k = int((lams - 0.5).abs().argmin())
    print(
        f"  funnel contrast: ‖R‖(λ = {lams[j]:.2f}) / ‖R‖(λ = {lams[k]:.2f}) = "
        f"{funnels[0][-1][j] / funnels[0][-1][k]:.1e} (smooth profiles exist only at λ = 1/(2i+2))"
    )

    # 2. The blow-up itself: u(x, t) = τ^λ U(x / τ^{1+λ}) with the found stable profile.
    lam0, c0 = torch.tensor(found[0]["lam"], **f64), found[0]["coef"]
    taus = torch.logspace(0, -3, 12 if fast_mode() else 30, **f64)  # τ = T − t
    x = torch.linspace(-1, 1, 801, **f64)
    u = torch.stack([tau**lam0 * equation.burgers_profile(c0, lam0, x / tau ** (1 + lam0))[0] for tau in taus])
    slope = equation.burgers_profile(c0, lam0, torch.zeros(1, **f64))[1].abs() * taus**-1.0  # |U′(0)| / τ
    print(
        f"blow-up: max|u_x| · (T − t) = {(slope * taus).min():.6f} … {(slope * taus).max():.6f} for T − t = 1 … 1e-3 "
        "(the gradient diverges like 1/(T − t); u itself stays bounded, |u| ≤ τ^λ max|U| on the window)"
    )

    # 3. CCF: funnel and search.
    clams = torch.linspace(0.45, 1.6, 7 if fast_mode() else 24, **f64)
    _, ccf_hist = solve.ccf_funnel_history(clams, 6 if fast_mode() else 30)
    t0 = time.perf_counter()
    ccf = solve.ccf_search(
        torch.tensor([params["ccf_lambda0"]], **f64),
        n_golden=4 if fast_mode() else 14,
        n_fit=8 if fast_mode() else 30,
        polish=2 if fast_mode() else 4,
        warm=10 if fast_mode() else 60,
    )
    timing["CCF"] = time.perf_counter() - t0
    lam_c = ccf["lam"][0].item()
    print(
        f"CCF (θ_t − (Hθ)θ_x = 0): search from λ0 = {params['ccf_lambda0']:.2f} → λ = {lam_c:.5f}, "
        f"‖R‖ = {ccf['res'][0]:.1e} ({solve.CCF_COL} collocation points, {ccf['coef'].shape[1]} basis functions)"
    )
    print(
        "  published: stable λ ≈ 1.18 (Eggers & Fontelos 2020, similarity exponent of the ground state); unstable "
        "λ₁ ≈ 0.6057 and λ₂ ≈ 0.4713 (Wang et al. 2023, 2025). This reduced CPU search resolves only the stable "
        "branch: the unstable profiles have sharp gradients that need far more resolution."
    )
    print("compute time: " + ", ".join(f"{k} {v:.1f} s" for k, v in timing.items()))
    return plot.animate(
        lams,
        funnels,
        paths,
        found,
        EXACT,
        taus,
        x,
        u,
        slope,
        clams,
        ccf_hist[-1],
        ccf,
        CCF_REFERENCE,
        out_dir / "blowup_profile_search.gif",
    )


if __name__ == "__main__":
    cli(run, __file__)
