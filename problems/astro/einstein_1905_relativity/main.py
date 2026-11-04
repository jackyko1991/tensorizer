"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    beta_fast, n_clocks, dEdx = params["beta_fast"], params["n_clocks"], params["dEdx"]

    # light clocks: every clock and every instant at once
    beta = tensors.clock_speeds(beta_fast, n_clocks)  # (B,)
    t, x, y = solve.light_clocks(beta, D=1.0, t_max=8.4, n=1681)
    ticks = solve.count_ticks(y)
    for b, k in zip(beta.tolist(), ticks.tolist()):
        g = equation.gamma(torch.tensor(b)).item()
        print(
            f"clock at β = {b:.3f}: γ = {g:.3f}, ticks {k} in t = 8.4 (rest clock: {int(ticks[0])}),"
            f" rod length L/L0 = {solve.contracted_length(1.0, torch.tensor(b)).item():.3f}"
        )
    w = equation.velocity_addition(torch.tensor(0.9), torch.tensor(0.9))
    w_c = equation.velocity_addition(torch.tensor(1.0), torch.tensor(0.9))
    print(f"0.9c ⊕ 0.9c = {w:.4f}c (not 1.8c); c ⊕ 0.9c = {w_c:.1f}c")
    flashes = torch.tensor([[0.0, -1, 0, 0], [0.0, 1, 0, 0]], dtype=torch.float64)  # t = 0 on the platform
    dt = equation.transform(equation.boost(torch.tensor(0.6)), flashes)
    print(f"flashes simultaneous on the platform, 2 light-units apart: Δt′ = {dt[1, 0] - dt[0, 0]:.3f} at 0.6c")

    # E = mc²
    b_grid = torch.linspace(0.01, 0.99, 99, dtype=torch.float64)
    dK = solve.kinetic_energy_drop(1.0, b_grid, torch.zeros_like(b_grid))
    dm = dK / (equation.gamma(b_grid) - 1)
    print(f"1905 mass–energy argument: Δm·c²/L = {dm.min():.12f} … {dm.max():.12f} for β = 0.01 … 0.99")
    names = list(tensors.NUCLIDES)
    Z, N, M = tensors.nuclear_masses(names)
    B = equation.binding_energy(Z, N, M)
    for n, b, a in zip(names, B.tolist(), (Z + N).tolist()):
        print(f"  {n:6s} binding energy {b:9.3f} MeV  ({b / a:.3f} MeV per nucleon)")
    photons = solve.annihilate_at_rest(tensors.M_ELECTRON)
    print(f"e⁺e⁻ at rest → 2 γ of {photons[0, 0]:.5f} MeV each (2 m_e c² = {2 * tensors.M_ELECTRON:.5f} MeV)")

    # Frisch & Smith (1963): muons from Mt Washington to sea level
    data = tensors.frisch_smith()
    mu = solve.muon_gamma(data, dEdx)
    print(
        f"muons: lab flight time {mu['t_lab']:.2f} μs; without dilation {mu['no_dilation_rate']:.0f}/h would arrive,"
        f" measured {data['rate'][1]:.0f} ± {data['err'][1]:.0f}/h"
    )
    print(
        f"  γ from the counts {mu['gamma_fit']:.2f} ± {mu['gamma_err']:.2f} (muon clock: {mu['tau_rest']:.2f} μs);"
        f" γ from the speeds {mu['gamma_pred']:.2f}; published 8.8 ± 0.8 vs 8.4 ± 2"
    )
    print(f"  speed at sea level {mu['beta_bottom'][0]:.4f} … {mu['beta_bottom'][1]:.4f} (published 0.9881 … 0.9897)")
    return plot.animate(beta, t, x, y, (names, B, Z + N), mu, out_dir / "einstein_1905_relativity.gif")


if __name__ == "__main__":
    cli(run, __file__)
