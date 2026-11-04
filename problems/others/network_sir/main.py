"""Main loop: real contact network → batched stochastic SIR + mean field → threshold sweep → vaccination → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

GAMMA, DT = 1.0, 0.1  # recovery rate (time unit = the mean infectious period) and time step


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    gen = torch.Generator().manual_seed(0)
    runs = 20 if fast else params["n_runs"]
    n_steps = 120 if fast else 400

    A = tensors.load_workplace(dev)
    n = A.shape[0]
    deg = A.sum(1)
    lam, lam_nb = equation.lambda_max(A).item(), equation.lambda_nonbacktracking(A).item()
    print(f"workplace network: N = {n} people, {int(deg.sum()) // 2} contact pairs, mean degree {deg.mean():.1f}, "
          f"max {int(deg.max())}")  # fmt: skip
    print(f"λ_max(A) = {lam:.2f} → mean-field threshold β/γ = 1/λ_max = {1 / lam:.3f}")
    print(f"non-backtracking λ_NB = {lam_nb:.2f} → message-passing SIR threshold β/γ = {1 / (lam_nb - 1):.3f}")

    # one epidemic, B realisations at once; realisation 0 starts at the biggest hub and is drawn
    ratio = params["beta_x_lambda"] / lam  # β/γ
    beta = torch.full((runs,), ratio * GAMMA, device=dev)
    first = torch.randint(n, (runs,), generator=gen).to(dev)
    first[0] = int(deg.argmax())
    infected0 = torch.zeros(runs, n, dtype=torch.bool, device=dev)
    infected0[torch.arange(runs), first] = True
    sim = solve.simulate(A, beta, GAMMA, DT, n_steps, infected0, None, gen, record=True)
    x0 = torch.zeros(1, n, dtype=torch.float64, device=dev)
    x0[0, first[0]] = 1.0
    mf = solve.mean_field(A, beta[:1], GAMMA, x0, DT, n_steps)[:, :, 0].sum(-1)  # (T, 3) expected S, I, R
    conserved = bool((sim["counts"].sum(-1) == n).all())
    final = sim["ever"][-1].float() / n
    print(f"β/γ = {ratio:.3f} ({params['beta_x_lambda']:.1f} × the mean-field threshold), {runs} realisations: "
          f"S + I + R = N at every step: {conserved}")  # fmt: skip
    print(f"  fraction ever infected: mean {final.mean():.2f}, realisations with a major outbreak (> 20 %) "
          f"{(final > 0.2).float().mean():.0%}; mean field {mf[-1, 2] / n:.2f}")  # fmt: skip

    # the threshold: final size against β/γ (all ratios × runs in one batch)
    ratios = torch.linspace(0, 4, 9 if fast else 25, dtype=torch.float64) / lam
    sweep_sim, sweep_mf = solve.threshold_sweep(A, ratios.to(dev), GAMMA, DT, n_steps, 10 if fast else 80, gen)
    for k in range(0, len(ratios), max(1, len(ratios) // 6)):
        print(f"  β/γ · λ_max = {ratios[k] * lam:4.2f}: simulated final size {sweep_sim[k]:.3f}, mean field "
              f"{sweep_mf[k]:.3f}")  # fmt: skip

    # vaccination on synthetic networks with the same number of people and links
    n_syn = 200 if fast else params["n_synthetic"]
    ba = tensors.barabasi_albert(n_syn, 3, gen, dev)
    er = tensors.erdos_renyi(n_syn, ba.sum(1).mean().item(), gen, dev)
    fractions = torch.linspace(0, 0.4, 5 if fast else 9)
    vacc = {}
    for name, G in (("Barabási–Albert", ba), ("Erdős–Rényi", er)):
        vacc[name] = solve.vaccination_sweep(G, fractions, params["vacc_beta"], GAMMA, DT, n_steps,
                                             8 if fast else 30, gen)  # fmt: skip
        lam_g = equation.lambda_max(G).item()
        t, r = vacc[name]["targeted"], vacc[name]["random"]
        print(f"{name}: mean degree {G.sum(1).mean():.1f}, max {int(G.sum(1).max())}, λ_max = {lam_g:.1f}; "
              f"vaccinate {fractions[2]:.0%}: hubs → {t[0][2]:.2f} infected, random → {r[0][2]:.2f}")  # fmt: skip
    pos = tensors.layout(A)
    return plot.animate(A, pos, sim, mf, lam, lam_nb, ratio, ratios, sweep_sim, sweep_mf, fractions, vacc,
                        params["vacc_beta"], DT, out_dir / "network_sir.gif")  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
