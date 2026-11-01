"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors

NAMES = ("[A]", "ln[A]", "1/[A]")
SIGN = torch.tensor([-1.0, -1.0, 1.0], dtype=tensors.DTYPE)  # slope = -k, -k, +k for orders 0, 1, 2


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    Ea = params["Ea"] * 1e3  # kJ/mol → J/mol
    A = 10 ** params["log10_A"]
    true_order = int(params["order"])

    T = tensors.temperatures(params["T_min"], params["T_max"], dev)  # (T,) K
    k = equation.rate_constant(T, A, Ea)  # (T,)
    n = tensors.orders(dev)  # (O,)
    t_end = tensors.run_lengths(k, n)  # (O, T) s
    conc0 = tensors.initial_state(len(n), len(T), dev)  # (O, T)

    # 1) integrate all three rate laws at all temperatures at once
    tau, traj = solve.integrate_runs(conc0, k, n, t_end)  # (N+1,), (N+1, O, T)

    # 2) half-life method on the smooth curves: t½ from [A]_0 → [A]_0/2, then from [A]_0/2 → [A]_0/4
    t_50 = solve.crossing_time(tau, traj, t_end, 0.5 * tensors.A0)
    t_25 = solve.crossing_time(tau, traj, t_end, 0.25 * tensors.A0)
    n_half = equation.order_from_half_lives(t_50, t_25 - t_50)  # (O, T)
    print("half-life method on the integrated curves (should give 0, 1, 2):")
    for o in range(len(n)):
        ratio = ((t_25 - t_50) / t_50)[o, 0].item()
        print(f"  rate law n = {o}:  t½(2)/t½(1) = {ratio:.3f}  →  n = {n_half[o].mean().item():.3f}")

    # 3) a noisy "experiment" on the reaction of unknown order: which plot is straight?
    tau_s, clean = solve.sample(tau, traj, true_order)  # (S,), (T, S)
    data = tensors.measure(clean, params["noise"])  # (T, S)
    times = tau_s[None, :] * t_end[true_order][:, None]  # (T, S) real sampling times, s
    beta, R2 = solve.fit_linearizations(times, data)  # (3, T, 2), (3, T)
    best = int(R2.mean(1).argmax())
    print(f"\nmeasured run, noise σ = {params['noise']:.3f} M — mean R² of each straight-line plot:")
    for g in range(3):
        mark = "  ← straightest" if g == best else ""
        print(f"  {NAMES[g]:<6} vs t:  R² = {R2[g].mean().item():.5f}{mark}")
    print(f"  ⇒ reaction order n = {best} (true: {true_order})")

    # 4) k from the slope of the straight plot, then the Arrhenius fit
    k_fit = SIGN[best].item() * beta[best, :, 1]  # (T,)
    Ea_fit, A_fit = equation.arrhenius_fit(T, k_fit.clamp_min(1e-300))
    print("\n   T (K)    k true       k fit        run length")
    for t in range(len(T)):
        run_time = plot.fmt_time(t_end[true_order, t].item())
        print(f"  {T[t].item():6.1f}  {k[t].item():.4e}  {k_fit[t].item():.4e}  {run_time}")
    print(f"\nArrhenius fit:  E_a = {Ea_fit / 1e3:.2f} kJ/mol (true {Ea / 1e3:.2f})   A = {A_fit:.3e} (true {A:.3e})")

    return plot.animate(
        tau, traj[:, true_order], t_end[true_order], T, tau_s, data, beta, R2, best, k, k_fit, Ea_fit, A_fit,
        out_dir / "arrhenius_kinetics.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
