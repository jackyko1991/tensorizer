"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    body = tensors.BODIES[params["body"]]
    mu, unit, t_unit, t_label = body["mu"], body["unit_km"], body["t_unit"], body["t_label"]
    r1, r2, rb = tensors.radii_km(params, body)
    kinds = tensors.spacecraft_kinds(params["transfer"])

    # Fly every spacecraft at once; coast for half a target orbit after the last burn.
    y0 = tensors.initial_state(r1, len(kinds), mu, dev)  # (B, 2, 2)
    burn_t, burn_dv = tensors.burn_schedule(kinds, r1, r2, rb, mu, dev)  # (B, 3) each
    T_inner = 2 * torch.pi * (min(r1, r2) ** 3 / mu) ** 0.5  # period of the tighter circle
    t_end = float(burn_t.max()) + 0.5 * 2 * torch.pi * (r2**3 / mu) ** 0.5
    dt = max(T_inner / (60 if fast_mode() else 250), t_end / 60000)  # ≥ 250 steps per inner orbit, ≤ 60k steps
    times, traj, burn_pos = solve.fly(y0, mu, burn_t, burn_dv, t_end, dt)

    print(f"central body: {params['body']}   r1 = {r1:,.0f} km   r2 = {r2:,.0f} km   r2/r1 = {r2 / r1:.3f}")
    r = traj[:, :, 0, :].norm(dim=-1)  # (T, B)
    E = equation.energy(traj, mu)  # (T, B)
    for b, kind in enumerate(kinds):
        dv, tb = burn_dv[b], burn_t[b]
        k_arr = int(torch.searchsorted(times, tb[-1]))  # sample right at the last burn = arrival
        coast = times > tb[-1]
        between = (times > tb[0]) & (times < tb[1])  # first coasting arc (the transfer ellipse)
        drift = (E[between, b] / E[between, b][0] - 1).abs().max()
        spread = (r[coast, b].max() - r[coast, b].min()) / r2  # 0 for a perfectly circular final orbit
        burns = "  ".join(f"Δv{k + 1} = {float(v):+.3f}" for k, v in enumerate(dv) if v != 0) or "no burn needed"
        total, t_arr = float(dv.abs().sum()), float(tb[-1]) / t_unit
        print(f"{kind:>11}: {burns} km/s   total = {total:.3f} km/s   time = {t_arr:.2f} {t_label}")
        print(
            f"{'':>11}  integrated: r(arrival)/r2 = {float(r[k_arr, b]) / r2:.6f}"
            f"   final orbit (r_max - r_min)/r2 = {float(spread):.1e}   ΔE/E on the first ellipse = {float(drift):.1e}"
        )

    R_any, R_every = solve.crossover_any_rb(), solve.crossover_every_rb()
    print(f"bi-elliptic beats Hohmann above r2/r1 ≈ {R_any:.2f} (r_b → ∞) and for every r_b > r2 above ≈ {R_every:.2f}")

    sun = tensors.BODIES["sun"]
    au = torch.tensor(sun["unit_km"], dtype=tensors.DTYPE)
    dv1, dv2, t_em = equation.hohmann(au, tensors.MARS_AU * au, sun["mu"])
    print(
        f"Earth → Mars Hohmann (heliocentric, circular coplanar orbits): Δv1 ≈ {float(dv1):.2f} km/s,"
        f" Δv2 ≈ {float(dv2):.2f} km/s, total ≈ {float(dv1 + dv2):.2f} km/s, flight ≈ {float(t_em) / 86400:.0f} days"
    )

    curves = solve.total_dv_curves(tensors.ratio_grid(), params["rb_ratio"])
    return plot.animate(
        times / t_unit,
        traj / unit,
        burn_t / t_unit,
        burn_dv,
        burn_pos / unit,
        kinds,
        (r1 / unit, r2 / unit, rb / unit),
        params["body"],
        body["unit"],
        t_label,
        tensors.ratio_grid(),
        curves,
        (R_any, R_every),
        params["rb_ratio"],
        out_dir / "orbit_transfer.gif",
    )


if __name__ == "__main__":
    cli(run, __file__)
