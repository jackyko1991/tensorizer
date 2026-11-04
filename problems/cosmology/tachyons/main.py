"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    u0, beta_max = params["u"], params["beta_max"]
    speeds = tensors.signal_speeds(u0, dev)  # (U,)
    betas = tensors.frames(beta_max, 90, dev)  # (F,)

    # 1. imaginary mass: E² = p² + m² with m = iμ (μ = 1 here)
    m = torch.tensor(1j, dtype=torch.complex128, device=dev)
    p = speeds / torch.sqrt(speeds**2 - 1)  # momentum that gives each speed when μ = 1
    e = equation.energy(p, m)
    four = torch.stack([e.real, p], dim=-1)  # (U, 2)
    m2 = equation.minkowski_square(equation.boost(four, betas))  # (F, U)
    print(f"signal speeds u = {speeds.tolist()} c,  energies E = {[round(x, 3) for x in e.real.tolist()]} μ")
    print(f"invariant m² over all {len(betas)} frames: {m2.min().item():.6f} … {m2.max().item():.6f} (= −μ²)")

    # 2. the same signals seen from every frame: arrival before emission when β > 1/u
    events = tensors.emission_and_reception(speeds)  # (U, 2)
    seen = equation.boost(events, betas)  # (F, U, 2)
    energy_seen = equation.boost(four, betas)[..., 0]  # (F, U)
    feinberg = solve.reinterpret(seen[..., 0], energy_seen)
    for u in speeds.tolist():
        print(f"u = {u:5.2f} c: time order flips for β > 1/u = {1 / u:.3f}")
    assert torch.equal(feinberg["emitter_is_receiver"], seen[..., 0] < 0)

    # 3. Tolman's antitelephone with Bob receding at β
    loop = equation.antitelephone(torch.tensor(u0, dtype=torch.float64, device=dev), betas, t_send=1.0)
    threshold = 2 * u0 / (1 + u0**2)  # β above which u > (1 + √(1 − β²))/β
    print(f"antitelephone with u = {u0:g} c: the reply beats the message for β > {threshold:.3f}")
    print(f"at β = {beta_max:g}: reply returns at t = {loop['t_back'][-1].item():.3f} (message sent at t = 1)")

    # 4. the OPERA anomaly read as a tachyon, against SN1987A
    data = tensors.load_measurements()
    opera = data["OPERA (first result)"]
    mu = equation.mass_from_speed(opera["energy_gev"], opera["dv_over_c"])
    sn = data["SN1987A"]
    predicted = equation.speed_excess(sn["energy_gev"], mu)
    print(f"OPERA 2011 as a tachyon: |m| = {1000 * mu:.0f} MeV; it would make 10 MeV SN1987A neutrinos")
    print(f"  (v − c)/c = {predicted:.1f} instead of < {sn['err_plus']:.0e}: ruled out (and the anomaly was an error)")

    # 5. tachyon condensation: a field with m² < 0 rolls off φ = 0
    phi0 = tensors.noise_field(256, 1e-3, seed=0, device=dev)
    field = solve.condense(phi0, params["m2"], params["lam"], 60.0, 40.0, 0.02, params["friction"])
    v = equation.vacuum(params["m2"], params["lam"])
    print(
        f"condensation: true vacuum |φ| = {v:.3f}, V_min = {-(params['m2'] ** 2) / (4 * params['lam']):.3f} < V(0) = 0;"
    )
    print(f"  final ⟨|φ|⟩ = {field['phi'][-1].abs().mean().item():.3f}, ⟨V⟩ = {field['potential'][-1].item():.3f}")
    return plot.animate(speeds, betas, seen, u0, loop, data, mu, field, v, params, out_dir / "tachyons.gif")


if __name__ == "__main__":
    cli(run, __file__)
