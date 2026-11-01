"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    # Many tiny autograd calls in float64: the CPU is the right device here.
    M, p, e = tensors.M, params["p"], params["e"]
    E, L = tensors.orbit_constants(p, e)
    a = p / (1 - e**2)
    tau_max = params["n_orbits"] * 2 * math.pi * a**1.5  # roughly n Newtonian periods

    tau, gr = solve.geodesic(tensors.bound_orbit(p, e), M, tau_max, params["dt"])
    newton = solve.newton(tensors.newton_orbit(p, e), M, 2 * math.pi * a**1.5, params["dt"] / 4)
    peri = solve.periapsis_angles(gr)
    precession = float((peri.diff() - 2 * math.pi).mean()) if len(peri) > 1 else float("nan")
    rdot = gr[:, 0, 1, 1]
    peri_steps = torch.cat([torch.zeros(1, dtype=torch.long), torch.nonzero((rdot[:-1] < 0) & (rdot[1:] >= 0))[:, 0]])

    r0 = params["r_infall"]
    tau_p, plunge = solve.infall(tensors.radial_infall(r0), M, 0.05)
    tau_h = solve.proper_infall_time(r0, 2 * M, M)

    drift = (equation.norm(gr, M) + 1).abs().max()
    print(f"orbit p = {p:g} M, e = {e:g}:  E = {E:.5f}, L = {L:.4f} M;  |g u u + 1| ≤ {drift:.1e} along the orbit")
    print(f"perihelion advance {math.degrees(precession):.2f}°/orbit;  weak-field 6πM/p gives "
          f"{math.degrees(equation.weak_field_precession(p, M)):.2f}°")  # fmt: skip
    print(f"ISCO at r = {equation.isco_radius(M):g} M, photon sphere at r = {equation.photon_sphere_radius(M):g} M")
    print(f"plunge from r = {r0:g} M: reaches r = {float(plunge[-1, 0, 0, 1]):.2f} M after τ = {float(tau_p[-1]):.2f} "
          f"but t = {float(plunge[-1, 0, 0, 0]):.1f};  τ to the horizon (exact) = {tau_h:.2f} GM/c³")  # fmt: skip

    orbit = {
        "gr": gr,
        "newton": newton,
        "E": E,
        "L": L,
        "precession": precession,
        "weak_deg": math.degrees(equation.weak_field_precession(p, M)),
        "peri_steps": peri_steps.numpy(),
    }
    return plot.animate(orbit, {"tau": tau_p, "traj": plunge, "tau_horizon": tau_h}, M, out_dir / "black_hole.gif")


if __name__ == "__main__":
    cli(run, __file__)
