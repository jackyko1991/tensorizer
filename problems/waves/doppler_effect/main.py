"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    c, f, t_max = tensors.C_SOUND, params["f"], params["t_max"]
    v_s = params["mach"] * c
    x_start = -0.5 * v_s * t_max  # keep the source's path centred on the origin

    t_e = tensors.emission_times(f, t_max, dev)  # (E,)
    s = tensors.source_position(t_e, v_s, x_start)  # (E, 2)
    obs = tensors.observers(x_start, v_s, t_max, c, dev)  # (2, 2)
    result = solve.solve(t_e, s, obs, c, t_max)

    heard = equation.heard_frequency(result["arrivals"]).tolist()
    ahead, behind = equation.doppler_formula(f, c, v_s)
    print(f"source speed v_s = {v_s:.0f} m/s  (Mach {params['mach']:.2f}),  emitted f = {f:.2f} Hz")
    print(f"ahead : heard {heard[0]:.3f} Hz   formula f c/|c - v_s| = {ahead:.3f} Hz")
    print(f"behind: heard {heard[1]:.3f} Hz   formula f c/(c + v_s)  = {behind:.3f} Hz")
    if params["mach"] > 1:
        mu = equation.cone_half_angle(s, result["radii"][-1], tensors.source_position(t_e.new_tensor(t_max), v_s, x_start))
        print(f"Mach cone half-angle {math.degrees(mu.mean().item()):.2f}°  (asin(c/v_s) = {math.degrees(math.asin(c / v_s)):.2f}°)")
        print("ahead of a supersonic source the fronts arrive in reverse order, all after the cone passes")
    return plot.animate(result, t_e, s, obs, v_s, c, out_dir / "doppler_effect.gif")


if __name__ == "__main__":
    cli(run, __file__)
