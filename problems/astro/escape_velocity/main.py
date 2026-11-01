"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    GM, R = tensors.GM, tensors.R
    v_esc = equation.escape_speed(GM, R)
    fractions = tensors.speed_fractions(params["n_rockets"], params["f_min"], params["f_max"])
    y0 = tensors.initial_state(fractions, v_esc, params["tilt"], dev)

    times, traj = solve.solve(y0, GM, R, params["t_max"], params["dt"])
    E = equation.energy(y0, GM)  # (B,) fixed by the launch
    gone = solve.escaped(traj, R)
    r_top = solve.highest_point(traj)
    r_max = equation.max_radius(fractions * v_esc, GM, R) if params["tilt"] == 0 else None

    print(f"v_esc = √(2GM/R) = {v_esc:.4f}")
    for b, f in enumerate(fractions.tolist()):
        if abs(E[b]) < 1e-5:
            fate = "E = 0: slows down forever, never returns"
        else:
            fate = "escapes" if gone[b] else ("falls back" if E[b] < 0 else "still climbing")
        print(f"v = {f:.2f} v_esc   E = {E[b]:+.3f}   highest r = {r_top[b]:6.2f} R   {fate}")
    GM_sun, c = 1.327e20, 2.998e8  # m³/s², m/s
    print(f"dark star: v_esc = c when R = 2GM/c², e.g. {equation.dark_star_radius(GM_sun, c) / 1e3:.2f} km for the Sun")
    return plot.animate(times, traj, fractions, r_max, R, out_dir / "escape_velocity.gif")


if __name__ == "__main__":
    cli(run, __file__)
