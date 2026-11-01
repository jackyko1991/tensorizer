"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import venturi_flow_rate

R1, RHO = 0.05, 1000.0  # inlet radius (m), water density (kg/m³)


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    r2 = R1 / params["area_ratio"] ** 0.5  # throat radius from A1/A2
    p1 = params["p1"] * 1000  # inlet gauge pressure in Pa
    x = tensors.axis(400, dev)  # (N,)
    flow = solve.steady_flow(x, R1, r2, params["u1"], p1, RHO)  # dict of (N,) tensors
    x0, eta = tensors.tracers(7, 14, dev)  # (n,), (n,)
    times, tracer_x = solve.move_tracers(x0, R1, r2, flow["Q"], params["t_max"])

    u1, u2 = flow["u"][0].item(), flow["u"].max().item()
    dp = flow["p"][0].item() - flow["p"].min().item()
    A1, A2 = flow["A"][0].item(), flow["A"].min().item()
    print(f"flow rate Q = A1 u1 = {flow['Q'] * 1000:.3f} L/s")
    print(f"throat speed u2 = {u2:.3f} m/s,  u2/u1 = {u2 / u1:.3f},  A1/A2 = {A1 / A2:.3f}")
    print(f"pressure drop p1 - p2 = {dp:.0f} Pa  (Bernoulli: {0.5 * RHO * (u2**2 - u1**2):.0f} Pa)")
    print(f"Venturi meter: Q from the pressure drop = {venturi_flow_rate(dp, A1, A2, RHO) * 1000:.3f} L/s")
    if flow["p"].min() < 0:
        print("warning: the throat pressure is below atmospheric — raise p1 (a real throat could cavitate)")
    return plot.animate(x, flow, times, tracer_x, eta, R1, r2, out_dir / "bernoulli_venturi.gif")


if __name__ == "__main__":
    cli(run, __file__)
