"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    T1, T2, TE, n_echoes = params["T1"], params["T2"], params["TE"], params["n_echoes"]
    eps = tensors.levi_civita(dev)
    R, M0 = tensors.relaxation(T1, T2, dev)
    dw = tensors.off_resonance(params["n_spins"], params["t2_prime"], params["spread"], dev)
    M_init = tensors.equilibrium_state(params["n_spins"], dev)
    pulse_len = 0.2  # ms, a short "hard" pulse
    pulses = tensors.pulse_list(TE, n_echoes, pulse_len)
    t90 = pulse_len / 2

    t_max = t90 + (n_echoes + 0.4) * TE
    times, M = solve.solve(M_init, dw, eps, R, M0, pulses, t_max)

    t2_star = solve.fid_decay_time(times, M, t90, TE)
    print(f"T2 = {T2:.1f} ms,  T2' = {params['t2_prime']:.1f} ms,  measured T2* (FID to 1/e) = {t2_star:.2f} ms")
    t_echo, height = solve.echo_peaks(times, M, TE, n_echoes, t90)
    for k, (te, h) in enumerate(zip(t_echo.tolist(), height.tolist()), start=1):
        expected = math.exp(-k * TE / T2)
        print(f"echo {k}: t = {te - t90:6.2f} ms (expected {k * TE:.1f})  |s| = {h:.4f}  e^(-t/T2) = {expected:.4f}")
    return plot.animate(times, M, dw, pulses, T2, t2_star, t90, out_dir / "bloch_spin_echo.gif")


if __name__ == "__main__":
    cli(run, __file__)
