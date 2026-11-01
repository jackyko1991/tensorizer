"""Main loop: parameters → tensors → simulate the echo signal → spectrogram → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors

R_VESSEL = 3e-3  # m, artery radius
LENGTH_MM = 20.0  # length of vessel shown in the animation


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    f0 = params["f0"] * 1e6  # MHz → Hz
    prf = params["prf"] * 1e3  # kHz → Hz
    theta = math.radians(params["theta"])

    t = tensors.sample_times(prf, params["heart_rate"], 2, dev)  # (T,)
    v_max = tensors.cardiac_waveform(t, params["v_peak"], params["heart_rate"])  # (T,)
    scat = tensors.scatterers(params["n_scatterers"], R_VESSEL, LENGTH_MM, dev)

    v, s = solve.simulate(v_max, scat, R_VESSEL, f0, theta, prf)
    s = s + tensors.noise(len(s), 0.03 * math.sqrt(params["n_scatterers"]), dev)
    win_t, freqs, power = solve.spectrogram(s, prf)

    f_peak = equation.doppler_shift(v_max.max(), f0, theta).item()
    v_nyq = abs(prf / 2 * equation.C_TISSUE / (2 * f0 * math.cos(theta))) if math.cos(theta) > 1e-6 else math.inf
    print(f"{len(t)} pulses at PRF {prf / 1e3:.1f} kHz × {len(scat['r'])} scatterers → signal {tuple(s.shape)}")
    print(f"peak systolic velocity {v_max.max().item():.2f} m/s → Doppler shift {f_peak / 1e3:.2f} kHz")
    print(f"Nyquist limit PRF/2 = {prf / 2e3:.2f} kHz ↔ {v_nyq:.2f} m/s along the vessel")
    print("aliasing: peak wraps to the other side of the spectrum" if f_peak > prf / 2 else "no aliasing")
    return plot.animate(t, v, v_max, scat, win_t, freqs, power, f0, theta, prf, LENGTH_MM, out_dir / "doppler.gif")


if __name__ == "__main__":
    cli(run, __file__)
