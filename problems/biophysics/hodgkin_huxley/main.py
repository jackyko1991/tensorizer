"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    g, E = tensors.channels(dev)  # (3,), (3,) for Na, K, leak
    I_ext = tensors.stimulus_currents(params["I_max"], dev)  # (B,) µA/cm²
    phi = tensors.temperature_factor(params["temp"])
    y0 = tensors.initial_state(len(I_ext), dev)  # (4, B) = (V, m, h, n) at rest

    times, traj = solve.solve(y0, I_ext, g, E, tensors.C_M, phi, params["t_end"])  # (T,), (T, 4, B)
    spikes = solve.spike_times(times, traj[:, 0])
    rate = solve.firing_rate(spikes, params["t_end"])  # (B,) Hz

    n_spikes = [len(s) for s in spikes]
    fires = [b for b in range(len(I_ext)) if n_spikes[b] > 0]
    repeats = [b for b in range(len(I_ext)) if rate[b] > 0]
    print(
        f"{len(I_ext)} axons, I = 0 … {params['I_max']:g} µA/cm², T = {params['temp']:g} °C (φ = {phi:.2f}), "
        f"{len(times) - 1} RK4 steps of {times[1]:.3f} ms"
    )
    print(f"resting potential {traj[0, 0, 0]:.1f} mV, highest spike peak {traj[:, 0].max():.1f} mV")
    if fires:
        print(f"threshold for one spike: I ≈ {I_ext[fires[0]]:.2f} µA/cm²")
    if repeats:
        b = repeats[0]
        print(
            f"threshold for repetitive firing: I ≈ {I_ext[b]:.2f} µA/cm², starting at {rate[b]:.1f} Hz "
            "(it jumps from zero: a type II neuron)"
        )
    for b in range(0, len(I_ext), max(1, len(I_ext) // 8)):
        print(f"  I = {I_ext[b]:5.2f} µA/cm²: {n_spikes[b]:3d} spikes, steady rate {rate[b]:5.1f} Hz")
    return plot.animate(times, traj, I_ext, rate, n_spikes, out_dir / "hodgkin_huxley.gif")


if __name__ == "__main__":
    cli(run, __file__)
