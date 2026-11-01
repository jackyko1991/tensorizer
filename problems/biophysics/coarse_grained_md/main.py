"""Main loop: native fold → Gō model → Langevin runs at many temperatures → Q(t) and the melting curve."""

from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import total_energy


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    native = tensors.native_structure(params["helix_length"], dev)  # (N, 3) Å
    model = tensors.go_model(native)
    n_temps, n_rep = params["temperatures"], 4  # 2 replicas start folded, 2 unfolded
    T_b = tensors.temperatures(params["T_min"], params["T_max"], n_temps, dev)  # (B,)
    T = T_b.repeat_interleave(n_rep)  # (S,) = (B·R,)
    x0 = tensors.initial_chains(native, n_temps, n_rep)  # (S, N, 3)
    print(
        f"Gō model: {native.shape[0]} beads, {int(model['native'].sum())} native contacts, "
        f"native energy {total_energy(native[None], model).item():.1f} ε"
    )
    print(f"{n_temps} temperatures × {n_rep} replicas = {len(T)} chains integrated together")

    n_steps = 600 if fast_mode() else params["steps"]
    dt = 0.02  # τ; the stiff bonds (period ≈ 0.44 τ) set the limit
    times, Q, frame_t, frames = solve.langevin(x0, T, model, dt, params["friction"], n_steps)
    q_nat, q_unf, q_all = solve.melting_curve(times, Q, n_temps, n_rep)
    print(f"simulated {n_steps} steps × dt {dt} = {n_steps * dt:.0f} τ")
    print("   T     ⟨Q⟩ from folded   ⟨Q⟩ from unfolded")
    for b in range(n_temps):
        print(f"  {T_b[b].item():.2f}      {q_nat[b].item():.2f}             {q_unf[b].item():.2f}")
    Tm = solve.melting_temperature(T_b, q_all)
    print(f"melting temperature (⟨Q⟩ = ½): T_m ≈ {Tm:.2f} ε/k_B")
    return plot.animate(native, times, Q, frame_t, frames, T_b, n_rep, Tm, out_dir / "coarse_grained_md.gif")


if __name__ == "__main__":
    cli(run, __file__)
