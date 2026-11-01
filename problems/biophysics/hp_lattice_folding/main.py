"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import contact_map, squared_distances


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    seq, e_ground = tensors.sequence(params["sequence"])
    h = tensors.hydrophobic(seq, dev)  # (N,)
    R = params["replicas"]
    n_steps = 300 if fast_mode() else params["steps"]
    x0 = tensors.straight_chains(R, len(seq), dev)  # (R, N, 2)
    temps = tensors.temperatures(params["T_start"], params["T_end"], n_steps, dev)  # (S,)
    print(f"sequence {seq} (N = {len(seq)}, {int(h.sum())} H), known ground state E = {e_ground}")
    print(f"{R} replicas × {n_steps} Monte Carlo steps, T from {params['T_start']} to {params['T_end']}")

    out = solve.anneal(x0, h, temps, n_records=60, seed=0)
    E, best_E = out["E"], out["best_E"]
    print(f"acceptance {out['acceptance']:.1%}")
    for k in (0, len(out["record_steps"]) // 4, len(out["record_steps"]) // 2, len(out["record_steps"]) - 1):
        e, s = out["energies"][k], out["record_steps"][k].item()
        print(f"step {s:5d}: mean E = {e.mean().item():6.2f}   lowest E = {int(e.min().item()):3d}")
    found = (best_E == e_ground).float().mean().item()
    print(
        f"lowest energy reached: {best_E.min().item():.0f} (ground state {e_ground}); {found:.0%} of replicas found it"
    )

    r = int(E.argmin())  # the replica that ends lowest is the one we watch fold
    C = contact_map(squared_distances(out["x"][r : r + 1]))[0]
    hh = [(i, j) for i in range(len(seq)) for j in range(i + 2, len(seq)) if C[i, j] and seq[i] == seq[j] == "H"]
    print(f"replica {r}: E = {E[r].item():.0f}, H–H contacts {hh}")
    return plot.animate(out, r, seq, e_ground, out_dir / "hp_lattice_folding.gif")


if __name__ == "__main__":
    cli(run, __file__)
