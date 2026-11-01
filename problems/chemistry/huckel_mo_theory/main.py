"""Main loop: carbon skeletons → H = αI + βA → batched eigh → filling, bond orders, 4n + 2 rule → figure."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import chain_levels, cyclic_levels, is_closed_shell, occupations, pi_energy


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    alpha, beta = tensors.ALPHA, tensors.BETA  # energies in units of |β| from α
    beta_kj = params["beta"]  # |β| in kJ/mol, only to convert the answers
    names, xys, A, pad, n_el = tensors.batch(dev)
    res = solve.solve(A, pad, n_el, alpha, beta)
    print(f"{len(names)} molecules in one batch, H of shape {tuple(A.shape)}; energies E = α + xβ, β < 0")
    print(f"{'molecule':15s} {'x of each MO (bonding first)':44s} {'gap/|β|':>8s} {'deloc./|β|':>10s} {'kJ/mol':>7s}")
    for b, name in enumerate(names):
        N = len(xys[b])
        x = ((res["E"][b, :N] - alpha) / beta).round(decimals=6) + 0.0  # E = α + xβ (+ 0.0 drops "-0.000")
        de = round((res["deloc"][b] / beta).item(), 6) + 0.0  # in units of β: positive = stabilised (β < 0)
        gap = round(res["gap"][b].item(), 6) + 0.0
        xs = " ".join(f"{v:+.3f}" for v in x.tolist())
        print(f"{name:15s} {xs:44s} {gap:8.3f} {de:10.3f} {de * beta_kj:7.0f}")

    b = names.index("butadiene")
    P = res["P"][b]
    print(f"butadiene π bond orders: C1–C2 = {P[0, 1]:.3f}, C2–C3 = {P[1, 2]:.3f}  (Coulson: 0.894, 0.447)")
    b = names.index("benzene")
    print(f"benzene π bond order {res['P'][b, 0, 1]:.3f} (2/3), π charges max |q| = {res['q'][b, :6].abs().max():.1e}")

    # Hückel's 4n + 2 rule: eigh of every ring against the closed form α + 2β cos(2πk/N)
    rings = []
    for N in range(3, params["ring_max"] + 1):
        E_num = torch.linalg.eigvalsh(alpha * torch.eye(N, dtype=torch.float64) + beta * tensors.ring_adjacency(N))
        E = cyclic_levels(N, alpha, beta)
        ne = tensors.ring_electrons(N)
        closed = is_closed_shell(E, ne)
        n = occupations(E[None], [ne])[0]
        chain = chain_levels(N, alpha, beta)
        e_ring = pi_energy(E[None], n[None])[0]
        e_chain = pi_energy(chain[None], occupations(chain[None], [ne]))[0]
        rings.append({"N": N, "ne": ne, "E": E, "n": n, "closed": closed, "ring_minus_chain": e_ring - e_chain})
        kind = "closed shell, aromatic" if closed else "half-filled degenerate pair, antiaromatic"
        print(
            f"ring N = {N:2d}, {ne:2d} π e⁻: max |eigh - closed form| = {(E_num - E).abs().max():.1e}, {kind}, "
            f"E_ring - E_chain = {(e_ring - e_chain).item():+.3f}|β|"
        )
    return plot.figure(names, xys, res, rings, params["molecule"], beta_kj, out_dir / "huckel_mo_theory.png")


if __name__ == "__main__":
    cli(run, __file__)
