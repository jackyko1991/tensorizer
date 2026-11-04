"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import overlap

SIZE = 32  # faces are SIZE × SIZE pixels, so N = 1024 neurons


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_faces, beta = params["n_faces"], params["beta"]
    faces = tensors.faces(n_faces, SIZE, 0, dev)  # (P, S, S)
    queries = tensors.corrupt(faces, params["mask"], params["noise"], 1)  # (P, S, S)
    x, q = faces.reshape(n_faces, -1), queries.reshape(n_faces, -1)  # (P, N)
    n = x.shape[1]
    sims = overlap(x[:, None], x[None])  # (P, P)
    off = sims[~torch.eye(n_faces, dtype=torch.bool, device=dev)]
    print(f"{n_faces} faces drawn from random parameters, {SIZE}×{SIZE} = {n} neurons")
    print(f"faces are correlated: mean overlap between two faces {off.mean():.2f}, largest {off.max():.2f}")

    loads = torch.arange(1, n_faces + 1, device=dev)  # networks storing the first 1, 2, …, P faces
    used = (torch.arange(n_faces, device=dev)[None] < loads[:, None]).float()  # (F, P)
    classic = solve.classic_recall(x, used, q, params["sweeps"], 2)  # (F, P, N)
    modern = solve.modern_recall(x, used, q, beta)  # (F, P, N)
    m_classic = (overlap(classic, x[None]) * used).sum(-1) / used.sum(-1)  # mean over the stored faces (F,)
    m_modern = (overlap(modern, x[None]) * used).sum(-1) / used.sum(-1)
    for k in sorted({1, 2, 4, len(loads) // 2, len(loads) - 1}):
        print(
            f"{int(loads[k]):3d} faces stored (α = {loads[k] / n:.3f}): mean recall overlap "
            f"classic {m_classic[k]:.3f}   modern (β = {beta}) {m_modern[k]:.3f}"
        )

    alphas = torch.tensor([0.02, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.2, 0.25, 0.3, 0.4, 0.6, 1.0, 1.5, 2.0])
    cap_c, cap_m, bit_err = solve.capacity_curve(alphas.to(dev), params["n_random"], 0.1, 8, params["sweeps"], 1.0, 3)
    print(f"random patterns, N = {params['n_random']}, 10 % of bits flipped:")
    for a, c, m, e in zip(alphas.tolist(), cap_c.tolist(), cap_m.tolist(), bit_err.tolist()):
        if round(a, 3) in (0.05, 0.1, 0.14, 0.2, 0.3, 1.0, 2.0):
            theory = 0.5 * math.erfc(1 / math.sqrt(2 * a))
            print(
                f"  α = {a:4.2f}: classic {c:.3f}  modern {m:.3f}   "
                f"one-step bit errors {e:.4f} (Φ(−1/√α) = {theory:.4f})"
            )
    print("the classic network fails past α ≈ 0.138 (and much earlier for correlated faces); the modern one does not")
    curve = (alphas, cap_c.cpu(), cap_m.cpu())
    return plot.animate(faces, queries, classic, modern, loads, m_classic, m_modern, curve, out_dir / "hopfield.gif")


if __name__ == "__main__":
    cli(run, __file__)
