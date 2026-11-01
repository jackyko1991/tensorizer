"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import substitution_scores


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    local = params["mode"] == "local"
    go, ge = params["gap_open"], params["gap_extend"]  # a gap of k letters costs go + (k − 1)·ge
    W = tensors.substitution_matrix(params["match"], params["mismatch"], dev)  # (4, 4)
    make = tensors.planted_pairs if local else tensors.related_pairs
    data = make(params["n_pairs"], params["length"], params["mutation"], 0, dev)
    codes_a, len_a, codes_c, len_c = data[:4]  # (B, n), (B,), (B, m), (B,)
    B, n, m = codes_a.shape[0], codes_a.shape[1], codes_c.shape[1]
    print(f"{'Smith–Waterman (local)' if local else 'Needleman–Wunsch (global)'}: {B} pairs, up to {n} × {m} letters")
    print(f"scores: match {params['match']:+g}, mismatch {params['mismatch']:+g}, gap {go:g} + {ge:g} per extra letter")

    S = torch.zeros(B, n + 1, m + 1, device=dev)  # S_{bij}, row and column 0 unused
    S[:, 1:, 1:] = substitution_scores(tensors.one_hot(codes_a), W, tensors.one_hot(codes_c))
    H, E, F = solve.fill(S, go, ge, local)
    print(f"{n + m - 1} wavefront updates fill {B * n * m} cells (one per anti-diagonal instead of one per cell)")

    scores, ends = solve.best_cells(H, len_a, len_c, local)  # (B,), (B, 2)
    print(f"best scores: {[round(s) for s in scores.tolist()]}")
    a, c = tensors.decode(codes_a[0]), tensors.decode(codes_c[0])
    path, top, bottom = solve.traceback(H[0], E[0], F[0], S[0], a, c, tuple(ends[0].tolist()), go, ge, local)
    same = sum(x == y for x, y in zip(top, bottom))
    print(f"pair 0: score {scores[0]:.0f}, alignment length {len(top)}, identity {same}/{len(top)}")
    for k in range(0, len(top), 60):
        print(f"  {top[k : k + 60]}\n  {bottom[k : k + 60]}")
    if local:
        span = data[4][0].tolist()
        (i0, j0), (i1, j1) = path[0], path[-1]
        print(f"found a[{i0}:{i1}] ~ c[{j0}:{j1}]; motif planted at a[{span[0]}:{span[1]}], c[{span[2]}:{span[3]}]")

    done_at = len_a + len_c  # anti-diagonal on which each pair's table is complete
    return plot.animate(
        H[0, : len_a[0] + 1, : len_c[0] + 1], path, a, c, top, bottom, scores, done_at, local, out_dir / "alignment.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
